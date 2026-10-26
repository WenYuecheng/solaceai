"""SolaceAI 多级记忆体系与短期对话缓存服务 (Multi-Tier Memory & Conversation Cache)

本模块构建并管理心理咨询对话的多层次上下文记忆生态：
1. Redis 短期会话快照 (`RedisShortTermMemoryStore`)：针对高频对话消息提供带滑动窗口截断的高性能缓存；
2. 关系型数据库持久化记忆 (`SqlAgentStoreItemService`)：针对学生显式授权的长期事实进行持久化与租户隔离；
3. LangGraph 状态图检查点与工作记忆 (`AgentMemoryStore`)：负责多轮智能体调用中的记忆提取与匹配；
4. 严格隐私与授权保护：仅在学生明确指令授权时触发长期记忆沉淀，并执行 PII 脱敏过滤。
"""

from __future__ import annotations

import json
import re
import logging
import threading
import hashlib
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

from sqlalchemy.orm import Session

from langgraph.store.base import BaseStore, PutOp
from langgraph.store.memory import InMemoryStore

from app.core.config import Settings
from app.models.entities import AgentStoreItem
from app.services.privacy import PrivacySanitizer

logger = logging.getLogger(__name__)


def explicit_memory_content(text: str) -> str | None:
    """从学生输入中解析并提取显式授权的长期记忆写入指令 (Extract Explicit Memory Directive)

    安全机制：
    1. 遵循最小授权原则，仅在子句或标点边界显式出现“请记住”、“帮我记住”等直接授权动词时触发写入；
    2. 严格排除否定语境（'不要记住', '别记住'）与引用转述语境（引号、'引用'、'转述'），防范提示词注入与越权记忆污染。

    Args:
        text (str): 用户输入的原始对话文本。

    Returns:
        str | None: 提取出的待记忆关键事实文本；若未授权或语境不合规返回 None。
    """
    # 仅在句首或标点边界的直接明确请求方赋予写入长期记忆的授权
    match = re.search(r'(?:^|[，,。！!？?；;\n：:]|对了[，,]?)\s*(?:请|请你|帮我|请帮我)?记住[：:\s]*(.+)', text)
    if not match:
        match = re.search(r'(?:请你|请帮我|帮我|请)记住[：:\s]*(.+)', text)
    if not match or any(mark in text[:match.end()] for mark in ('不要记住', '别记住', '不必记住', '不用记住', '引用', '转述', '“', '「', '"')):
        return None
    return match.group(1).strip() or None


_CONTEXT_LOCKS = [threading.RLock() for _ in range(64)]


def context_lock(scope: Any) -> threading.RLock:
    """根据作用域对象哈希散列获取分段互斥锁 (Get Sharded Context Reentrant Lock)

    采用 64 槽位的哈希分段锁设计，降低不同用户/会话在并发写状态时的锁竞争。

    Args:
        scope (Any): 锁作用域对象（如 session_id, user_id 等）。

    Returns:
        threading.RLock: 分配给该作用域的可重入互斥锁。
    """
    digest = hashlib.sha256(repr(scope).encode()).digest()
    return _CONTEXT_LOCKS[int.from_bytes(digest[:2], "big") % len(_CONTEXT_LOCKS)]


_REDIS_CLIENT_SINGLETON = None
_REDIS_COOLDOWN_UNTIL = 0.0


class RedisShortTermMemoryStore:
    """基于 Redis 的短期对话记忆快照与窗口管理组件 (Redis History Snapshot Store)"""

    def __init__(self, settings: Settings):
        """初始化 Redis 短期记忆存储组件"""
        self.settings = settings
        self.privacy = PrivacySanitizer()
        self._client = None

    @classmethod
    def is_in_cooldown(cls) -> bool:
        """检查 Redis 是否处于故障 5 秒冷却熔断期"""
        global _REDIS_COOLDOWN_UNTIL
        return time.monotonic() < _REDIS_COOLDOWN_UNTIL

    @classmethod
    def trip_cooldown(cls, seconds: float = 5.0) -> None:
        """触发 Redis 故障熔断冷却期（默认 5 秒内不再重试访问）"""
        global _REDIS_COOLDOWN_UNTIL
        _REDIS_COOLDOWN_UNTIL = time.monotonic() + seconds

    @property
    def client(self):
        """延迟初始化并复用 Redis 客户端连接（无同步 ping 阻塞）"""
        if self._client is not None:
            return self._client
        if not getattr(self.settings, "redis_history_cache_enabled", True):
            return None
        if not getattr(self.settings, "redis_url", ""):
            return None
        if self.is_in_cooldown():
            return None

        global _REDIS_CLIENT_SINGLETON
        if _REDIS_CLIENT_SINGLETON is not None:
            return _REDIS_CLIENT_SINGLETON

        try:
            redis_module = import_module("redis")
            timeout = getattr(self.settings, "redis_socket_timeout_seconds", 0.2) or 0.2
            client = redis_module.Redis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                socket_timeout=timeout,
                socket_connect_timeout=timeout,
            )
            _REDIS_CLIENT_SINGLETON = client
            return _REDIS_CLIENT_SINGLETON
        except Exception as exc:
            self.trip_cooldown(5.0)
            logger.warning("Redis client initialization failed: %s", exc)
            return None

    def _key(self, user_id_or_session: int | str, session_public_id: str | None = None) -> str:
        """生成带业务命名空间前缀的版本化 Redis 键名 (Namespaced Key)"""
        if session_public_id is None:
            return f"{self.settings.redis_namespace}:short-term-memory:{user_id_or_session}"
        return f"{self.settings.redis_namespace}:history:v2:{user_id_or_session}:{session_public_id}"

    def load(self, user_id: int, session_id: str, latest_message_id: int) -> dict | None:
        """从 Redis 读取有效完整的会话历史快照，若水位不一致或异常则返回 None (Load Valid Snapshot)"""
        if not getattr(self.settings, "redis_history_cache_enabled", True):
            return None
        client = self.client
        if client is None:
            return None

        key = self._key(user_id, session_id)
        try:
            raw = client.get(key)
        except Exception as exc:
            self.trip_cooldown(5.0)
            logger.warning("Redis history cache load failed: %s", exc)
            return None

        if raw is None:
            return None

        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return None
            if data.get("schemaVersion") != 2:
                return None
            if data.get("userId") != user_id or data.get("sessionId") != session_id:
                return None
            if not data.get("complete"):
                return None
            if data.get("lastMessageId") != latest_message_id:
                return None
            messages = data.get("messages")
            if not isinstance(messages, list):
                return None
            if len(messages) > self.settings.redis_memory_max_messages:
                return None

            # 校验消息有序性与字段结构
            prev_id = -1
            for item in messages:
                if not isinstance(item, dict):
                    return None
                item_id = item.get("id")
                if not isinstance(item_id, int) or item_id <= prev_id:
                    return None
                if not isinstance(item.get("role"), str) or not isinstance(item.get("content"), str):
                    return None
                prev_id = item_id

            if messages and prev_id != latest_message_id:
                return None
            if not messages and latest_message_id != 0:
                return None

            return data
        except Exception as exc:
            logger.warning("Redis history cache payload invalid: %s", exc)
            return None

    def store(self, user_id: int, session_id: str, messages: list[dict]) -> bool:
        """仅当消息总数未超限且快照体积未超限时，原子保存完整脱敏快照至 Redis (Store Snapshot)"""
        if not getattr(self.settings, "redis_history_cache_enabled", True):
            return False
        if len(messages) > self.settings.redis_memory_max_messages:
            return False
        client = self.client
        if client is None:
            return False

        last_message_id = messages[-1]["id"] if messages else 0
        snapshot = {
            "schemaVersion": 2,
            "userId": user_id,
            "sessionId": session_id,
            "lastMessageId": last_message_id,
            "complete": True,
            "messages": [
                {
                    "id": m["id"],
                    "role": m["role"].lower(),
                    "content": self.privacy.sanitize(m["content"]),
                    "createdAt": m.get("createdAt") or "",
                }
                for m in messages
            ],
        }

        try:
            serialized = json.dumps(snapshot, ensure_ascii=False)
            if len(serialized.encode("utf-8")) > 256 * 1024:
                return False
            key = self._key(user_id, session_id)
            client.set(key, serialized, ex=self.settings.redis_memory_ttl_seconds)
            return True
        except Exception as exc:
            self.trip_cooldown(5.0)
            logger.warning("Redis history cache store failed: %s", exc)
            return False

    def evict(self, user_id: int, session_id: str) -> bool:
        """从 Redis 主动驱逐指定用户的会话快照 (Evict Snapshot)"""
        if not getattr(self.settings, "redis_history_cache_enabled", True):
            return False
        client = self.client
        if client is None:
            return False
        key = self._key(user_id, session_id)
        try:
            client.delete(key)
            return True
        except Exception as exc:
            self.trip_cooldown(5.0)
            logger.warning("Redis history cache evict failed: %s", exc)
            return False

    def append(self, session_public_id: str, role: str, content: str) -> None:
        """向后兼容保留的旧版追加接口桩（新架构改由 commit 后统一 evict）"""
        pass

    def _serialize(self, role: str, content: str) -> str:
        """格式化脱敏消息体为 JSON 字符串（兼容既有单测）"""
        return json.dumps(
            {
                "role": role.lower(),
                "content": self.privacy.sanitize(content),
                "createdAt": datetime.utcnow().isoformat(),
            },
            ensure_ascii=False,
        )


class AgentMemoryStore:
    """基于独立 SQLite 文件的历史长期偏好与增量摘要存储组件 (Legacy SQLite Memory Store)"""

    def __init__(self, path: str):
        """初始化 SQLite 记忆存储并自举创建必要数据表"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS preferences(id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, session_id TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS summaries(user_id INTEGER, session_id TEXT, covered INTEGER, content TEXT, PRIMARY KEY(user_id,session_id))"
            )

    def _connect(self) -> sqlite3.Connection:
        """创建配置了行映射字典工厂 (Row Factory) 的 SQLite 连接对象"""
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def remember_explicit(self, user_id: int, session_id: str, text: str) -> int | None:
        """解析用户显式授权指令并安全持久化长期事实/偏好 (Explicit Memory Ingestion)"""
        raw = explicit_memory_content(text)
        if raw is None:
            return None
        content = PrivacySanitizer().sanitize(raw)[:1000]
        if not content:
            return None
        with self._connect() as conn:
            previous = conn.execute(
                "SELECT id FROM preferences WHERE user_id=? AND content=?",
                (user_id, content),
            ).fetchone()
            if previous:
                return previous["id"]
            return conn.execute(
                "INSERT INTO preferences(user_id,session_id,content,created_at) VALUES(?,?,?,?)",
                (user_id, session_id, content, datetime.now(timezone.utc).isoformat()),
            ).lastrowid

    def list(self, user_id: int) -> list[dict]:
        """获取指定用户当前保存的全部长期偏好记忆记录列表 (List Memories)"""
        with self._connect() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM preferences WHERE user_id=? ORDER BY id DESC LIMIT 100",
                    (user_id,),
                )
            ]

    def search(self, user_id: int, query: str, limit: int = 3) -> list[dict]:
        """基于大模型语义理解检索匹配与当前问题最相关的长期偏好记忆 (Semantic Memory Search)"""
        from app.services.context import select_context

        memories = self.list(user_id)
        client = getattr(self, "selector", None)
        if not client or not memories:
            return []
        selection = select_context(client, query, [], [], memories, memory_search=True)
        by_id = {m["id"]: m for m in memories}
        return [by_id[i] for i in selection.selected_memory_ids][: min(3, limit)]

    def delete(self, user_id: int, memory_id: int) -> bool:
        """删除指定用户归属的特定长期偏好记忆记录 (Delete Memory)"""
        with self._connect() as conn:
            return (
                conn.execute(
                    "DELETE FROM preferences WHERE id=? AND user_id=?",
                    (memory_id, user_id),
                ).rowcount
                > 0
            )

    def working_context(
        self,
        user_id: int,
        session_id: str,
        messages: list,
        context_tokens: int = 8192,
        summarizer=None,
        **policy_options,
    ) -> tuple[str, list]:
        """装配并恢复指定会话的工作上下文滑动窗口与增量摘要 (Assemble Working Context)"""
        from app.services.context import WorkingContext

        expected = [None]

        def load():
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT covered,content FROM summaries WHERE user_id=? AND session_id=?",
                    (user_id, session_id),
                ).fetchone()
                expected[0] = dict(row) if row else None
                return expected[0]

        def save(value):
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT covered,content FROM summaries WHERE user_id=? AND session_id=?",
                    (user_id, session_id),
                ).fetchone()
                if (dict(row) if row else None) != expected[0]:
                    raise RuntimeError("Summary checkpoint changed concurrently")
                conn.execute(
                    "INSERT OR REPLACE INTO summaries VALUES(?,?,?,?)",
                    (user_id, session_id, value["covered"], value["content"]),
                )
            expected[0] = dict(value)

        with context_lock((self.path, user_id, session_id)):
            return WorkingContext(load, save).prepare(
                messages, context_tokens, summarizer, **policy_options
            )


class SQLBusinessStore(BaseStore):
    """基于业务关系型数据库 (SQLAlchemy) 的 LangGraph 状态与命名空间存储适配器 (SQL Business Store)"""

    def __init__(self, db: Session):
        """初始化 SQL 业务存储适配器"""
        self.db = db

    def batch(self, ops: list) -> list:
        """批量同步执行 LangGraph Store 操作（支持 Get/Put/Delete）"""
        operations = list(ops)
        snapshot = InMemoryStore()
        rows = self.db.query(AgentStoreItem).all()
        for row in rows:
            snapshot.put(
                tuple(json.loads(row.namespace)), row.key, json.loads(row.value_json)
            )
        results = snapshot.batch(operations)
        for op in operations:
            if not isinstance(op, PutOp):
                continue
            namespace = json.dumps(op.namespace, ensure_ascii=False)
            row = (
                self.db.query(AgentStoreItem)
                .filter_by(namespace=namespace, key=op.key)
                .first()
            )
            if op.value is None:
                if row:
                    self.db.delete(row)
            elif row:
                row.value_json = json.dumps(op.value, ensure_ascii=False)
            else:
                self.db.add(
                    AgentStoreItem(
                        namespace=namespace,
                        key=op.key,
                        value_json=json.dumps(op.value, ensure_ascii=False),
                    )
                )
        if any(isinstance(op, PutOp) for op in operations):
            self.db.commit()
        return results

    async def abatch(self, ops: list) -> list:
        """异步批量执行 LangGraph Store 操作（直接复用当前会话，避免跨线程共享 Session）"""
        # 调用方持有该数据库 Session；严禁跨线程共享传递以确保线程安全性
        return self.batch(ops)


class SQLAgentMemoryStore(AgentMemoryStore):
    """企业级统一命名空间长期记忆管理服务 (Enterprise SQL Agent Memory Store)"""

    def __init__(self, db: Session, legacy_path: str | None = None):
        """初始化统一 SQL 记忆服务并按需执行历史迁移"""
        self.store = SQLBusinessStore(db)
        self.db = db
        if (
            legacy_path
            and Path(legacy_path).is_file()
            and self.store.get(("migration",), "legacy-memory") is None
        ):
            with sqlite3.connect(legacy_path) as connection:
                connection.row_factory = sqlite3.Row
                for row in connection.execute("SELECT * FROM preferences"):
                    value = dict(row)
                    key = str(value["id"])
                    namespace = ("users", str(value["user_id"]), "preferences")
                    if self.store.get(namespace, key) is None:
                        self.store.put(namespace, key, value)
                for row in connection.execute("SELECT * FROM summaries"):
                    value = dict(row)
                    namespace = ("users", str(value["user_id"]), "summaries")
                    if self.store.get(namespace, value["session_id"]) is None:
                        self.store.put(namespace, value["session_id"], value)
            # 在存储区中设置迁移标记，防止多次重复迁移导致已被用户删除的历史记忆被意外复活
            self.store.put(("migration",), "legacy-memory", {"done": True})

    def remember_explicit(self, user_id: int, session_id: str, text: str) -> int | None:
        """解析并保存用户显式授权记录的偏好或事实到命名空间记忆中 (Explicit Memory Ingestion)"""
        raw = explicit_memory_content(text)
        if raw is None:
            return None
        content = PrivacySanitizer().sanitize(raw)[:1000]
        if not content:
            return None
        for row in self.list(user_id):
            if row["content"] == content:
                return row["id"]
        namespace = ("users", str(user_id), "preferences")
        memory_id = uuid.uuid4().int >> 96
        self.store.put(
            namespace,
            str(memory_id),
            {
                "id": memory_id,
                "user_id": user_id,
                "session_id": session_id,
                "content": content,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return memory_id

    def list(self, user_id: int) -> list[dict]:
        """获取指定用户在当前数据库中存储的全部长期记忆偏好 (List Memories)"""
        items = self.store.search(("users", str(user_id), "preferences"), limit=100)
        return sorted(
            [item.value for item in items],
            key=lambda row: row["created_at"],
            reverse=True,
        )

    def delete(self, user_id: int, memory_id: int) -> bool:
        """从命名空间中删除特定记忆项并校验所有权 (Delete Memory)"""
        namespace = ("users", str(user_id), "preferences")
        if self.store.get(namespace, str(memory_id)) is None:
            return False
        self.store.delete(namespace, str(memory_id))
        return True

    def working_context(
        self,
        user_id: int,
        session_id: str,
        messages: list,
        context_tokens: int = 8192,
        summarizer=None,
        **policy_options,
    ) -> tuple[str, list]:
        """装配并恢复指定会话的工作上下文滑动窗口与增量摘要 (Assemble Working Context)"""
        from app.services.context import WorkingContext

        namespace = ("users", str(user_id), "summaries")

        encoded_namespace = json.dumps(namespace, ensure_ascii=False)
        expected = [None]

        def load():
            row = (
                self.db.query(AgentStoreItem.value_json)
                .filter_by(namespace=encoded_namespace, key=session_id)
                .first()
            )
            expected[0] = row[0] if row else None
            return json.loads(expected[0]) if row else None

        def save(value):
            encoded = json.dumps(value, ensure_ascii=False)
            try:
                if expected[0] is None:
                    self.db.add(
                        AgentStoreItem(
                            namespace=encoded_namespace,
                            key=session_id,
                            value_json=encoded,
                        )
                    )
                    self.db.flush()
                else:
                    changed = (
                        self.db.query(AgentStoreItem)
                        .filter_by(
                            namespace=encoded_namespace,
                            key=session_id,
                            value_json=expected[0],
                        )
                        .update({"value_json": encoded}, synchronize_session=False)
                    )
                    if changed != 1:
                        raise RuntimeError("Summary checkpoint changed concurrently")
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise
            expected[0] = encoded

        with context_lock((str(self.db.get_bind().url), user_id, session_id)):
            return WorkingContext(load, save).prepare(
                messages, context_tokens, summarizer, **policy_options
            )
