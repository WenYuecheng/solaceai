"""SolaceAI Chroma 向量数据库与 Embedding 服务 (Chroma Vector Store & Embedding Client)

本模块封装本地持久化向量数据库 ChromaDB 与外部/本地 Embedding 接口的交互能力：
1. 向量模型与接口适配：支持 OpenAI text-embedding-3 协议及兼容 API，内置批次控制与 Token 计费治理；
2. 向量库隔离与管理：集合级增删改查、HNSW 余弦距离空间索引配置、快照自动化备份与按需轮转归档；
3. 差量一致性同步 (`sync_chunks`)：比对关系型数据库中的切片与向量集合中的向量，安全更新最新切片并清理孤儿向量；
4. 优雅降级防护：缺少 API 密钥或依赖缺失时，平滑降级至词法检索（BM25），不阻断主交互流程。
"""

from __future__ import annotations

import os
import shutil

# 关闭 Chroma 匿名遥测，避免 posthog >= 3.0 版本中 posthog.capture() 方法签名不兼容导致的异常
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

import httpx
from app.services.conversation_access import paid_post

from app.core.config import Settings
from app.models.entities import KnowledgeChunk

# 检索路径标识标签
PRIMARY_RETRIEVAL_LABEL = "Chroma dense + BM25 hybrid"
FALLBACK_RETRIEVAL_LABEL = "local BM25 + lexical fallback ranking"

# 批量请求向量模型时的批次大小限制（防范超出网关单次请求限制）
EMBEDDING_BATCH_SIZE = 20


class VectorStoreUnavailable(RuntimeError):
    """向量数据库不可用或模型推理异常 (Vector Store Unavailable Error)"""
    pass


@dataclass
class VectorSearchHit:
    """向量检索单条命中结果数据传输对象 (Vector Search Hit DTO)

    Attributes:
        chunk_id: 知识切片自增主键 ID。
        source: 来源文档名称。
        source_index: 切片在原始文档中的序号。
        content: 切片文本正文内容。
        score: 余弦相似度归一化得分 (0.0~1.0)。
    """
    chunk_id: int | None
    source: str
    source_index: int
    content: str
    score: float


class KnowledgeVectorStore(Protocol):
    """知识向量存储统一接口协议 (Knowledge Vector Store Protocol)

    规范所有向量存储后端（Chroma / Milvus 等）必须实现的核心方法。
    """
    def upsert_chunks(self, chunks: list[KnowledgeChunk], embeddings=None, *, metadata: Callable | None = None) -> int:
        """批量插入或更新切片向量及其业务元数据。"""
        ...

    def delete_chunks(self, ids: list[int]) -> None:
        """根据切片主键 ID 列表从向量库中批量删除切片。"""
        ...

    def rebuild(self, chunks: list[KnowledgeChunk], embeddings=None, *, metadata: Callable | None = None) -> int:
        """清空向量集合并全量重新构建切片索引。"""
        ...


class ChromaKnowledgeStore:
    """基于 ChromaDB 的知识向量数据库封装 (Chroma Vector Store Client)

    负责切片向量的批量写入、差量同步、快照备份与多条件近邻检索。
    """

    def __init__(self, settings: Settings):
        """初始化 Chroma 向量知识库客户端

        执行环境依赖探测、目录初始化、API 密钥可用性校验以及持久化客户端实例创建。

        Args:
            settings (Settings): 系统全局配置对象。

        Raises:
            VectorStoreUnavailable: 当配置了强制要求向量检索 (knowledge_vector_required=True)
                且缺少必要密钥或依赖时抛出。
        """
        self.settings = settings
        self.can_embed = False
        self.error = ""

        # 1. 检查配置是否开启向量检索
        if not settings.knowledge_vector_enabled:
            self.error = "Chroma 向量库未启用"
            return

        # 2. 检查 Embedding 接口鉴权密钥
        if not settings.embedding_api_key:
            if settings.knowledge_vector_required:
                raise VectorStoreUnavailable("缺少 EMBEDDING_API_KEY，无法启用 Chroma 向量检索")
            self.error = f"缺少 EMBEDDING_API_KEY，Chroma 向量检索不可用，已回退到{FALLBACK_RETRIEVAL_LABEL}"
            return

        # 3. 动态导入 chromadb 依赖
        try:
            import chromadb
        except ImportError as exc:
            if settings.knowledge_vector_required:
                raise VectorStoreUnavailable("缺少 chromadb 依赖，无法启用 Chroma 向量检索") from exc
            self.error = f"缺少 chromadb 依赖，Chroma 向量检索不可用，已回退到{FALLBACK_RETRIEVAL_LABEL}"
            return

        # 4. 初始化持久化目录与客户端
        persist_dir = self._resolve_path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir = persist_dir

        from chromadb.config import Settings as ChromaSettings
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_model": settings.embedding_model},
        )
        self.can_embed = settings.knowledge_vector_enabled

    def upsert_chunks(
        self,
        chunks: list[KnowledgeChunk],
        embeddings: list[list[float]] | None = None,
        *,
        metadata: Callable | None = None,
    ) -> int:
        """向 Chroma 集合中插入或更新切片文档及其向量嵌入，并自动触发备份快照 (Upsert Chunks)

        Args:
            chunks (list[KnowledgeChunk]): 待写入的知识切片实体列表。
            embeddings (list[list[float]] | None, optional): 预先计算好的向量列表。若为 None 则实时调用模型计算。
            metadata (Callable | None, optional): 提取切片附加元数据的回调函数。

        Returns:
            int: 成功写入的切片数量。

        Raises:
            ValueError: 切片与向量数量不匹配或切片包含空内容时抛出。
        """
        if embeddings is None:
            embeddings = self.embed_texts([chunk.content for chunk in chunks]) if chunks else []
        if len(chunks) != len(embeddings) or any(chunk.id is None or not chunk.content.strip() for chunk in chunks):
            raise ValueError('切片与Embedding必须一一对应')
        rows = chunks
        if not rows:
            return 0
        ids = [self._id(chunk.id) for chunk in rows]
        documents = [chunk.content for chunk in rows]
        metadatas = [
            {
                "db_id": int(chunk.id),
                "source": chunk.source,
                "source_index": int(chunk.source_index),
                "document_id": getattr(chunk, 'document_id', '') or '',
                "tenant_id": getattr(chunk, 'tenant_id', 'default') or 'default',
                "knowledge_base_id": getattr(chunk, 'knowledge_base_id', 'default') or 'default',
                "version_id": getattr(chunk, 'version_id', None) or 0,
            }
            for chunk in rows
        ]
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        self.snapshot()
        return len(rows)

    def sync_chunks(self, chunks: list[KnowledgeChunk], embeddings: list[list[float]]) -> int:
        """与数据库全量切片进行差量同步 (Differential Vector Synchronization)

        确保关系库与向量库一致：写入全部成功后才清理孤儿向量，若写入失败则保留旧版本可检索。

        Args:
            chunks (list[KnowledgeChunk]): 关系数据库当前有效的全量切片列表。
            embeddings (list[list[float]]): 对应切片的特征向量列表。

        Returns:
            int: 成功同步的切片数量。
        """
        count = self.upsert_chunks(chunks, embeddings)
        valid_ids = {self._id(int(chunk.id)) for chunk in chunks if chunk.id is not None}
        current_ids = set(self.collection.get().get("ids", []))
        stale_ids = sorted(current_ids - valid_ids)
        if stale_ids:
            self.collection.delete(ids=stale_ids)
        return count

    def rebuild(
        self,
        chunks: list[KnowledgeChunk],
        embeddings: list[list[float]] | None = None,
        *,
        metadata: Callable | None = None,
    ) -> int:
        """全量重建向量索引 (Rebuild Entire Vector Index)

        灾备恢复入口：根据给定的全量切片刷新或生成向量并与当前集合执行完全覆盖同步。

        Args:
            chunks (list[KnowledgeChunk]): 待索引的切片实体列表。
            embeddings (list[list[float]] | None, optional): 预置的向量列表。
            metadata (Callable | None, optional): 附加元数据生成函数。

        Returns:
            int: 成功重建的索引条目数量。
        """
        if embeddings is None:
            embeddings = self.embed_texts([chunk.content for chunk in chunks]) if chunks else []
        return self.sync_chunks(chunks, embeddings)

    def delete_chunks(self, ids: list[int]) -> None:
        """根据切片主键 ID 列表从 Chroma 集合中物理移除对应向量 (Delete Chunks By IDs)

        Args:
            ids (list[int]): 待删除的切片 ID 列表。
        """
        if ids and self.can_embed:
            self.collection.delete(ids=[self._id(int(value)) for value in ids])

    def has_exact_chunk_ids(self, chunks: list[KnowledgeChunk]) -> bool:
        """比对 Chroma 集合中的向量 ID 集合与数据库切片 ID 集合是否完全一致 (Verify ID Consistency)

        Args:
            chunks (list[KnowledgeChunk]): 待核对的数据库切片列表。

        Returns:
            bool: 集合 ID 完全一致返回 True，存在孤儿或遗漏返回 False。
        """
        valid_ids = {self._id(int(chunk.id)) for chunk in chunks if chunk.id is not None}
        current_ids = set(self.collection.get().get("ids", []))
        return current_ids == valid_ids

    def delete_source(self, source: str) -> None:
        """根据文档来源名称删除 Chroma 中对应的所有向量切片 (Delete Vector Slices By Source)

        Args:
            source (str): 来源文件名或源文档标识。
        """
        if not self.can_embed:
            return
        self.collection.delete(where={"source": source})

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        *,
        active_ids: list[int] | None = None,
    ) -> list[VectorSearchHit]:
        """执行向量最近邻高维空间相似度检索 (Vector Nearest-Neighbor Search)

        基于 HNSW 索引执行余弦距离检索，支持在指定 `active_ids` 活跃切片集合内进行过滤过滤。

        Args:
            query_embedding (list[float]): 用户查询文本的特征嵌入向量。
            top_k (int): 期望召回的最相似切片数量。
            active_ids (list[int] | None, optional): 限定检索生效的切片 ID 集合（用于版本过滤）。

        Returns:
            list[VectorSearchHit]: 按相似度得分降序排列的命中结果对象列表。
        """
        if active_ids == []:
            return []
        options = {'where': {'db_id': {'$in': active_ids}}} if active_ids is not None else {}
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            **options,
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            hits.append(
                VectorSearchHit(
                    chunk_id=int(metadata["db_id"]) if metadata.get("db_id") is not None else None,
                    source=str(metadata.get("source", "")),
                    source_index=int(metadata.get("source_index", 0)),
                    content=document or "",
                    # 将余弦距离转换为 [0.0, 1.0] 相似度分数
                    score=1.0 / (1.0 + max(0.0, distance)),
                )
            )
        return hits

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量调用 Embedding 接口生成文本向量 (Batch Text Embedding)

        Args:
            texts (list[str]): 待生成向量的文本字符串列表。

        Returns:
            list[list[float]]: 每个文本对应的浮点向量列表。

        Raises:
            VectorStoreUnavailable: 向量化服务未启用或请求失败时抛出。
        """
        if not self.can_embed:
            raise VectorStoreUnavailable(self.error or "Chroma 向量检索不可用")
        return self._embed(texts)

    def snapshot(self) -> str | None:
        """对当前 Chroma 向量数据库目录生成时间戳快照归档并触发老旧快照清理 (Atomic Snapshot Backup)

        Returns:
            str | None: 归档快照的目录物理路径字符串；若不可用则返回 None。
        """
        if not self.can_embed:
            return None
        if not self.persist_dir.exists():
            return None
        snapshot_root = self._resolve_path(self.settings.chroma_snapshot_dir)
        snapshot_root.mkdir(parents=True, exist_ok=True)
        destination = snapshot_root / datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        shutil.copytree(self.persist_dir, destination)
        self._prune_snapshots(snapshot_root)
        return str(destination)

    def count(self) -> int:
        """获取当前 Chroma 集合中存储的高维向量索引实体总数量 (Get Collection Vector Count)

        Returns:
            int: 集合内向量总数。
        """
        if not self.can_embed:
            return 0
        return int(self.collection.count())

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """底层 HTTP 调用分批向 Embedding API 发送向量化请求 (Low-Level HTTP Embedding Request)

        集成 API 预算门禁与使用量计量提交 (`ApiBudget.commit`)。

        Args:
            texts (list[str]): 输入文本列表。

        Returns:
            list[list[float]]: 生成的浮点向量列表。

        Raises:
            VectorStoreUnavailable: 响应数量与请求不匹配或接口异常时抛出。
        """
        headers = {"Authorization": f"Bearer {self.settings.embedding_api_key}"}
        embeddings = []
        normalized_texts = [text if text.strip() else " " for text in texts]

        for start in range(0, len(normalized_texts), EMBEDDING_BATCH_SIZE):
            batch = normalized_texts[start:start + EMBEDDING_BATCH_SIZE]
            from app.services.api_budget import ApiBudget, guard_remote

            # 1. 执行 API 预算前置扣减预留
            reservation = guard_remote(
                self.settings,
                "embedding",
                self.settings.embedding_model,
                self.settings.embedding_base_url,
                sum(len(text.encode("utf-8")) + 64 for text in batch) + 256,
            )

            # 2. 发起 HTTP 向量推理请求
            response = paid_post(
                f"{self.settings.embedding_base_url.rstrip('/')}/embeddings",
                headers=headers,
                json={
                    "model": self.settings.embedding_model,
                    "input": batch,
                    "dimensions": self.settings.embedding_dimensions,
                },
                timeout=self.settings.embedding_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()

            # 3. 提交真实消耗的 Token 计量
            usage = payload.get("usage", {})
            if reservation and isinstance(usage.get("prompt_tokens"), int):
                ApiBudget(self.settings).commit(reservation, usage["prompt_tokens"])

            # 4. 按返回 index 排序以确保顺序严格匹配
            rows = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
            embeddings.extend(row.get("embedding") for row in rows)

        if len(embeddings) != len(texts) or any(not embedding for embedding in embeddings):
            raise VectorStoreUnavailable("OpenAI embeddings 接口返回向量数量不匹配")
        return [[float(value) for value in embedding] for embedding in embeddings]

    def _resolve_path(self, value: str) -> Path:
        """解析路径配置字符串：若为相对路径则锚定至项目根目录，若为绝对路径则保持原样。"""
        path = Path(value)
        return path if path.is_absolute() else self.settings.project_root / path

    def _prune_snapshots(self, snapshot_root: Path) -> None:
        """根据配置的 `chroma_snapshot_keep` 数量自动淘汰最早的快照目录 (Prune Old Snapshots)"""
        keep = max(1, self.settings.chroma_snapshot_keep)
        snapshots = sorted([path for path in snapshot_root.iterdir() if path.is_dir()], reverse=True)
        for stale in snapshots[keep:]:
            shutil.rmtree(stale, ignore_errors=True)

    def _id(self, chunk_id: int) -> str:
        """根据知识切片数字主键 ID 构造 ChromaDB 唯一的文本主键标识符。"""
        return f"knowledge-chunk-{chunk_id}"


# 别名映射，保持系统向后兼容性
ChromaKnowledgeVectorStore = ChromaKnowledgeStore
