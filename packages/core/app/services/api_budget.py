"""SolaceAI 远程付费 API 持久化预算管控服务 (API Budget Service)"""
from __future__ import annotations

import ipaddress
import math
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo


class BudgetExceeded(RuntimeError):
    """当超出配置的付费 API 预算额度或缺少定价配置时抛出的异常。"""
    pass


def is_local_endpoint(url: str) -> bool:
    """判断指定 URL 是否为本机回环或容器内部免计费地址。"""
    host = urlsplit(url).hostname or ''
    if host in {'localhost', 'host.docker.internal'}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class ApiBudget:
    """远程付费 API 调用持久化预算管控服务（基于 SQLite 强一致性锁）。"""

    def __init__(self, settings):
        """初始化预算管控服务，建立存储计费流水的 SQLite 数据表。"""
        self.settings = settings
        configured=getattr(settings,'api_budget_cny',None)
        self.limit=None if configured is None else max(0.0,float(configured))
        if self.limit is not None and not math.isfinite(self.limit):
            raise BudgetExceeded('Invalid configured spending limit')
        daily = getattr(settings, 'api_daily_budget_cny', None)
        self.daily_limit = None if daily is None else float(daily)
        if self.daily_limit is not None and (not math.isfinite(self.daily_limit) or self.daily_limit < 0):
            raise BudgetExceeded('Invalid configured daily spending limit')
        path = Path(getattr(settings, 'api_budget_path', 'data/api-budget.sqlite'))
        if not path.is_absolute():
            path = Path(getattr(settings, 'project_root', Path.cwd())) / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with self._connect() as db:
            # Serialize both schema detection and migration across worker processes.
            db.execute('BEGIN IMMEDIATE')
            db.execute('CREATE TABLE IF NOT EXISTS api_charges (id TEXT PRIMARY KEY, kind TEXT, model TEXT, amount REAL NOT NULL, settled INTEGER NOT NULL DEFAULT 0, budget_day TEXT NOT NULL)')
            columns = {row[1] for row in db.execute('PRAGMA table_info(api_charges)')}
            if 'budget_day' not in columns:
                db.execute('ALTER TABLE api_charges ADD COLUMN budget_day TEXT')
                # Historical rows have no timestamp: charge them to migration day
                # rather than silently granting another full daily allowance.
                db.execute('UPDATE api_charges SET budget_day=?', (self._day(),))
            db.execute('CREATE INDEX IF NOT EXISTS api_charges_day ON api_charges(budget_day)')

    @staticmethod
    def _day() -> str:
        """Return the billing day in China, independent of server timezone."""
        return datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()

    def _connect(self) -> sqlite3.Connection:
        """获取 SQLite 连接，设置 30 秒超时以应对并发排队。"""
        return sqlite3.connect(self.path, timeout=30)

    def _cost(self, kind: str, input_units: int, output_units: int) -> float:
        """根据调用类型与计费单价精确计算单次调用花费（元）。"""
        if input_units < 0 or output_units < 0:
            raise BudgetExceeded('Invalid usage')
        prefix = 'chat_input' if kind == 'chat' else kind
        rate = float(getattr(self.settings, f'api_price_{prefix}_cny_per_million', 0))
        out_rate = float(getattr(self.settings, 'api_price_chat_output_cny_per_million', 0)) if kind == 'chat' else 0
        if kind not in {'chat', 'embedding', 'rerank'} or not math.isfinite(rate) or rate <= 0 or (kind == 'chat' and (not math.isfinite(out_rate) or out_rate <= 0)):
            raise BudgetExceeded('Remote API pricing is missing; call blocked')
        return (input_units * rate + output_units * out_rate) / 1_000_000

    def reserve(self, kind: str, model: str, input_units: int, max_output_units: int = 0) -> str:
        """发起远程调用前预先锁定预算额度。"""
        if not getattr(self.settings, 'paid_api_enabled', False):
            raise BudgetExceeded('Paid API calls are disabled; use offline mode')
        amount = self._cost(kind, input_units, max_output_units)
        key = uuid.uuid4().hex
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            day = self._day()
            used, daily_used = db.execute(
                'SELECT COALESCE(SUM(amount),0), COALESCE(SUM(CASE WHEN budget_day=? THEN amount ELSE 0 END),0) FROM api_charges',
                (day,),
            ).fetchone()
            if self.limit is not None and used + amount > self.limit:
                raise BudgetExceeded('Configured API spending limit would be exceeded')
            if self.daily_limit is not None and daily_used + amount > self.daily_limit:
                raise BudgetExceeded('Configured daily API spending limit would be exceeded')
            db.execute('INSERT INTO api_charges(id,kind,model,amount,budget_day) VALUES (?,?,?,?,?)', (key,kind,model,amount,day))
        return key

    def commit(self, key: str, input_units: int, output_units: int = 0) -> None:
        """远程调用成功后，依据服务端返回的真实用量结算并更新实际支出。"""
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT kind,amount,settled FROM api_charges WHERE id=?', (key,)).fetchone()
            if row is None or row[2]:
                return
            actual = self._cost(row[0], input_units, output_units)
            # 超出预期的用量无法追溯撤销供应商扣费；记录实际用量，阻止后续超额请求，保留保守预留。
            db.execute('UPDATE api_charges SET amount=?,settled=1 WHERE id=?', (actual,key))

    def status(self) -> dict:
        """查询当前预算总额、已结算支出与未结算预扣总额。"""
        with self._connect() as db:
            day = self._day()
            spent, reserved, daily_spent, daily_reserved = db.execute(
                'SELECT COALESCE(SUM(CASE WHEN settled=1 THEN amount ELSE 0 END),0),'
                'COALESCE(SUM(CASE WHEN settled=0 THEN amount ELSE 0 END),0),'
                'COALESCE(SUM(CASE WHEN settled=1 AND budget_day=? THEN amount ELSE 0 END),0),'
                'COALESCE(SUM(CASE WHEN settled=0 AND budget_day=? THEN amount ELSE 0 END),0) FROM api_charges',
                (day, day),
            ).fetchone()
        return dict(limit_cny=self.limit, spent_cny=spent, reserved_cny=reserved,
                    remaining_cny=None if self.limit is None else max(0,self.limit-spent-reserved),
                    daily_limit_cny=self.daily_limit, daily_spent_cny=daily_spent,
                    daily_reserved_cny=daily_reserved, budget_day=day, budget_timezone='Asia/Shanghai',
                    daily_remaining_cny=None if self.daily_limit is None else max(0,self.daily_limit-daily_spent-daily_reserved),
                    enabled=bool(getattr(self.settings,'paid_api_enabled',False)))


def guard_remote(settings, kind: str, model: str, url: str, input_units: int, max_output_units: int = 0) -> str | None:
    """针对外部远程端点的预算看门狗；若端点为本地回环则直接放行。"""
    from app.services.conversation_access import check_turn
    check_turn()
    if is_local_endpoint(url):
        return None
    if not getattr(settings, 'paid_api_enabled', False):
        raise BudgetExceeded('Remote API calls are disabled; use offline mode')
    if getattr(settings, 'api_budget_cny', None) is None and getattr(settings, 'api_daily_budget_cny', None) is None:
        return None
    return ApiBudget(settings).reserve(kind, model, input_units, max_output_units)


def conservative_input_units(messages) -> int:
    """保守估算输入消息序列的计费单位（UTF-8 字节加协议信封开销）。"""
    # UTF-8 字节数通常大于等于常规分词器 Token 计数；额外补充 64 字节消息包装信封开销。
    return sum(len(str(getattr(m,'content',m)).encode('utf-8')) + 64 for m in messages) + 256
