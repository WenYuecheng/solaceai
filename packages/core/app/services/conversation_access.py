"""Durable per-account admission, shared with the mounted API cost ledger.

Admissions count even when generation fails. A lease is released only after its
worker has stopped; process death recovers after the enforced turn deadline.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from zoneinfo import ZoneInfo

from fastapi import HTTPException
import httpx

logger = logging.getLogger(__name__)
_current_turn: ContextVar[TurnLease | None] = ContextVar('solace_paid_turn', default=None)


class ConversationAccess:
    def __init__(self, settings):
        path = Path(settings.api_budget_path)
        if not path.is_absolute():
            path = Path(getattr(settings, 'project_root', Path.cwd())) / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.timeout = float(getattr(settings, 'conversation_timeout_seconds', 180))
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('CREATE TABLE IF NOT EXISTS conversation_admissions (id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, admitted REAL NOT NULL, day TEXT NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS conversation_admissions_user ON conversation_admissions(user_id, admitted)')
            db.execute('CREATE TABLE IF NOT EXISTS conversation_leases (user_id INTEGER PRIMARY KEY, token TEXT NOT NULL, expires REAL NOT NULL)')

    def _connect(self):
        return sqlite3.connect(self.path, timeout=5)

    def acquire(self, user_id: int, *, timestamp: float | None = None) -> TurnLease:
        deadline = time.monotonic() + self.timeout
        stamp = time.time() if timestamp is None else timestamp
        day = datetime.fromtimestamp(stamp, ZoneInfo('Asia/Shanghai')).date().isoformat()
        token = uuid.uuid4().hex
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM conversation_leases WHERE expires<=?', (stamp,))
            if db.execute('SELECT 1 FROM conversation_leases WHERE user_id=?', (user_id,)).fetchone():
                raise HTTPException(429, '本账号正在回复，请等待完成。')
            minute, daily = db.execute(
                'SELECT COALESCE(SUM(admitted>?),0), COALESCE(SUM(day=?),0) FROM conversation_admissions WHERE user_id=? AND (admitted>? OR day=?)',
                (stamp - 60, day, user_id, stamp - 60, day),
            ).fetchone()
            if minute >= 6:
                raise HTTPException(429, '每分钟最多发起6次对话，请稍后再试。')
            if daily >= 30:
                raise HTTPException(429, '今日已达30次对话上限，请明日再试。')
            db.execute('INSERT INTO conversation_admissions VALUES (?,?,?,?)', (token, user_id, stamp, day))
            # Five seconds cover callback scheduling and closing the upstream socket.
            db.execute('INSERT INTO conversation_leases VALUES (?,?,?)', (user_id, token, stamp + self.timeout + 5))
            db.execute('DELETE FROM conversation_admissions WHERE admitted<? AND day<>?', (stamp - 86400 * 2, day))
        return TurnLease(self, user_id, token, deadline)


@dataclass
class TurnLease:
    access: ConversationAccess
    user_id: int
    token: str
    deadline: float
    cancelled: threading.Event = field(default_factory=threading.Event)

    def remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if self.cancelled.is_set() or remaining <= 0:
            raise TimeoutError('Conversation cancelled or timed out')
        return remaining

    def release(self):
        try:
            with self.access._connect() as db:
                db.execute('DELETE FROM conversation_leases WHERE user_id=? AND token=?', (self.user_id, self.token))
        except sqlite3.Error:
            # Failure remains closed until deadline; never admit a second worker.
            logger.error('Could not release conversation lease', exc_info=True)


def check_turn() -> float | None:
    turn = _current_turn.get()
    return turn.remaining() if turn is not None else None


def paid_post(url, **kwargs):
    """Bound total upstream time during admitted turns, including slow trickles.

    Sync callers run in worker threads. The async transport's outer timeout is
    absolute, unlike HTTPX's per-read inactivity timeout.
    """
    remaining = check_turn()
    if remaining is None:
        return httpx.post(url, **kwargs)

    async def send():
        timeout = kwargs.pop('timeout', 60)
        async with asyncio.timeout(min(60, remaining)):
            async with httpx.AsyncClient(timeout=timeout) as client:
                return await client.post(url, **kwargs)

    return asyncio.run(send())


async def protected_stream(service, user, request, lease: TurnLease):
    """Keep ownership after SSE disconnect until the actual worker finishes."""
    service._turn_lease = lease
    stream = service.stream_chat(user, request)
    try:
        while True:
            # Keep context and timeout ownership inside each pull: ASGI servers
            # may close/pull an iterator from a different task after a yield.
            token = _current_turn.set(lease)
            try:
                async with asyncio.timeout(lease.remaining()):
                    event = await anext(stream)
            except StopAsyncIteration:
                break
            finally:
                _current_turn.reset(token)
            yield event
    finally:
        lease.cancelled.set()
        try:
            await stream.aclose()
        finally:
            worker = getattr(service, '_worker_task', None)
            thread_finished = getattr(service, '_worker_finished', None)
            if thread_finished is not None:
                # asyncio task cancellation does not stop to_thread's thread.
                # That real thread owns final release while still running.
                if thread_finished.is_set():
                    lease.release()
                if worker is not None and not worker.done():
                    worker.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
            elif worker is not None and not worker.done():
                def finished(task):
                    # Disconnect leaves no consumer to retrieve worker failures.
                    if not task.cancelled():
                        task.exception()
                    lease.release()
                worker.add_done_callback(finished)
            else:
                lease.release()
