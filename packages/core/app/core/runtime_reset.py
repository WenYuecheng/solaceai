"""Maintenance-only conversation reset. Stop all writers before invoking this module.

Unknown tables, knowledge stores and API spending ledgers are always retained.
SQLite backups are native snapshots; other SQLAlchemy engines use a full logical
JSON export containing table DDL, column names and tagged scalar values. Backup
files contain private records and password hashes and are created with mode 0600.
"""
from __future__ import annotations

import argparse
import base64
from datetime import date, datetime, time, timezone
from decimal import Decimal
import importlib
import json
import os
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from sqlalchemy import MetaData, create_engine, delete, select
from sqlalchemy.schema import CreateTable

RESET_TABLES = frozenset({
    'chat_sessions', 'chat_messages', 'session_visibility', 'agent_turn_receipts',
    'psychological_reports', 'risk_cases', 'case_notes', 'alert_records',
    'excel_records', 'tool_jobs', 'dead_letter_records', 'agent_run_traces',
    'tool_audit_records', 'tool_executions', 'chat_message_runs', 'answer_feedback',
    'run_metrics', 'agent_store_items', 'public_answer_traces', 'auth_sessions',
    'email_auth_codes', 'auth_rate_limits',
})
MEMORY_TABLES = frozenset({'preferences', 'summaries', 'store', 'store_vectors', 'items'})
CHECKPOINT_TABLES = frozenset({'checkpoints', 'writes', 'checkpoint_blobs', 'checkpoint_writes'})


def _backup_path(directory: Path, label: str, suffix: str) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    path = directory / f'{label}-{stamp}-{uuid4().hex[:12]}{suffix}'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    return path


def _json_scalar(value):
    if isinstance(value, bytes):
        return {'__type__': 'bytes', 'base64': base64.b64encode(value).decode('ascii')}
    if isinstance(value, (datetime, date, time, Decimal)):
        return {'__type__': type(value).__name__, 'value': str(value)}
    raise TypeError(f'Unsupported backup scalar type: {type(value).__name__}')


def _write_json(path: Path, payload) -> None:
    with path.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, default=_json_scalar)
        stream.flush()
        os.fsync(stream.fileno())


def backup_database(engine, directory: Path) -> Path:
    """Back up every table, including retained knowledge, costs and unknown tables."""
    if engine.dialect.name == 'sqlite':
        path = _backup_path(directory, 'database', '.sqlite')
        with engine.connect() as conn, sqlite3.connect(path) as target:
            conn.connection.driver_connection.backup(target)
        return path
    path = _backup_path(directory, 'database', '.json')
    # A consistent transaction snapshot requires InnoDB on MySQL and stopped writers.
    with engine.connect().execution_options(isolation_level='REPEATABLE READ') as conn:
        with conn.begin():
            metadata = MetaData()
            metadata.reflect(bind=conn)
            tables = []
            for table in metadata.sorted_tables:
                tables.append({
                    'name': table.name,
                    'ddl': str(CreateTable(table).compile(dialect=engine.dialect)),
                    'columns': list(table.c.keys()),
                    'rows': [list(row) for row in conn.execute(select(table))],
                })
            _write_json(path, {'format': 'solaceai-logical-backup-v1', 'dialect': engine.dialect.name, 'tables': tables})
    return path


def reset_runtime_data(engine, backup_dir: Path, *, profile: str, prune_online_accounts: bool = False) -> dict:
    """Back up, then atomically delete only the named runtime tables in FK order."""
    if profile not in {'local', 'offline', 'online'}:
        raise ValueError('Unknown reset profile')
    if profile == 'online' and not prune_online_accounts:
        raise ValueError('Online reset requires explicit prune_online_accounts=True')
    if profile != 'online' and prune_online_accounts:
        raise ValueError('Account prune is only allowed for the online profile')
    metadata = MetaData()
    metadata.reflect(bind=engine)
    if 'user_accounts' not in metadata.tables:
        raise ValueError('Expected application schema is missing; initialize/migrate it before resetting')
    # Fail before backup/deletion if an unknown table references a reset target.
    # This also avoids silently cascading away data from a retained table.
    targets = set(RESET_TABLES)
    if prune_online_accounts:
        targets.add('user_accounts')
    for table in metadata.tables.values():
        if table.name not in targets:
            if any(fk.column.table.name in targets for fk in table.foreign_keys):
                raise ValueError(f'Retained table {table.name} references reset data; review schema before resetting')
    backup = backup_database(engine, backup_dir)
    deleted = {}
    with engine.begin() as conn:
        for table in reversed(metadata.sorted_tables):
            if table.name in RESET_TABLES:
                deleted[table.name] = conn.execute(delete(table)).rowcount
        if prune_online_accounts:
            from app.core.security import hash_password
            users = metadata.tables['user_accounts']
            deleted['user_accounts'] = conn.execute(delete(users)).rowcount
            conn.execute(users.insert(), [
                dict(username='admin', display_name='Administrator', password_hash=hash_password('admin123'), roles_csv='ROLE_ADMIN', created_at=datetime.now(timezone.utc).replace(tzinfo=None)),
                dict(username='student', display_name='Student', password_hash=hash_password('student123'), roles_csv='ROLE_STUDENT', created_at=datetime.now(timezone.utc).replace(tzinfo=None)),
            ])
    return {'backup': backup, 'deleted': deleted}


def reset_auxiliary_sqlite(path: Path, backup_dir: Path, *, tables: set[str] | frozenset[str]) -> dict:
    """Reset known memory/checkpoint tables while retaining the file and schema."""
    path = Path(path)
    if not path.exists():
        return {'backup': None, 'deleted': {}}
    engine = create_engine('sqlite:///' + str(path))
    try:
        metadata = MetaData()
        metadata.reflect(bind=engine)
        backup = backup_database(engine, backup_dir)
        deleted = {}
        with engine.begin() as conn:
            for table in reversed(metadata.sorted_tables):
                if table.name in tables:
                    deleted[table.name] = conn.execute(delete(table)).rowcount
        return {'backup': backup, 'deleted': deleted}
    finally:
        engine.dispose()


def reset_redis_sessions(client, namespace: str, backup_dir: Path) -> dict:
    """Export Redis DUMP/PTTL values, then remove only chat-history cache keys."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+(?::[A-Za-z0-9_-]+)*', namespace):
        raise ValueError('A literal nonempty Redis namespace is required')
    patterns = (f'{namespace}:history:v2:*', f'{namespace}:short-term-memory:*')
    records = []
    for pattern in patterns:
        for key in client.scan_iter(match=pattern, count=500):
            value = client.dump(key)
            if value is not None:
                records.append({'key': key, 'dump': value, 'pttl': client.pttl(key)})
    backup = _backup_path(backup_dir, 'redis-sessions', '.json')
    _write_json(backup, {'format': 'solaceai-redis-backup-v1', 'namespace': namespace, 'keys': records})
    deleted = sum(client.delete(row['key']) for row in records)
    return {'backup': backup, 'deleted': deleted}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=('local', 'offline', 'online'))
    parser.add_argument('--prune-online-accounts', action='store_true')
    parser.add_argument('--reset-agent-state', action='store_true', help='Back up and clear checkpoint and memory SQLite stores')
    parser.add_argument('--reset-redis-sessions', action='store_true', help='Back up and clear only namespaced chat cache keys')
    parser.add_argument('--backup-dir', type=Path, default=None)
    parser.add_argument('--execute', action='store_true', help='Apply the reset; otherwise show its scope only')
    args = parser.parse_args(argv)
    if args.profile == 'local':
        from app.core.config import Settings
        settings = Settings()
    else:
        settings = importlib.import_module(f'apps.{args.profile}.settings').load_settings()
    from app.core.paths import PROJECT_ROOT
    backup_dir = args.backup_dir or PROJECT_ROOT / 'data/runtime-reset-backups' / args.profile
    if args.profile == 'online' and not args.prune_online_accounts:
        parser.error('online reset requires --prune-online-accounts')
    if args.profile != 'online' and args.prune_online_accounts:
        parser.error('--prune-online-accounts requires online profile')
    if not args.execute:
        print(f'Preview only: {args.profile}; runtime tables: {", ".join(sorted(RESET_TABLES))}')
        print(f'Backups: {backup_dir}; agent state: {args.reset_agent_state}; Redis chat caches: {args.reset_redis_sessions}')
        print('Stop all app/worker processes, then add --execute to apply.')
        return
    engine = create_engine(settings.database_url, hide_parameters=True)
    try:
        protected_paths = {Path(p).resolve() for p in (getattr(settings, 'api_budget_path', ''), engine.url.database if engine.dialect.name == 'sqlite' else '') if p}
        state_paths = [Path(getattr(settings, name)) for name in ('agent_checkpoint_path', 'agent_memory_path')]
        state_paths = [p if p.is_absolute() else PROJECT_ROOT / p for p in state_paths]
        if args.reset_agent_state and any(p.resolve() in protected_paths for p in state_paths):
            raise ValueError('Agent state path overlaps the application database or paid cost ledger')
        redis_client = None
        if args.reset_redis_sessions:
            from redis import Redis
            redis_client = Redis.from_url(settings.redis_url, socket_timeout=5)
            redis_client.ping()
        result = reset_runtime_data(engine, backup_dir, profile=args.profile, prune_online_accounts=args.prune_online_accounts)
        print(f'Database reset complete. Backup: {result["backup"]}')
        print(json.dumps(result['deleted'], sort_keys=True))
        if args.reset_agent_state:
            for path, tables in zip(state_paths, (CHECKPOINT_TABLES, MEMORY_TABLES)):
                state_result = reset_auxiliary_sqlite(path, backup_dir, tables=tables)
                print(f'Agent state backup: {state_result["backup"]}')
        if redis_client is not None:
            cache_result = reset_redis_sessions(redis_client, settings.redis_namespace, backup_dir)
            print(f'Redis chat cache reset: {cache_result["deleted"]}; backup: {cache_result["backup"]}')
    finally:
        engine.dispose()


if __name__ == '__main__':
    main()
