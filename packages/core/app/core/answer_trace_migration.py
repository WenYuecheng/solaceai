"""SolaceAI 执行轨迹数据表无损平滑扩容迁移 (MySQL Answer Trace Schema Migration)

本模块提供针对 MySQL 数据库方言的在线 DDL 结构升级兼容能力：
1. 解决 MySQL 原生 `TEXT` 字段最大 64KB 存储上限问题，在不破坏现有数据、不丢弃表记录、
   不改变 JSON 结构的前提下，平滑将 `steps_json` 字段扩容为 `MEDIUMTEXT` (最大 16MB)；
2. 幂等与数据库方言感知：仅在 MySQL 引擎且检测到字段未扩容时执行 `ALTER TABLE`，
   非 MySQL 方言（如 SQLite）或已扩容状态下安全跳过。
"""
from typing import Any
from sqlalchemy import Engine, inspect, text
from sqlalchemy.dialects.mysql import MEDIUMTEXT, LONGTEXT


def migrate_answer_trace_schema(engine: Engine) -> None:
    """平滑升级 MySQL 数据库中 public_answer_traces 表字段容量 (Widen Trace Column Capacity)

    执行逻辑：
    1. 校验当前数据库底层驱动方言是否为 'mysql'，非 MySQL 直接静默退出；
    2. 打开事务连接并获取表结构元数据检查器 (`inspector`)；
    3. 校验目标表 `public_answer_traces` 是否存在；
    4. 检查 `steps_json` 列的物理存储类型，若尚未升级为 MEDIUMTEXT/LONGTEXT，
       则执行 DDL 将其变更扩容为 `MEDIUMTEXT NOT NULL`。

    Args:
        engine (Engine): SQLAlchemy 数据库引擎实例。
    """
    if engine.dialect.name != 'mysql':
        return
    with engine.begin() as connection:
        inspector = inspect(connection)
        if not inspector.has_table('public_answer_traces'):
            return
        column = next(c for c in inspector.get_columns('public_answer_traces') if c['name'] == 'steps_json')
        if not isinstance(column['type'], (MEDIUMTEXT, LONGTEXT)):
            connection.execute(text('ALTER TABLE public_answer_traces MODIFY COLUMN steps_json MEDIUMTEXT NOT NULL'))
