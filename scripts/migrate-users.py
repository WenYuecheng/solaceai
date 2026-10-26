"""SolaceAI 用户账号跨库恢复与增量同步运维工具 (CLI Account Migration & Backup Tool)

本脚本面向系统管理员与运维人员，用于将全局主数据库中的用户账号增量迁移同步至特定运行模式（Online/Offline）的隔离数据库中：
1. 幂等增量迁移：仅同步目标库中缺失的用户实体，已存在用户自动保留；
2. 自动化前置快照备份：在执行任何数据写入前，自动调用 SQLite 原生在线热备份 API (`current.backup`)
   生成时间戳快照文件（存放于 `target/account-migration-backups/`，权限严格控制为 0600），防范意外数据破坏；
3. 严格数据完整性保护：迁移过程在独立事务中执行，遇到冲突自动回滚并释放连接引擎。
"""
import argparse
from datetime import datetime
import importlib
import os
from pathlib import Path
import sqlite3
import sys

# 注入工程根路径以支持绝对导入
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    """跨库账号迁移主函数 (Main Migration Execution Routine)

    执行步骤：
    1. 解析目标 Profile 参数 ('online' 或 'offline')；
    2. 读取顶层全局 `.env` 中的原始 `DATABASE_URL` 作为数据源；
    3. 校验源数据库与目标数据库是否相同，相同则无需迁移；
    4. 执行数据库 Schema 自动检查与建表 (`create_schema`)；
    5. 创建严格受限权限 (0600) 的时间戳 SQLite 本地冷备份快照；
    6. 调用 `migrate_missing_users` 执行无损事务同步并打印统计摘要。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', choices=('online', 'offline'), help='目标运行模式环境 (online/offline)')
    args = parser.parse_args()

    from dotenv import dotenv_values
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    # 加载目标 Profile 的隔离配置
    settings = importlib.import_module(f'apps.{args.profile}.settings').load_settings()
    source_url = dotenv_values(ROOT / '.env').get('DATABASE_URL')
    if not source_url:
        raise SystemExit('Original DATABASE_URL is not configured')
    if source_url == settings.database_url:
        print('Profile already uses original database; no migration needed')
        return

    from app.core.config import configure_settings
    configure_settings(settings)
    from app.core.bootstrap import create_schema
    from app.core.database import SessionLocal, engine
    from app.core.user_migration import migrate_missing_users

    create_schema()
    if engine.dialect.name != 'sqlite':
        raise SystemExit('This migration requires a SQLite target with an automatic backup')

    # 在迁移前创建带权限保护的安全备份副本
    backup_root = ROOT / 'target/account-migration-backups'
    backup_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    backup_path = backup_root / f'{args.profile}-{datetime.now():%Y%m%d-%H%M%S}.sqlite'
    descriptor = os.open(backup_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)

    # 利用 SQLite 原生热备份 API 创建完整物理副本
    with sqlite3.connect(engine.url.database) as current, sqlite3.connect(backup_path) as backup:
        current.backup(backup)

    source_engine = create_engine(source_url)
    try:
        with Session(source_engine, autoflush=False) as source, SessionLocal() as target:
            result = migrate_missing_users(source, target)
        print(f"{args.profile}: imported {result['imported']} missing accounts; kept {result['existing']} existing accounts")
        print(f'Backup: {backup_path}')
    finally:
        source_engine.dispose()


if __name__ == '__main__':
    main()
