"""SolaceAI 用户认证模式演进与邮箱凭据字段增量迁移 (User Auth Schema Migration)

本模块负责在现有关系型数据库中平滑推进身份认证架构演进：
1. 增量扩展可选邮箱字段：为历史版本的 `user_accounts` 用户表无损添加可为空的 `email VARCHAR(254) NULL` 字段，
   完全保持历史存量密码凭据与用户名体系兼容；
2. 跨数据库方言唯一索引幂等创建：针对 MySQL 不支持 `CREATE INDEX IF NOT EXISTS` 的语法限制，
   在事务连接内先利用元数据探针（`inspector.get_indexes`）检索是否存在目标唯一索引，
   确认缺失后再显式创建 `ix_user_accounts_email` 唯一约束索引，适配 SQLite 与 MySQL。
"""
from sqlalchemy import Engine, inspect, text


def migrate_auth_schema(engine: Engine) -> None:
    """平滑为 user_accounts 表增量迁移 email 字段及唯一约束索引 (Migrate User Email Column & Index)

    执行逻辑：
    1. 开启事务连接并探测当前数据库中是否存在 `user_accounts` 表；
    2. 检查现有列清单，若缺失 `email` 字段则执行 `ALTER TABLE ... ADD COLUMN email VARCHAR(254) NULL`；
    3. 重新检查表的现有索引，若不存在覆盖 `email` 单列的唯一索引，
       则执行 `CREATE UNIQUE INDEX ix_user_accounts_email ON user_accounts (email)`。

    Args:
        engine (Engine): SQLAlchemy 数据库引擎实例。
    """
    with engine.begin() as connection:
        inspector = inspect(connection)
        if not inspector.has_table('user_accounts'):
            return
        columns = {column['name'] for column in inspector.get_columns('user_accounts')}
        if 'email' not in columns:
            connection.execute(text('ALTER TABLE user_accounts ADD COLUMN email VARCHAR(254) NULL'))
        if 'requires_setup' not in columns:
            connection.execute(text('ALTER TABLE user_accounts ADD COLUMN requires_setup BOOLEAN NOT NULL DEFAULT 0'))
            # One-time recognition of the exact historical auto-provisioning shape.
            if {'display_name', 'roles_csv'}.issubset(columns):
                rows = connection.execute(text('SELECT id, username FROM user_accounts WHERE email IS NOT NULL AND display_name = :name AND roles_csv = :roles'), {'name': '新同学', 'roles': 'ROLE_USER'}).all()
                import re
                for row in rows:
                    if re.fullmatch(r'student_[0-9a-f]{24}', row.username):
                        connection.execute(text('UPDATE user_accounts SET requires_setup = 1 WHERE id = :id'), {'id': row.id})
        # MySQL 无原生 CREATE INDEX IF NOT EXISTS 语法，通过元数据探针显式判断以保证幂等性
        indexes = inspect(connection).get_indexes('user_accounts')
        if not any(index['unique'] and index['column_names'] == ['email'] for index in indexes):
            connection.execute(text('CREATE UNIQUE INDEX ix_user_accounts_email ON user_accounts (email)'))
