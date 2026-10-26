"""SQLite/MySQL 知识库身份标识与多版本生命周期增量数据库迁移模块

本模块为系统平滑升级提供热迁移机制，负责：
1. 检查 `knowledge_chunk_metadata`、`knowledge_chunks`、`knowledge_document_versions`
   三个核心数据表中是否缺失新架构所需的身份与多版本字段；
2. 针对尚未拥有 `document_id` 的旧版本历史切片与版本记录，基于哈希算法补全确定性的全局唯一文档标识；
3. 修正并绑定切片所属的具体 `version_id`，保证活跃（ACTIVE）版本的唯一性；
4. 确保在 `knowledge_document_versions` 表上创建 `active_key` 唯一索引约束，防止并发版本冲突。
"""

import hashlib
from sqlalchemy import inspect, text


def migrate_knowledge_schema(engine) -> None:
    """执行知识库多版本存储架构的增量 DDL 检查与数据平滑升级

    该函数设计为完全幂等（Idempotent）：多次重复执行不会破坏现有数据或抛出重复字段异常。

    执行逻辑拆解：
    1. 动态探查表结构：通过 `inspect(conn).get_columns(table)` 获取现有物理列名集合；
    2. 字段级平滑补充：若新版本所需的元数据列不存在，则通过 `ALTER TABLE ADD COLUMN` 动态添加；
    3. 历史数据兼容回填 (Legacy Backfill)：
       - 对于空 `document_id` 的历史记录，按 'legacy:default:default:{source}' 计算 SHA-256 补全；
       - 为每个历史文档推断并设置唯一的激活版本 (`active_key`)，并同步更新切片的 `version_id`；
    4. 创建互斥唯一约束：确保 `knowledge_document_versions(active_key)` 唯一索引就绪。

    Args:
        engine: SQLAlchemy Engine 数据库引擎实例。

    Raises:
        Exception: 数据库底层 DDL 执行发生非预期错误时抛出。
    """
    # 待校验并补充的表结构及列定义字典映射
    fields = {
        'knowledge_chunk_metadata': {
            'page_no': 'INTEGER NULL',
            'version_id': 'INTEGER NULL',
        },
        'knowledge_chunks': {
            'document_id': "VARCHAR(64) NOT NULL DEFAULT ''",
            'knowledge_base_id': "VARCHAR(64) NOT NULL DEFAULT 'default'",
            'tenant_id': "VARCHAR(64) NOT NULL DEFAULT 'default'",
            'version_id': 'INTEGER NULL',
        },
        'knowledge_document_versions': {
            'document_id': "VARCHAR(64) NOT NULL DEFAULT ''",
            'knowledge_base_id': "VARCHAR(64) NOT NULL DEFAULT 'default'",
            'tenant_id': "VARCHAR(64) NOT NULL DEFAULT 'default'",
            'active_key': 'VARCHAR(64) NULL',
            'attempts': 'INTEGER NOT NULL DEFAULT 0',
            'error': 'TEXT NULL',
        },
    }

    with engine.begin() as conn:
        if engine.dialect.name == 'mysql':
            from sqlalchemy.dialects.mysql import LONGTEXT
            columns = inspect(conn).get_columns('knowledge_document_versions')
            metadata = next(column for column in columns if column['name'] == 'metadata_json')
            if not isinstance(metadata['type'], LONGTEXT):
                # Docling stores the document layout, text and structural chunks;
                # even a small PDF routinely exceeds MySQL TEXT's 64KB limit.
                conn.execute(text('ALTER TABLE knowledge_document_versions MODIFY COLUMN metadata_json LONGTEXT NOT NULL'))
        # 第一阶段：动态探测并按需执行 ALTER TABLE 新增列
        for table, columns in fields.items():
            existing = {c['name'] for c in inspect(conn).get_columns(table)}
            for name, definition in columns.items():
                if name not in existing:
                    # 动态注入缺失的列定义
                    conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {name} {definition}'))

        # 第二阶段：查找历史未关联 document_id 的旧数据来源并进行确定性标识回填
        sources = conn.execute(
            text(
                "SELECT DISTINCT source FROM knowledge_chunks WHERE document_id = '' "
                "UNION "
                "SELECT DISTINCT source FROM knowledge_document_versions WHERE document_id = ''"
            )
        ).scalars().all()

        for source in sources:
            # 基于确定性前缀生成稳定哈希作为旧文档的 document_id
            doc_id = hashlib.sha256(('legacy:default:default:' + source).encode()).hexdigest()

            # 回填切片表与版本表中的 document_id
            for table in ('knowledge_chunks', 'knowledge_document_versions'):
                conn.execute(
                    text(f"UPDATE {table} SET document_id=:id WHERE source=:source AND document_id=''"),
                    {'id': doc_id, 'source': source},
                )

            # 查询该文档当前已有的版本列表
            versions = conn.execute(
                text("SELECT id, status FROM knowledge_document_versions WHERE document_id=:id ORDER BY id DESC"),
                {'id': doc_id},
            ).all()

            # 查找是否存在标记为 ACTIVE 的版本
            active = next((v.id for v in versions if v.status == 'ACTIVE'), None)

            # 若存在版本但没有 ACTIVE 状态，则将最新版本号赋予给孤立切片
            if versions and active is None:
                conn.execute(
                    text('UPDATE knowledge_chunks SET version_id=:version WHERE document_id=:id AND version_id IS NULL'),
                    {'id': doc_id, 'version': versions[0].id},
                )

            # 若已有 ACTIVE 状态版本，则计算确定性激活互斥键，并将其他多余 ACTIVE 归档为 ARCHIVED
            if active is not None:
                key = hashlib.sha256(('default:default:' + doc_id).encode()).hexdigest()
                conn.execute(
                    text("UPDATE knowledge_document_versions SET status='ARCHIVED', active_key=NULL WHERE document_id=:id AND status='ACTIVE' AND id<>:active"),
                    {'id': doc_id, 'active': active},
                )
                conn.execute(
                    text('UPDATE knowledge_document_versions SET active_key=:key WHERE id=:id'),
                    {'key': key, 'id': active},
                )
                conn.execute(
                    text('UPDATE knowledge_chunks SET version_id=:version WHERE document_id=:id AND version_id IS NULL'),
                    {'id': doc_id, 'version': active},
                )

        # 第三阶段：同步更新切片扩展元数据表中的 version_id
        conn.execute(
            text(
                'UPDATE knowledge_chunk_metadata '
                'SET version_id=(SELECT version_id FROM knowledge_chunks WHERE knowledge_chunks.id=knowledge_chunk_metadata.chunk_id) '
                'WHERE version_id IS NULL'
            )
        )

        # 第四阶段：保证 active_key 唯一索引存在，防止并发导入版本破坏唯一性
        indexes = {i['name'] for i in inspect(conn).get_indexes('knowledge_document_versions')}
        constraints = inspect(conn).get_unique_constraints('knowledge_document_versions')
        has_unique_active_key = 'uq_knowledge_active_key' in indexes or any(
            c['column_names'] == ['active_key'] for c in constraints
        )
        if not has_unique_active_key:
            conn.execute(text('CREATE UNIQUE INDEX uq_knowledge_active_key ON knowledge_document_versions(active_key)'))
