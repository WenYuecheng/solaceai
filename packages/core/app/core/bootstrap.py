"""SolaceAI 系统启动自举模块 (System Bootstrap & Database Seeding)

本模块承担系统启动运行前的环境初始化工作，包括：
1. 数据库底层表结构定义（DDL）的自动化检测与构建；
2. 增量数据库结构迁移（针对知识库切片、元数据与文档版本控制字段）；
3. 初始系统用户数据播种（默认内置管理员与演示学生账号，使用 Argon2 安全加密）；
4. 预置心理健康领域专业知识库文件的自动扫描与批量摄入同步。
"""

from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import Base, engine
from app.core.security import hash_password
from app.models.entities import UserAccount
from app.services.knowledge import KnowledgeService


def create_schema() -> None:
    """自动创建数据库表结构并执行增量迁移 (DDL Create Tables & Schema Migration)

    执行流程：
    1. 调用 SQLAlchemy ORM 元数据 `Base.metadata.create_all` 扫描所有实体模型，
       若数据表尚不存在则执行对应 DDL 语句创建新表。
    2. 动态引入并执行 `migrate_knowledge_schema` 增量迁移函数，确保旧版本数据库
       能够平滑补充缺失的字段（如 document_id, version_id 等）与唯一索引约束。

    Side Effects:
        在目标数据库中创建物理数据表及更新表结构定义。
    """
    # 1. 扫描所有注册在 Base 上的 ORM 实体，创建缺失的数据表
    from app.models import auth, answer_trace  # 显式导入以注册认证凭据表与学生端可见执行轨迹表元数据
    from app.core.auth_migration import migrate_auth_schema
    migrate_auth_schema(engine)
    Base.metadata.create_all(bind=engine)
    from app.core.answer_trace_migration import migrate_answer_trace_schema
    migrate_answer_trace_schema(engine)

    # 2. 延迟导入迁移模块以避免循环依赖，执行知识库多版本架构增量迁移
    from app.core.knowledge_migration import migrate_knowledge_schema
    migrate_knowledge_schema(engine)


def seed_data(db: Session) -> None:
    """注入系统初始种子数据 (Seed Initial Data)

    执行逻辑：
    1. 账号初始化：检测用户表 `UserAccount` 是否为空。若为空，则创建两个初始测试账号：
       - admin: 拥有 ROLE_ADMIN 与 ROLE_USER 权限的心理辅导员/系统管理员；
       - student: 拥有 ROLE_USER 权限的普通学生演示账号；
       密码统一采用现代 Argon2id 密码哈希算法进行加盐散列。
    2. 知识库预置数据同步：扫描 `app/knowledge` 目录下所有预置的 Markdown 格式心理科普与干预文件，
       调用 `KnowledgeService.ensure_source` 进行防重校验与文档切片入库。

    Args:
        db (Session): 当前活动的 SQLAlchemy 数据库事务会话。

    Raises:
        Exception: 数据库写入异常或知识库解析失败时向上抛出，由外层事务负责回滚。
    """
    # 1. 检查并初始化默认系统用户（若已存在用户数据则跳过，保证幂等性）
    if db.query(UserAccount).count() == 0:
        # 创建默认辅导员/管理员账号
        admin = UserAccount(
            username="admin",
            display_name="Counselor Admin",
            password_hash=hash_password("admin123"),
        )
        admin.roles = {"ROLE_ADMIN", "ROLE_USER"}

        # 创建默认学生演示账号
        student = UserAccount(
            username="student",
            display_name="Demo Student",
            password_hash=hash_password("student123"),
        )
        student.roles = {"ROLE_USER"}

        db.add_all([admin, student])
        db.commit()

    # 2. 扫描并自动同步预置心理健康知识库
    # 系统启动流程严禁强依赖远程付费 API 或全量构建耗时模型索引；两种后端均在首次检索到达时按需增量构建缺失向量
    seed_settings = get_settings().model_copy(update={"knowledge_vector_enabled": False})
    service = KnowledgeService(db, seed_settings)
    # 定位知识库文件目录：当前文件所在目录上一级即为 app 目录
    root = Path(__file__).resolve().parents[1]
    knowledge_dir = root / "knowledge"

    if knowledge_dir.exists():
        for file in sorted(knowledge_dir.glob("*.md")):
            # ensure_source 内部会自动比对文件内容哈希，若无变动则直接跳过，避免重复处理
            service.ensure_source(file.name, file.read_text(encoding="utf-8"))
