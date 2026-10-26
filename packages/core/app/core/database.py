"""SolaceAI 数据库持久化基础设施模块 (Database Persistence & Session Management)

本模块构建并管理系统的关系型数据库持久化能力，包含：
1. 声明式 ORM 模型基类 `Base` 的定义；
2. SQLAlchemy 2.0 `Engine` 实例构建与连接池优化参数配置（自动探测存活、连接生命期循环回收）；
3. SQLite 方言专有 PRAGMA 调优（启用外键级联检查、WAL 预写日志并发读写模式、忙等待超时）；
4. 线程隔离的会话工厂 `SessionLocal` 与生命周期受控的依赖注入生成器 `get_db`；
5. 编程式上下文事务支持 `session_scope`。
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 ORM 声明式模型公共基类

    所有业务实体类均继承此类以自动获取元数据注册、类型映射及映射关系检查能力。
    """
    pass


def apply_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
    """SQLite 数据库底层物理连接初始化钩子函数 (Connection Pragmas Hook)

    针对 SQLite 嵌入式文件数据库，在每个原生底层连接建立时执行性能与可靠性调优：
    - PRAGMA foreign_keys=ON: 强制开启外键约束检查（SQLite 默认关闭外键约束）；
    - PRAGMA busy_timeout=5000: 锁竞争时最大等待 5000ms，降低并发写冲突抛出 database is locked 的概率；
    - PRAGMA journal_mode=WAL: 启用 Write-Ahead Logging 预写式日志模式，支持读写并发无锁争用。

    Args:
        dbapi_connection: 底层 DB-API 原始连接对象（通常为 sqlite3.Connection）。
        _connection_record: SQLAlchemy 连接池内部维护的记录对象。
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA journal_mode=WAL")
    finally:
        cursor.close()


# 加载全局配置并组装数据库引擎参数
settings = get_settings()
engine_kwargs = {
    # 每次从连接池获取物理连接前执行轻量 ping 检测，自动丢弃数据库端因超时已关闭的无效连接
    "pool_pre_ping": True,
    # 物理连接最大存活时间（秒），默认 1 小时，规避 MySQL wait_timeout 导致的静默断开问题
    "pool_recycle": 3600,
}

# SQLite 专有跨线程连接配置：允许跨线程复用（开发测试环境）
if settings.database_url.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

# 创建全局唯一的 SQLAlchemy 引擎实例
engine = create_engine(
    settings.database_url,
    **engine_kwargs,
)

# 若当前使用的是 SQLite，则监听连接创建事件以自动注入 Pragma 性能与安全调优
if settings.database_url.startswith("sqlite"):
    event.listen(engine, "connect", apply_sqlite_pragmas)

# 创建线程安全的数据库会话工厂
# - autoflush=False: 避免在查询时触发隐式刷新，提升性能并让持久化时机可控
# - autocommit=False: 遵循现代 SQLAlchemy 显式事务规范，由业务代码显式执行 commit 或 rollback
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    """FastAPI 路由依赖注入数据库会话生成器 (Database Session Dependency)

    作为 FastAPI 依赖项使用，确保每个 HTTP 请求拥有独立的数据库会话，
    并在请求完成或发生异常时通过 `finally` 块确定性地关闭会话、归还连接至连接池。

    Yields:
        Session: 当前请求生命周期内有效的 SQLAlchemy 会话对象。

    Example:
        ```python
        @router.get("/items")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
        ```
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def session_scope() -> Session:
    """编程式创建独立数据库会话实例 (Programmatic Session Factory)

    适用于后台任务、消息队列监听器、定时作业或 CLI 工具等非 HTTP 请求上下文。
    调用方在完成数据库交互后务必负责显式调用 `close()` 关闭会话或结合上下文管理器使用。

    Returns:
        Session: 全新的独立 SQLAlchemy 会话对象。
    """
    return SessionLocal()
