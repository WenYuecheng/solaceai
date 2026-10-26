"""SolaceAI 线下高安全模式 ASGI 服务入口与命令行启动器 (Offline ASGI Application Entrypoint)

本模块作为线下私有化模式的顶层执行入口：
1. 双重执行环境兼容：无缝支持标准模块启动 (`python -m apps.offline.main`) 与 IDE 单文件直接执行；
2. 隔离存储目录预先初始化：在 SQLAlchemy 数据库引擎初始化前，提前创建数据专属目录；
3. 本地推理 Worker 进程生命周期托管：
   - 应用启动时自动触发 `start_local_rag` 幂等自举启动 BGE-M3/Cross-Encoder 独立工作子进程；
   - 应用退出时通过 lifespan 钩子安全触发 `stop_local_rag` 优雅终止进程；
4. 终端启动支持：提供标准 `run()` 函数，通过 Uvicorn 拉起服务监听（默认端口 8082）。
"""
from pathlib import Path
from typing import Any

# 兼容 python -m 启动与 PyCharm Run File 操作，动态注入工程根目录
if not __package__:
    import sys
    source_root = str(Path(__file__).resolve().parents[2])
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    __package__ = 'apps.offline'

from .settings import load_settings


def create_app(settings: Any = None):
    """构建并初始化线下高安全模式的 FastAPI 应用实例 (Offline Application Factory)

    执行逻辑：
    1. 加载并全局配置 ProfileSettings 单例；
    2. 在加载关系型数据库引擎之前，提前创建记忆库、检查点、API 预算以及 SQLite 文件的父级物理目录；
    3. 注册本地推理进程的启停生命周期钩子 `worker_lifecycle=(start_local_rag, stop_local_rag)`；
    4. 调用核心应用工厂 `app.application.create_app()` 完成全量中间件与路由装配。

    Args:
        settings (Any, optional): 可选注入的特化配置实例，默认调用 `load_settings()` 加载。

    Returns:
        FastAPI: 完成线下特化依赖与进程托管装配的 FastAPI 核心实例。
    """
    from app.core.config import configure_settings
    settings = settings or load_settings()
    configure_settings(settings)
    # 在导入进程级数据库引擎之前，提前预建隔离存储物理目录
    for value in (
        settings.agent_memory_path,
        settings.agent_checkpoint_path,
        settings.api_budget_path,
        settings.excel_path,
    ):
        Path(value).resolve().parent.mkdir(parents=True, exist_ok=True)
    if settings.database_url.startswith('sqlite:///') and ':memory:' not in settings.database_url:
        Path(settings.database_url.removeprefix('sqlite:///')).resolve().parent.mkdir(parents=True, exist_ok=True)
    from app.application import create_app as build
    from .local_rag import start_local_rag, stop_local_rag
    return build(worker_lifecycle=(start_local_rag, stop_local_rag))


_app = None


def __getattr__(name: str) -> Any:
    """按需延迟加载 FastAPI ASGI 应用单例对象 (Lazy Load ASGI App Singleton)

    Args:
        name (str): 属性名称。

    Returns:
        Any: 当 name == 'app' 时返回单例 FastAPI 实例。

    Raises:
        AttributeError: 当请求的属性不是 'app' 时抛出。
    """
    global _app
    if name == 'app':
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)


def run() -> None:
    """启动线下高安全模式 Uvicorn 异步 Web 服务器 (Run Offline ASGI Server)

    读取线下 Profile 配置的主机地址与监听端口（默认 127.0.0.1:8082）并启动服务进程。
    """
    import uvicorn
    settings = load_settings()
    uvicorn.run('apps.offline.main:app', host=settings.server_host, port=settings.server_port)


if __name__ == '__main__':
    run()
