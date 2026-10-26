"""SolaceAI 系统应用启动入口模块 (FastAPI Main Application Entrypoint)

本模块作为整个后端服务的核心装配器，承担以下关键职责：
1. 项目根路径探测与 `sys.path` 动态挂载，确保全局绝对导入稳定；
2. FastAPI 核心应用生命周期 (`lifespan`) 管理：
   - 启动阶段：拉起独立的本地 BGE-M3/CrossEncoder 神经推理 Worker 进程、
     自动创建数据表结构（DDL）、播种初始用户及预置心理知识库；
   - 关闭阶段：两阶段安全清理并终止后台 RAG 推理子进程；
3. HTTP 中间件装配：静态前端页面与核心资产的防客户端强缓存策略；
4. 业务 API 路由网关挂载与前端静态资源单页应用（SPA）托管；
5. 开发环境快速启动入口 (`run_server`)。
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from app.core.paths import PROJECT_ROOT

from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.bootstrap import create_schema, seed_data
from app.core.config import get_settings
from app.core.database import SessionLocal

logger = logging.getLogger(__name__)


def create_app(*, worker_lifecycle=None) -> FastAPI:
    """组装并配置 SolaceAI FastAPI 核心应用实例 (Application Factory)

    通过工厂函数模式构建 FastAPI 应用，集成异步生命周期管理、安全响应头中间件、
    业务 API 网关路由及前端单页应用静态托管。

    Returns:
        FastAPI: 完成全量依赖装配的 FastAPI 应用对象。
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """FastAPI 异步生命周期管理器 (Async Lifespan Context Manager)

        负责服务进程启动 (Startup) 与终止 (Shutdown) 两个阶段的资源准备与善后回收。

        Args:
            app (FastAPI): 正在启动的 FastAPI 实例。

        Yields:
            None: 移交控制权至请求事件循环，维持系统运行。
        """
        start_local_rag, stop_local_rag = worker_lifecycle or (lambda settings: None, lambda process: None)

        # 1. 启动阶段：尝试初始化或复用本地独立的 BGE-M3/CrossEncoder 神经推理子进程
        rag_process = None
        try:
            rag_process = start_local_rag(get_settings())
        except Exception:
            logger.exception("本地 RAG 服务拉起失败，系统知识检索链路将自动回退至词法或降级模式")

        # 2. 启动阶段：初始化持久化存储、播种领域基础数据
        try:
            # 自动创建数据表及增量字段迁移
            create_schema()
            # 打开临时会话播种默认管理员、学生账号及预置心理学文档
            db = SessionLocal()
            try:
                seed_data(db)
            finally:
                db.close()

            # 进入正常对外服务运行期
            yield
        finally:
            # 3. 停机阶段：两阶段平滑关闭神经推理子进程，释放端口与计算资源
            stop_local_rag(rag_process)

    # 实例化 FastAPI 核心网关应用
    app = FastAPI(
        title="SolaceAI Python",
        version="1.0.0",
        description="面向大学生的心理健康辅助系统，提供知识检索与来源引用、情绪支持对话、风险报告及管理员个案跟进。",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def no_cache_frontend_assets(request: Request, call_next) -> Response:
        """HTTP 响应头安全过滤中间件：前端静态资产禁用缓存 (Disable Static Cache Middleware)

        避免前端 SPA 单页应用更新后客户端浏览器因 304 或强缓存导致使用旧版 JS/CSS/HTML 逻辑。

        Args:
            request (Request): 当前传入的 HTTP 请求上下文。
            call_next: 后续路由或中间件调用可调用对象。

        Returns:
            Response: 注入了 'Cache-Control: no-store' 响应头的 HTTP 响应。
        """
        response: Response = await call_next(request)
        path = request.url.path
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-store"
        return response

    # 挂载业务 API 路由网关（包含会话、问答流、个案、知识库、管理员报表等接口）
    app.include_router(router)

    # 挂载单页应用静态资产目录 (HTML, CSS, JS, Fonts)
    static_dir = Path(get_settings().static_dir) if get_settings().static_dir else Path(__file__).resolve().parent / "static"
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


# 全局 ASGI 应用实例统一由各 Profile 应用入口（apps.online / apps.offline）按需延迟创建。

