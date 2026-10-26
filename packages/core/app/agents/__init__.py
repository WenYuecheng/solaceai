"""SolaceAI 多智能体架构入口与运行时工厂包 (Multi-Agent Runtime Entrypoint)

本包提供多智能体协同运行时的创建工厂与架构运行状态探测：
1. 运行时初始化工厂 `create_agent_runtime`：根据全局配置实例化基于 LangGraph 的有向图多智能体状态机；
2. 架构运行时状态探测 `agent_framework_status`：查询当前生效的智能体框架、调度器类型与降级模式。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from sqlalchemy.orm import Session
from app.core.config import Settings

if TYPE_CHECKING:
    from app.agents.graph_runtime import GraphAgentRuntime


def create_agent_runtime(db: Session, settings: Settings, redis_memory: Any = None) -> GraphAgentRuntime:
    """创建并初始化 LangGraph 多智能体协同运行时实例 (Create Multi-Agent Graph Runtime)

    装配数据库会话与系统配置，构建包含决策节点、工具观测节点和独立安全审核节点的执行状态图。

    Args:
        db (Session): 当前请求生命周期内的 SQLAlchemy 事务会话。
        settings (Settings): 系统全局配置单例对象。
        redis_memory (Any, optional): 可选的 Redis 短期记忆存储组件。

    Returns:
        GraphAgentRuntime: 装配完毕的 LangGraph 多智能体运行时实例。
    """
    from app.agents.graph_runtime import GraphAgentRuntime
    if settings.chat_backend == 'local':
        from app.services.chat_backends import make_local_runtime
        return make_local_runtime(db, settings, redis_memory)
    if settings.local_model_switch_enabled and settings.runtime_profile=='offline':
        from app.services.chat_backends import LocalModelClient
        return GraphAgentRuntime(db, settings, redis_memory=redis_memory,
            emotion_client=LocalModelClient(settings.local_gateway_url,'solace-emotion',settings=settings))
    if redis_memory is None:
        return GraphAgentRuntime(db, settings)
    return GraphAgentRuntime(db, settings, redis_memory=redis_memory)


def agent_framework_status(settings: Settings) -> dict:
    """查询当前多智能体架构的运行状态与能力矩阵 (Query Agent Framework Health Status)

    用于管理员控制台展示智能体调度拓扑、活跃驱动类型及降级回退标识。

    Args:
        settings (Settings): 全局配置对象。

    Returns:
        dict: 包含请求框架 ('requested')、当前活跃框架 ('active')、
            可用列表 ('available') 与降级标识 ('fallback') 的字典。
    """
    requested = settings.agent_framework.lower()
    return {
        'requested': requested,
        'active': 'langgraph',
        'available': ['langgraph'],
        'fallback': requested not in {'langgraph', 'graph'},
        'legacyArchived': True,
    }
