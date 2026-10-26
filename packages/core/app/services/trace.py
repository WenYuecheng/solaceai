"""SolaceAI 多智能体运行追踪与可观测性服务 (Agent Execution Tracing & Observability)"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from app.agents.result import AgentRunResult
from app.models.entities import AgentRunTrace, ChatSession, UserAccount


class AgentTraceService:
    """多智能体协同运行链路全景追踪服务 (Agent Observability & Trace Service)"""

    def __init__(self, db: Session):
        """初始化智能体链路追踪服务实例"""
        self.db = db

    def save_run(
        self,
        user: UserAccount,
        session: ChatSession,
        original_input: str,
        sanitized_input: str,
        memory_brief: str,
        agent_run: AgentRunResult,
        report_id: int | None,
    ) -> AgentRunTrace:
        """将单次多智能体协同运行的全部追踪数据结构化存入数据库"""
        trace = AgentRunTrace(
            user_id=user.id,
            session_id=session.id,
            report_id=report_id,
            intent=agent_run.intent.value,
            risk_level=agent_run.risk_level.value,
            original_input=original_input,
            sanitized_input=sanitized_input,
            memory_brief=memory_brief,
            agent_steps_json=_json(_agent_steps_with_collaboration(agent_run)),
            retrieved_knowledge_json=_json(agent_run.retrieved_knowledge),
            response_messages_json=_json(agent_run.response_messages),
            assessment_json=_json(agent_run.assessment or {}),
        )
        self.db.add(trace)
        self.db.commit()
        self.db.refresh(trace)
        return trace


def _json(value: Any) -> str:
    """将任意数据结构转换为 JSON 字符串（处理枚举、dataclass、Pydantic 模型）"""
    return json.dumps(_to_jsonable(value), ensure_ascii=False, default=str)


def _agent_steps_with_collaboration(agent_run: AgentRunResult) -> list[Any]:
    """聚合组装智能体执行步骤、协作事件、任务清单与工件"""
    entries: list[Any] = [*agent_run.steps]
    entries.extend(artifact for artifact in agent_run.collaboration_artifacts if isinstance(artifact,dict))
    entries.extend(
        {
            "kind": "agent_event",
            "type": getattr(event.type, "value", event.type),
            "actor": event.actor,
            "taskId": event.task_id,
            "artifactId": event.artifact_id,
            "message": event.message,
            "metadata": event.metadata,
        }
        for event in agent_run.collaboration_events
    )
    entries.extend(
        {
            "kind": "agent_task",
            "id": task.id,
            "title": task.title,
            "status": getattr(task.status, "value", task.status),
            "priority": getattr(task.priority, "value", task.priority),
            "requiredCapabilities": sorted(task.required_capabilities),
            "claimedBy": list(task.claimed_by),
            "createdBy": task.created_by,
            "metadata": task.metadata,
        }
        for task in agent_run.collaboration_tasks
    )
    entries.extend(
        {
            "kind": "agent_artifact",
            "id": artifact.id,
            "owner": artifact.owner,
            "artifactKind": artifact.kind,
            "confidence": artifact.confidence,
            "taskId": artifact.task_id,
            "metadata": artifact.metadata,
            "payload": artifact.payload,
        }
        for artifact in agent_run.collaboration_artifacts if not isinstance(artifact,dict)
    )
    return entries


def _to_jsonable(value: Any) -> Any:
    """递归转换不可直接 JSON 序列化的对象为基本数据类型（dict, list, str, number）"""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value
