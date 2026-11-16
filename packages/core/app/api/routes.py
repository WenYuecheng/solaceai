"""SolaceAI HTTP API 路由控制层 (API Gateway & Request Routing)

该模块集中定义和挂载 SolaceAI 系统所有的 RESTful API 与 SSE 流式通信端点：
- 系统探针与基础配置 (Health, Agent Status)
- 用户档案与记忆管理 (Profile, Explicit Memories)
- 心理咨询流式问答对话与证据追溯 (Chat Stream, Evidence Detail, Feedback)
- 报告与多智能体执行链路审查 (Psychological Reports, Agent Run Traces, Tool Audits)
- 知识库全生命周期治理 (Ingest, Preview, Rebuild, Backup, Delete)
- 危机个案流转与会话管理子路由 (Case Workflow, Conversations)

架构设计规范：
1. 依赖注入：数据库会话由 `Depends(get_db)` 注入，用户身份由 `Depends(current_user)` 或 `Depends(require_admin)` 鉴权。
2. 职责分离：路由层只负责参数校验、HTTP 协议解包与响应封包，具体业务下沉至 Service 领域服务层。
3. 高内聚低耦合：异常捕获精准映射为对应的 HTTP 状态码 (403, 404, 413, 415, 422, 503)。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from io import BytesIO
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.agents import agent_framework_status
from app.api.auth_routes import router as auth_router
from app.api.mail_routes import router as mail_router
from app.api.case_routes import router as case_router
from app.api.conversation_routes import router as conversation_router
from app.core.config import get_settings
from app.core.database import get_db
from app.core.security import current_user, require_admin
from app.models.entities import (
    AgentTurnReceipt,
    RunMetrics,
    ToolExecution,
    UserAccount,
)
from app.schemas.dtos import (
    AnswerFeedbackRequest,
    ChatRequest,
    KnowledgeIngestRequest,
    KnowledgeIngestResponse,
    authority,
)
from app.services.ai import finetuned_model_status
from app.services.business_tools import recover_execution
from app.services.chat import ChatService
from app.services.conversations import evidence_detail, save_feedback
from app.services.document_conversion import SUPPORTED_EXTENSIONS
from app.services.knowledge import KnowledgeService
from app.services.memory import SQLAgentMemoryStore
from app.services.report import ReportService, export_reports
from app.services.skills import SolaceAISkillLibrary

router = APIRouter()

router.include_router(auth_router)
router.include_router(mail_router)

# 挂载业务子路由
router.include_router(case_router)
router.include_router(conversation_router)


# ============================================================================
# 系统探针与基础信息接口 (System & Diagnostics)
# ============================================================================

@router.get("/actuator/health")
def health() -> dict[str, str]:
    """系统健康状态探针接口 (Liveness & Readiness Probe)。

    Returns:
        dict: 状态健康指示字典 {"status": "UP"}。
    """
    return {"status": "UP"}


@router.get("/api/profile")
def profile(user: Annotated[UserAccount, Depends(current_user)]) -> dict[str, Any]:
    """获取当前登录用户的个人基本档案与 RBAC 权限角色信息。

    Args:
        user: 当前已认证的用户实体对象。

    Returns:
        dict: 用户 ID、用户名、昵称以及带有 ROLE_ 前缀的标准权限列表。
    """
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "requiresSetup": user.requires_setup,
        "displayName": user.display_name,
        "roles": [authority(role) for role in user.roles],
    }


@router.get("/api/agent/status")
def agent_status(user: Annotated[UserAccount, Depends(current_user)]) -> dict[str, Any]:
    """获取多智能体系统与运行时基座全景健康状态。

    Args:
        user: 当前已认证的用户实体。

    Returns:
        dict: 包含模型路由、Embedding 向量库状态、微调模型状态、Agent 协作拓扑等。
    """
    settings = get_settings()
    provider = "mock" if settings.ai_provider == "mock" else "openai"
    model = "mock" if provider == "mock" else settings.openai_model
    framework = agent_framework_status(settings)
    model_routing = {
        role: {"mode": "model", "primary": {"provider": provider, "model": model}, "fallback": None}
        for role in ("UnderstandingAgent", "SafetyAgent", "ContextAgent", "ResponseAgent")
    }
    return {
        "runtimeProfile": settings.runtime_profile,
        "availableChatBackends": ['cloud','local'] if settings.runtime_profile=='offline' and settings.local_model_switch_enabled else ['cloud'],
        "defaultChatBackend": settings.chat_backend,
        "chatModels": {
            "cloud": {"provider": provider,"model": model},
            **({"local":{"provider":"vllm-metal","model":"solace-response","label":"Qwen3-4B · 回答 LoRA"}} if settings.runtime_profile=='offline' and settings.local_model_switch_enabled else {}),
        },
        "provider": provider,
        "model": model,
        "realModelEnabled": provider in {"ollama", "openai"},
        "paidApiEnabled": settings.paid_api_enabled,
        "agentFramework": framework,
        "agentModelRouting": model_routing,
        "embedding": {
            "enabled": settings.knowledge_vector_enabled,
            "required": settings.knowledge_vector_required,
            "baseUrl": settings.local_rag_url if settings.knowledge_backend == "milvus_bge" else settings.embedding_base_url,
            "model": "BAAI/bge-m3" if settings.knowledge_backend == "milvus_bge" else settings.embedding_model,
            "dimensions": settings.embedding_dimensions,
            "collection": settings.milvus_collection if settings.knowledge_backend == "milvus_bge" else settings.chroma_collection_name,
            "backend": settings.knowledge_backend,
            "sparseType": "bge_m3_learned_sparse" if settings.knowledge_backend == "milvus_bge" else "bm25",
        },
        "finetunedModel": finetuned_model_status(settings),
        "emotionClassifier": {
            "enabled": settings.emotion_classifier_enabled,
            "provider": "vllm-metal" if settings.local_model_switch_enabled else "ollama",
            "model": "solace-emotion" if settings.local_model_switch_enabled else settings.finetuned_model_name,
            "purpose": "辅助情绪观察，不独立决定风险等级",
        },
        "agents": [
            {"name": "MainAgent", "status": "CONFIGURED", "description": "在预算内选择只读工具、观察证据并生成答案"},
            {"name": "SafetyAgent", "status": "CONFIGURED", "description": "检查输入风险，审核实际高风险答案并提供安全兜底"},
            {"name": "RetrievalAgent", "status": "CONFIGURED", "description": "隔离上下文的复合检索角色，只接收检索查询"},
        ],
        "skills": SolaceAISkillLibrary.status_items(),
        "runtimeHarness": {
            "name": "SolaceAIAgentHarness",
            "status": "READY",
            "description": "统一管理单轮 Agent run 的输入脱敏、上下文注入、风险报告、工具计划和 trace 输出",
        },
        "loop": {
            "type": "langgraph-tool-loop",
            "maxSteps": settings.agent_max_decisions,
            "scheduler": "bounded-model-directed",
        },
        "collaboration": {
            "scheduler": "bounded-model-directed",
            "state": "sqlite-checkpointed-state",
            "messageBus": "typed graph state and bounded tool observations",
            "fixedWorkflow": False,
            "agentIsolation": {
                "prompt": "per-agent system prompt",
                "memory": "user-scoped explicit memory and query-only retrieval context",
                "model": "configured API model with separate role prompts",
                "tools": "per-agent tool permissions",
            },
        },
    }


# ============================================================================
# 用户记忆与偏好接口 (User Long-term Memories)
# ============================================================================

@router.get("/api/memories")
def list_memories(
    user: Annotated[UserAccount, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> list[dict[str, Any]]:
    """获取当前用户保存的显式偏好与长期记忆列表。

    Args:
        user: 当前登录用户。
        db: 数据库会话。

    Returns:
        list[dict]: 记忆项列表。
    """
    return SQLAgentMemoryStore(db, get_settings().agent_memory_path).list(user.id)


@router.delete("/api/memories/{memory_id}")
def delete_memory(
    memory_id: int,
    user: Annotated[UserAccount, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, bool]:
    """删除当前用户指定的某条显式记忆。

    Args:
        memory_id: 记忆记录 ID。
        user: 当前登录用户。
        db: 数据库会话。

    Returns:
        dict: {"deleted": True}

    Raises:
        HTTPException: 若记忆不存在或不归属于当前用户则抛出 404。
    """
    if not SQLAgentMemoryStore(db, get_settings().agent_memory_path).delete(user.id, memory_id):
        raise HTTPException(404, "记忆不存在")
    return {"deleted": True}


# ============================================================================
# 咨询对话与流式问答接口 (Chat & Streaming)
# ============================================================================

@router.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    user: Annotated[UserAccount, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> StreamingResponse:
    """学生心理辅导对话核心接口（基于 Server-Sent Events 流式传输）。

    Args:
        request: 包含会话 ID、用户输入文本和历史上下文的对话请求 DTO。
        user: 当前登录学生。
        db: 数据库会话。

    Returns:
        StreamingResponse: SSE 事件流（事件类型包括 delta, citations, report, done 等）。

    Raises:
        HTTPException: 若管理员账号尝试发起对话则拒绝 (403)。
    """
    if "ROLE_ADMIN" in user.roles:
        raise HTTPException(403, "管理员账号只能查看后台记录，不能发起学生对话。")
    from app.services.chat_backends import settings_for_backend
    settings = settings_for_backend(get_settings(), getattr(request, 'modelBackend', None))
    lease = None
    if settings.runtime_profile in {'online', 'light'}:
        from app.services.conversation_access import ConversationAccess
        try:
            access = await run_in_threadpool(ConversationAccess, settings)
            lease = await run_in_threadpool(access.acquire, user.id)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(503, '对话服务暂不可用，请稍后重试。') from None
    try:
        service = ChatService(db, settings)
    except BaseException:
        if lease is not None:
            lease.release()
        raise
    from app.services.conversation_access import protected_stream
    stream = protected_stream(service, user, request, lease) if lease is not None else service.stream_chat(user, request)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/feedback")
def answer_feedback(
    request: AnswerFeedbackRequest,
    user: Annotated[UserAccount, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    """提交针对指定单轮助手回答的质量评价与反馈 (Submit Answer Feedback)。

    Args:
        request: 反馈信息入参。
        user: 当前登录用户。
        db: 数据库会话。

    Returns:
        dict: 保存成功的反馈详情字典。
    """
    return save_feedback(db, user.id, request)


@router.get("/api/evidence/{run_id}/{chunk_id}")
def read_evidence(
    run_id: str,
    chunk_id: int,
    user: Annotated[UserAccount, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    """获取指定问答单轮中引用的知识切片快照详情 (Get Evidence Detail)。

    Args:
        run_id: 单轮 Agent 运行 UUID。
        chunk_id: 引用的知识切片 ID。
        user: 当前登录用户。
        db: 数据库会话。

    Returns:
        dict: 切片正文、来源与元数据快照。
    """
    return evidence_detail(db, user.id, run_id, chunk_id)


# ============================================================================
# 心理评估报告与追踪审查接口 (Reports & Audits)
# ============================================================================

@router.get("/api/reports/me")
def my_reports(
    user: Annotated[UserAccount, Depends(current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> list[Any]:
    """获取当前登录学生账号关联的历史心理健康评估报告列表。

    Args:
        user: 当前登录学生。
        db: 数据库会话。

    Returns:
        list: 该学生的时序心理评估报告列表。
    """
    return ReportService(db).latest_reports(user.id)


@router.get("/api/admin/reports")
def admin_reports(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> list[Any]:
    """获取全校/全系统所有学生近期的心理健康评估报告 (Admin List All Psychological Reports)

    面向心理辅导员与中心管理员，按时间倒序拉取全量学生的心理状态评估报告，
    包含情绪分类、量化分值、风险评级及大模型专业研判摘要。

    Args:
        _: 管理员 RBAC 权限依赖（非管理员直接阻断）。
        db (Session): 数据库事务会话。

    Returns:
        list[ReportResponse]: 全量心理评估报告响应列表。
    """
    return ReportService(db).latest_reports()


@router.get("/api/admin/excel-records")
def admin_excel(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> list[Any]:
    """获取心理评估报告归档至 Excel 电子表格的工具执行记录列表 (List Excel Archive Records)

    审查系统自动化作业将中高风险心理评估同步写入辅导员线下 Excel 台账的执行历史。

    Args:
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。

    Returns:
        list[ToolRecordResponse]: 包含导出状态、物理文件路径与时间戳的记录列表。
    """
    return ReportService(db).excel_records()


@router.get("/api/admin/alerts")
def admin_alerts(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> list[Any]:
    """获取高危心理危机事件告警通知的发送记录列表 (List Crisis Alert Records)

    审查触发 HIGH 风险熔断时，通过邮件或系统广播向值班辅导员外发紧急预警的投递日志。

    Args:
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。

    Returns:
        list[ToolRecordResponse]: 包含告警渠道、接收人列表、发送状态与时间的审计记录。
    """
    return ReportService(db).alert_records()


@router.get("/api/admin/cases")
def admin_cases(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> list[Any]:
    """获取后台高风险心理危机个案列表 (List Active Risk Cases)

    拉取所有处于 OPEN（待跟进）、ALERT_SENT（已预警）、ACKNOWLEDGED（已接手）或 CLOSED 状态的危机干预个案。

    Args:
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。

    Returns:
        list[RiskCaseResponse]: 危机个案流转卡片列表。
    """
    return ReportService(db).risk_cases()


@router.get("/api/admin/cases/{case_id}/notes")
def admin_case_notes(
    case_id: int,
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> list[Any]:
    """获取指定高风险危机个案的人工跟进记录时间线列表 (List Case Follow-Up Notes)

    展示辅导员或咨询师针对该学生线下约谈、心理疏导与家校联动的历史跟进轨迹。

    Args:
        case_id (int): 目标个案自增主键 ID。
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。

    Returns:
        list[CaseNoteResponse]: 按时间序排列的跟进记录列表。
    """
    return ReportService(db).case_notes(case_id)


@router.get("/api/admin/tool-jobs")
def admin_tool_jobs(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> list[Any]:
    """获取后台异步工具队列任务流转记录列表 (List Tool Execution Jobs)

    监控后台任务（如 Excel 台账写入、个案创建、邮件告警）的状态机流转、重试次数及异常原因。

    Args:
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。

    Returns:
        list[ToolJobResponse]: 异步任务列表。
    """
    return ReportService(db).tool_jobs()


@router.get("/api/admin/dead-letters")
def admin_dead_letters(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> list[Any]:
    """获取超过最大重试次数阈值后归档至死信队列的失败任务列表 (List Dead Letter Queue Records)

    针对发生持续网络超时、依赖不可用等致命故障的任务进行事后追踪排查。

    Args:
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。

    Returns:
        list[DeadLetterResponse]: 包含入参快照与致命失败原因的死信记录。
    """
    return ReportService(db).dead_letters()


@router.get("/api/admin/agent-traces")
def admin_agent_traces(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> list[dict[str, Any]]:
    """获取多智能体协同运行链路全景审计追踪记录 (List Agent Run Traces)

    聚合展示包括 Run ID、首字流式输出时间 (TTFT)、总端到端耗时、
    意图识别结果、敏感信息脱敏对比、RAG 召回知识以及决策图的协同步骤快照。

    Args:
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。

    Returns:
        list[dict[str, Any]]: 聚合了度量指标与状态图执行细节的追踪全景列表。
    """
    items = []
    for trace in ReportService(db).agent_run_traces():
        data = trace.model_dump()
        receipt = db.scalar(select(AgentTurnReceipt).where(AgentTurnReceipt.trace_id == trace.id))
        metrics = db.get(RunMetrics, receipt.run_id) if receipt else None
        data["runId"] = receipt.run_id if receipt else None
        data["firstTokenSeconds"] = metrics.first_token_seconds if metrics else None
        data["elapsedSeconds"] = metrics.elapsed_seconds if metrics else None
        items.append(data)
    return items


@router.get("/api/admin/tool-audits")
def admin_tool_audits(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> list[Any]:
    """获取受治理工具调用的安全合规审计历史记录 (List Tool Security Audits)

    审查智能体在执行各种副作用工具前，安全治理拦截器（Policy Engine）的放行与阻断决策。

    Args:
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。

    Returns:
        list[ToolAuditResponse]: 包含命中的策略、是否放行、裁决理由的审计列表。
    """
    return ReportService(db).tool_audits()


@router.get("/api/admin/conversations/{session_id}")
def admin_conversation(
    session_id: str,
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> Any:
    """获取指定会话完整且严格脱敏的历史对话详情（管理员审查视图）(Admin View Sanitized Conversation)

    严格执行隐私保护：对手机号、学号、姓名、证件号进行掩码脱敏后展示完整多轮问答。

    Args:
        session_id (str): 会话 public_id。
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。

    Returns:
        ConversationResponse: 脱敏后的时序对话消息明细。

    Raises:
        HTTPException(404): 会话 ID 不存在时抛出。
    """
    try:
        return ReportService(db).conversation(session_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/api/admin/reports/export")
def download_reports(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    start: str | None = None,
    end: str | None = None,
    risk: str | None = None,
) -> StreamingResponse:
    """管理员专用：筛选并下载心理健康支持评估台账 Excel 文件 (Download Report Ledger)

    支持按起止日期范围与风险等级组合过滤，在内存中动态生成标准 Excel 二进制字节流并回传。

    Args:
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。
        start (str | None, optional): 筛选起始日期 (YYYY-MM-DD)。
        end (str | None, optional): 筛选结束日期 (YYYY-MM-DD)。
        risk (str | None, optional): 风险等级 ('LOW' / 'MEDIUM' / 'HIGH')。

    Returns:
        StreamingResponse: 附带 attachment 头部的 xlsx 电子表格流式响应。

    Raises:
        HTTPException(422): 日期格式非法或风险等级不合法时抛出。
    """
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d") if start else None
        end_date = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1) if end else None
    except ValueError as exc:
        raise HTTPException(422, "日期格式应为 YYYY-MM-DD") from exc

    if risk and risk not in {"LOW", "MEDIUM", "HIGH"}:
        raise HTTPException(422, "风险等级无效")
    if start_date and end_date and start_date >= end_date:
        raise HTTPException(422, "结束日期不能早于开始日期")

    data = export_reports(db, start_date, end_date, risk)
    return StreamingResponse(
        BytesIO(data),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": 'attachment; filename="solaceai-reports.xlsx"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/api/admin/tool-executions")
def tool_executions(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> list[dict[str, Any]]:
    """管理员专用：获取最近 200 条业务工具执行状态机记录与审计结果 (List Tool Executions)

    展示智能体调用的工具名称、入参摘要、返回值、执行终态与幂等键信息。

    Args:
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。

    Returns:
        list[dict[str, Any]]: 格式化的工具执行记录列表。
    """
    records = db.scalars(select(ToolExecution).order_by(ToolExecution.id.desc()).limit(200)).all()
    return [
        {
            "id": row.id,
            "runId": row.run_id,
            "toolName": row.tool_name,
            "status": row.status,
            "error": row.error,
            "result": json.loads(row.result_json or "{}"),
            "createdAt": row.created_at,
            "updatedAt": row.updated_at,
        }
        for row in records
    ]


@router.post("/api/admin/tool-executions/{execution_id}/recover")
def recover_tool(
    execution_id: int,
    actor: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    """管理员专用：对执行失败或被外部中断的业务工具执行手动安全恢复重试 (Recover Tool Execution)

    基于确定性幂等键与上下文参数重新触发工具执行，防止人工重复补录导致数据重叠。

    Args:
        execution_id (int): 失败的工具执行记录 ID。
        actor (UserAccount): 当前操作管理员实体。
        db (Session): 数据库会话。

    Returns:
        dict[str, Any]: 恢复执行后的最新结果字典。

    Raises:
        HTTPException(409): 记录冲突或不允许恢复状态。
        HTTPException(403): 权限不足。
        HTTPException(503): 恢复执行发生底层系统故障。
    """
    try:
        return recover_execution(db, get_settings(), execution_id, actor)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, "工具恢复失败，请查看执行记录") from exc


# ============================================================================
# 知识库管理与摄取接口 (Knowledge Base Management)
# ============================================================================

@router.post("/api/admin/knowledge")
def ingest_knowledge(
    request: KnowledgeIngestRequest,
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> KnowledgeIngestResponse:
    """将管理员提交的纯文本或 Markdown 知识切分并写入知识库 (Ingest Raw Knowledge Text)

    执行 Markdown 标题大纲解析、滑动窗口语义分块、向量计算与数据库落库。

    Args:
        request (KnowledgeIngestRequest): 包含来源标识、正文与可选文档 ID 的请求 DTO。
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。

    Returns:
        KnowledgeIngestResponse: 包含来源文件名与成功生成的切片数。

    Raises:
        HTTPException(422): 文档内容质量校验未达标或解析失败。
    """
    document_id = request.document_id or uuid4().hex
    try:
        chunks = KnowledgeService(db, get_settings()).ingest(
            request.source, request.content, document_id=document_id
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return KnowledgeIngestResponse(source=request.source, chunks=chunks, document_id=document_id)


@router.get("/api/admin/knowledge/status")
def knowledge_status(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    """获取关系型数据库知识切片与向量索引库的一致性同步状态指标 (Knowledge Base Sync Status)

    汇总关系表中已持久化的切片总数、向量库中的向量记录总数、多版本健康度与一致性状态。

    Args:
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。

    Returns:
        dict[str, Any]: 知识库健康指标字典。
    """
    return KnowledgeService(db, get_settings()).status()


@router.post("/api/admin/knowledge/rebuild-vector")
def rebuild_knowledge_vector(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, int]:
    """全量重新构建知识向量索引库 (Disaster Recovery Vector Reindexing)

    灾备恢复机制：清空当前向量索引集合，从关系型数据库全量重新拉取有效切片，
    批量重新生成嵌入向量并重新写入向量索引。

    Args:
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。

    Returns:
        dict[str, int]: 重新建立向量索引成功的切片数量 {"indexedChunks": count}。

    Raises:
        HTTPException(503): 向量重构过程中底层引擎发生异常时抛出。
    """
    try:
        indexed = KnowledgeService(db, get_settings()).rebuild_vector_index()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"indexedChunks": indexed}


@router.post("/api/admin/knowledge/backup")
def backup_knowledge_vector(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    """创建当前知识向量库的可恢复数据快照 (Vector Snapshot Backup)

    将 Chroma/Milvus 向量持久化目录执行原子级快照归档并按保留策略轮转旧快照。

    Args:
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。

    Returns:
        dict[str, Any]: 包含备份快照路径与时间戳的元数据字典。

    Raises:
        HTTPException(503): 备份执行发生 I/O 错误时抛出。
    """
    try:
        snapshot = KnowledgeService(db, get_settings()).backup_vector_index()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"snapshot": snapshot}


@router.post("/api/admin/knowledge/file")
async def ingest_file(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    file: UploadFile = File(..., description="上传的文件对象（上限 10MB）"),
    document_id: str | None = None,
) -> KnowledgeIngestResponse:
    """管理员上传文档文件并解析导入心理知识库 (Upload and Ingest Multi-Format Document)

    支持 PDF (原生文字/扫描版 OCR)、Word (.docx)、PowerPoint (.pptx)、Excel (.xlsx)、
    HTML、纯文本等多种异构格式，通过统一版面解析管道（Docling）进行智能转录与结构化切块。

    Args:
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。
        file (UploadFile): 上传的多模态文件。
        document_id (str | None, optional): 指定文档稳定标识。

    Returns:
        KnowledgeIngestResponse: 摄入成功报告。

    Raises:
        HTTPException(413): 上传文件体积超过 10MB 上限。
        HTTPException(415): 文件扩展名不属于受支持的文档格式。
        HTTPException(422): 文档内容损坏或质量低于入库阈值。
    """
    data = await file.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件不能超过10MB")
    if not (file.filename or "").lower().endswith(tuple(SUPPORTED_EXTENSIONS) + (".metadata.json",)):
        raise HTTPException(415, "支持PDF、Word、PPTX、XLSX、文本、HTML、CSV、图片及转换元数据")
    document_id = document_id or uuid4().hex
    try:
        chunks = await run_in_threadpool(
            KnowledgeService(db, get_settings()).ingest_file,
            file.filename or "uploaded-file",
            data,
            document_id=document_id,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return KnowledgeIngestResponse(
        source=file.filename or "uploaded-file",
        chunks=chunks,
        document_id=document_id,
    )


@router.post("/api/admin/knowledge/preview")
async def preview_knowledge(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    file: UploadFile = File(..., description="待预览的文件对象（上限 10MB）"),
) -> Any:
    """管理员专用：上传文件并在不写入数据库的前提下预览解析出的 Markdown 与元数据 (Preview Document Parsing)

    用于管理员在正式入库前检查版面转录质量、标题大纲抽取正确性与字符识别准确度。

    Args:
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。
        file (UploadFile): 上传的待预览文件。

    Returns:
        dict: 包含转录 Markdown 文本、标题大纲层级结构与解析质量评分的字典。

    Raises:
        HTTPException(413): 文件超过 10MB。
        HTTPException(422): 解析器执行失败或文档格式损坏。
    """
    data = await file.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件不能超过10MB")
    try:
        return await run_in_threadpool(
            KnowledgeService(db, get_settings()).preview_file,
            file.filename or "uploaded-file",
            data,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/api/admin/knowledge/documents")
def knowledge_documents(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> list[Any]:
    """管理员专用：查询当前知识库中所有已索引文档的切片分布与版本历史 (List Knowledge Documents)

    展示各文档的物理来源名称、所属文档 ID、最新版本状态、切片数量统计及创建时间。

    Args:
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。

    Returns:
        list[dict]: 聚合文档元信息列表。
    """
    return KnowledgeService(db, get_settings()).documents()


@router.get("/api/admin/knowledge/document-content")
def knowledge_document_content(
    document_id: str,
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    offset: int = 0,
) -> dict:
    """管理员专用：分页检索特定知识文档的切片文本明细 (Get Knowledge Document Chunks)

    执行逻辑：
    1. 校验管理员鉴权；
    2. 基于 `document_id` 过滤当前租户与知识库范围内的切片记录；
    3. 支持按切片原始索引 `source_index` 升序分页加载（单页固定 50 条）。

    Args:
        document_id (str): 目标知识文档唯一哈希标识。
        _ (UserAccount): 当前已完成管理员鉴权的用户。
        db (Session): 数据库会话。
        offset (int, optional): 分页偏移量，默认为 0。

    Returns:
        dict: 包含文档切片总数 total 与单页切片文本数组 chunks 的字典。

    Raises:
        HTTPException(404): 文档不存在或无切片。
        HTTPException(422): 分页位置非法为负数。
    """
    from app.models.entities import KnowledgeChunk
    service = KnowledgeService(db, get_settings())
    rows = service._chunk_query().filter(KnowledgeChunk.document_id == document_id)
    total = rows.count()
    if not total:
        raise HTTPException(404, '未找到文档内容')
    if offset < 0:
        raise HTTPException(422, '分页位置不能为负数')
    chunks = rows.order_by(KnowledgeChunk.source_index, KnowledgeChunk.id).offset(offset).limit(50).all()
    return {'total': total, 'chunks': [{'id': row.id, 'content': row.content} for row in chunks]}


@router.delete("/api/admin/knowledge/documents")
def delete_knowledge_document(
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
    source: str | None = None,
    document_id: str | None = None,
) -> dict[str, int]:
    """管理员专用：级联物理删除知识文档及其切片与向量索引 (Delete Knowledge Document)

    执行连带清理：物理移除关系表切片、元数据、版本记录以及 Chroma/Milvus 向量索引中的对应条目。

    Args:
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。
        source (str | None, optional): 来源文件名。
        document_id (str | None, optional): 文档唯一标识 SHA-256。

    Returns:
        dict[str, int]: 被删除的物理切片总数量 {"deletedChunks": count}。

    Raises:
        HTTPException(422): 未指定 source 且未指定 document_id 时抛出。
    """
    try:
        count = KnowledgeService(db, get_settings()).delete_document(source, document_id=document_id)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"deletedChunks": count}


@router.post("/api/admin/knowledge/versions/{version_id}/retry")
def retry_knowledge_version(
    version_id: int,
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, int]:
    """重试处于异常或挂起状态的知识文档版本解析构建 (Retry Failed Document Version)

    针对因偶发超时、网络波动导致解析构建失败的版本触发异步重新解析与切片入库。

    Args:
        version_id (int): 待重试的文档版本记录主键 ID。
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。

    Returns:
        dict[str, int]: 重试构建生成的切片总数量 {"chunks": count}。

    Raises:
        HTTPException(422): 版本 ID 不存在或该状态不支持重试。
    """
    try:
        return {"chunks": KnowledgeService(db, get_settings()).retry_version(version_id)}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/api/admin/knowledge/versions/{version_id}/cleanup")
def cleanup_knowledge_version(
    version_id: int,
    _: Annotated[UserAccount, Depends(require_admin)],
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, int]:
    """物理清理历史或失败的孤儿知识切片版本记录 (Cleanup Orphan Document Version)

    安全清理已归档 (ARCHIVED) 或构建失败 (FAILED) 的历史切片残余，释放数据库存储空间。

    Args:
        version_id (int): 待清理的文档版本 ID。
        _: 管理员鉴权依赖。
        db (Session): 数据库会话。

    Returns:
        dict[str, int]: 清理释放的切片记录数量 {"chunks": count}。

    Raises:
        HTTPException(422): 版本记录不存在或处于激活状态不允许清理。
    """
    try:
        return {"chunks": KnowledgeService(db, get_settings()).cleanup_version(version_id)}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc
