"""SolaceAI 单轮对话多智能体运行宿主与业务编排器 (SolaceAI Agent Harness)

本模块定义了 SolaceAI 系统中驱动单轮对话端到端业务闭环的核心编排服务 `SolaceAIAgentHarness`。
负责协调用户请求接入、跨进程并发控制、个人隐私脱敏、多智能体协同运行、结果收据防重以及副作用工具安全派发。

核心架构设计与执行时序：
1. 分布式并发与会话排他锁 (Turn & Session Locking):
   基于文件系统 `fcntl.flock` 互斥锁，分别锁定当前会话 (`session:{user_id}:{session_id}`) 与运行轮次 (`run:{user_id}:{run_id}`)，
   防止多 Worker 进程或网络抖动导致的并发重叠执行与状态覆盖。
2. 隐私数据脱敏治理 (PII Sanitization):
   调用 `PrivacySanitizer` 剥离用户输入中的手机号、身份证、真实姓名等个人可识别敏感信息，
   确保流向外部模型提供商的请求仅包含脱敏后的 `model_input`。
3. 执行收据与幂等重放 (Turn Receipt Deduplication):
   通过 `AgentTurnReceipt` 记录单轮输入 SHA-256 摘要与执行状态。若接收到重复重试请求且该轮次已生成最终回复，
   直接命中缓存返回 `_cached_outcome`，避免昂贵的重复大模型调用与副作用多次触发。
4. 智能体运行时托管 (Runtime Orchestration):
   动态构建并托管 `GraphAgentRuntime`（基于 LangGraph），注入全局中断信号 (`cancel_event`) 与流式事件通道 (`event_sink`)。
5. 事务性结果持久化 (Atomic Persistence & Tool Dispatch):
   在同一事务生命周期内，原子完成用户消息存储、心理风险测评报告写入、端到端全链路 Trace 落库以及向 MCP 工具服务器分发副作用任务。
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.agents import create_agent_runtime
from app.agents.result import AgentStep
from app.core.config import Settings
from app.core.enums import IntentType, MessageRole
from app.models.entities import (
    AgentTurnReceipt,
    ChatMessage,
    ChatSession,
    PsychologicalReport,
    UserAccount,
)
from app.schemas.dtos import AiMessage, ChatRequest
from app.services.ai import AgentModelRoute, default_response_route
from app.services.assessment import PsychologyAssessment
from app.services.knowledge import SearchResult
from app.services.memory import RedisShortTermMemoryStore
from app.services.privacy import PrivacySanitizer
from app.services.trace import AgentTraceService
from app.services.answer_trace import TraceRecorder, load_answer_trace

logger = logging.getLogger(__name__)


@dataclass
class AgentToolPlan:
    """副作用工具执行计划 DTO。

    Attributes:
        report_id: 关联触发的心理健康评估报告 ID。
        risk_level: 评估得出的安全风险等级 (如 LOW / MEDIUM / HIGH)。
    """

    report_id: int | None
    risk_level: str | None

    @property
    def requires_tools(self) -> bool:
        """判断当前执行轮次是否需要触发后续工具副作用操作。

        Returns:
            bool: 若存在有效关联报告则返回 True，否则返回 False。
        """
        return self.report_id is not None


@dataclass
class AgentHarnessOutcome:
    """Harness 执行单轮业务闭环后的综合业务结果对象。

    Attributes:
        session: 当前对话会话持久化实体。
        original_input: 用户本轮原始输入纯文本。
        model_input: 经过 PII 隐私脱敏后的模型输入文本。
        intent: 研判识别的意图类型枚举。
        risk_level: 最终安全风控等级字符串。
        assessment: 结构化心理评估快照。
        response_messages: 模型生成并回传的消息列表。
        response_model_route: 回复阶段采用的模型路由配置元数据。
        agent_steps: 智能体多步协同决策与工具观测链路详情。
        retrieved_knowledge: 检索并引用的知识库切片集合。
        report_id: 关联生成的心理报告 ID。
        tool_plan: 计划执行的业务工具副作用任务。
        trace_id: 全链路追踪审计记录 ID。
        final_response: 最终审核通过后下发的回复文本。
        citations: 引用的真实切片元数据列表。
        run_id: 本轮 Agent 执行全局唯一 UUID。
    """

    session: ChatSession
    original_input: str
    model_input: str
    intent: IntentType
    risk_level: str | None
    assessment: PsychologyAssessment | None
    response_messages: list[AiMessage]
    response_model_route: AgentModelRoute
    agent_steps: list[AgentStep]
    retrieved_knowledge: list[SearchResult]
    report_id: int | None
    tool_plan: AgentToolPlan
    trace_id: int | None
    final_response: str | None = None
    citations: list[dict] | None = None
    run_id: str = ""
    trace_status: str = "completed"


class SolaceAIAgentHarness:
    """智能体运行宿主与业务编排核心服务 (Agent Harness Runtime)。

    负责承接 API 控制层请求，在严格的并发锁与事务保护下组织智能体完成单轮问答、
    意图研判、风险审核与数据持久化。
    """

    def __init__(self, db: Session, settings: Settings) -> None:
        """初始化 SolaceAIAgentHarness 运行宿主。

        Args:
            db: SQLAlchemy 数据库事务会话。
            settings: 全局运行时配置对象。
        """
        self.db = db
        self.settings = settings
        self.privacy = PrivacySanitizer()
        self.memory = RedisShortTermMemoryStore(settings)

    def run(self, user: UserAccount, request: ChatRequest) -> AgentHarnessOutcome:
        """跨进程文件锁保护的单轮对话多智能体业务运行入口。

        通过加锁会话键与运行键，实现并发互斥，避免同一会话多轮输入竞态破坏历史。

        Args:
            user: 请求用户账号实体。
            request: 对话请求入参 DTO。

        Returns:
            AgentHarnessOutcome: 完整的业务执行结果包。

        Raises:
            RuntimeError: 当检测到该会话或运行轮次已在执行中时，抛出冲突提示。
        """
        import fcntl
        from contextlib import ExitStack

        request = copy.copy(request)
        request.runId = getattr(request, "runId", None) or uuid.uuid4().hex

        # 本地 SQLite 检查点依赖共享文件系统；使用 flock 序列化跨 Worker 的重叠重试，进程退出时自动释放。
        lock_root = Path(self.settings.agent_checkpoint_path).parent / "turn-locks"
        lock_root.mkdir(parents=True, exist_ok=True)

        keys = [f"run:{user.id}:{request.runId}"]
        if getattr(request, "sessionId", None):
            keys.insert(0, f"session:{user.id}:{request.sessionId}")

        with ExitStack() as stack:
            for key in keys:
                name = hashlib.sha256(key.encode()).hexdigest()
                lock = stack.enter_context((lock_root / f"{name}.lock").open("a"))
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError("This session or turn is still running; retry later") from exc

            return self._run(user, request)

    def _run(self, user: UserAccount, request: ChatRequest) -> AgentHarnessOutcome:
        """在排他锁内执行单次用户交互的完整业务与智能体流程。

        Args:
            user: 当前认证用户。
            request: 用户对话请求体。

        Returns:
            AgentHarnessOutcome: 业务闭环执行结果。

        Raises:
            ValueError: 当检测到非法的会话越权、输入篡改或不存在的会话时。
        """
        original_input = request.message.strip()
        run_id = getattr(request, "runId", None) or uuid.uuid4().hex
        receipt = self.db.get(AgentTurnReceipt, run_id)
        digest = hashlib.sha256(original_input.encode()).hexdigest()

        if receipt:
            if receipt.user_id != user.id or receipt.input_hash != digest:
                raise ValueError("Run not found or input changed")
            session = self.db.get(ChatSession, receipt.session_id)
            from app.services.conversations import ConversationService

            if ConversationService(self.db).is_hidden(user.id, session.id):
                raise ValueError("Session not found")
            if request.sessionId and request.sessionId != session.public_id:
                raise ValueError("Run session mismatch")
            if receipt.final_response is not None:
                sink = getattr(self, 'event_sink', None)
                if sink:
                    sink({'type': 'meta', 'sessionId': session.public_id, 'runId': run_id})
                    prior = load_answer_trace(self.db, user.id, session.id, run_id)
                    for step in (prior or {}).get('steps', []):
                        sink({'type': 'step', 'sessionId': session.public_id, 'runId': run_id, 'step': step})
                return self._cached_outcome(session, original_input, self.privacy.sanitize(original_input), receipt)
        else:
            session = self._resolve_session(user, request.sessionId, original_input)
            receipt = AgentTurnReceipt(
                run_id=run_id,
                user_id=user.id,
                session_id=session.id,
                input_hash=digest,
            )
            self.db.add(receipt)
            self.db.commit()

        sink = getattr(self, 'event_sink', None)
        cancel = getattr(self, 'cancel_event', None)
        self.public_trace = TraceRecorder(self.db, user.id, session.id, session.public_id, run_id, sink, cancel, allow_details=self.settings.runtime_profile == 'offline')
        if sink:
            sink({'type': 'meta', 'sessionId': session.public_id, 'runId': run_id})
        try:
            with self.public_trace.stage('sanitize'):
                model_input = self.privacy.sanitize(original_input)
            outcome = self._execute_run(user, session, original_input, model_input, run_id, receipt)
        except BaseException:
            self.db.rollback()
            stopped = bool(cancel and cancel.is_set())
            self.public_trace.finish('cancelled' if stopped else 'failed')
            if not receipt.user_message_saved:
                self.save_message(user, session, MessageRole.USER, original_input, receipt=receipt)
            self._save_interrupted_message(user, session, receipt, '本轮已停止。' if stopped else '这次回复未能完成，请稍后重试。')
            raise
        stopped = bool(cancel and cancel.is_set())
        self.public_trace.finish('cancelled' if stopped else 'completed')
        outcome.trace_status = self.public_trace.row.status
        if stopped:
            self._save_interrupted_message(user, session, receipt, '本轮已停止。')
        return outcome

    def _execute_run(self, user, session, original_input, model_input, run_id, receipt):
        runtime = create_agent_runtime(self.db, self.settings, redis_memory=self.memory)
        runtime.cancel_event = getattr(self, "cancel_event", None)
        runtime.event_sink = getattr(self, "event_sink", None)
        runtime.public_trace = self.public_trace
        runtime.restart_interrupted = self.public_trace.previous_status in {'failed', 'cancelled'}
        knowledge = getattr(runtime, 'knowledge', None)
        if knowledge is not None:
            knowledge.public_trace = self.public_trace

        if getattr(runtime, "framework_name", "") == "langgraph":
            agent_run = runtime.run(user, session, original_input, model_input, run_id=run_id)
        else:
            agent_run = runtime.run(user, session, original_input, model_input)

        if not receipt.user_message_saved:
            self.save_message(user, session, MessageRole.USER, original_input, receipt=receipt)

        with self.public_trace.stage('report') as report_step:
            report = (
                self.db.get(PsychologicalReport, receipt.report_id)
                if receipt.report_id
                else self._create_report(user, session, original_input, agent_run, receipt=receipt)
            )
            if report is None:
                report_step['status'] = 'skipped'
        risk_level = report.risk_level if report is not None else None
        trace = AgentTraceService(self.db).save_run(
            user=user,
            session=session,
            original_input=original_input,
            sanitized_input=model_input,
            memory_brief=agent_run.memory_brief,
            agent_run=agent_run,
            report_id=report.id if report is not None else None,
        )
        tool_plan = AgentToolPlan(
            report_id=report.id if report is not None else None,
            risk_level=risk_level,
        )

        if risk_level in {"MEDIUM", "HIGH"}:
            import asyncio

            try:
                asyncio.run(self.dispatch_tools(tool_plan))
            except Exception:
                # ToolExecution 表中保留执行失败记录供管理员审计与手动恢复
                # 外部通知发送失败绝不能压制或阻断已通过安全审查的学生回复
                logger.exception("Required business tool failed; reviewed reply remains available")

        receipt.trace_id = trace.id
        receipt.intent = agent_run.intent.value
        receipt.risk = risk_level
        receipt.final_response = getattr(agent_run, "final_response", None)
        receipt.citations_json = json.dumps(getattr(agent_run, "citations", []), ensure_ascii=False)
        self.db.add(receipt)
        self.db.commit()

        return AgentHarnessOutcome(
            session=session,
            original_input=original_input,
            model_input=model_input,
            intent=agent_run.intent,
            risk_level=risk_level,
            assessment=agent_run.assessment,
            response_messages=agent_run.response_messages,
            response_model_route=agent_run.response_model_route,
            agent_steps=agent_run.steps,
            retrieved_knowledge=agent_run.retrieved_knowledge,
            report_id=report.id if report is not None else None,
            tool_plan=tool_plan,
            trace_id=trace.id,
            final_response=getattr(agent_run, "final_response", None),
            citations=getattr(agent_run, "citations", []),
            run_id=run_id,
        )

    def _cached_outcome(
        self,
        session: ChatSession,
        original_input: str,
        model_input: str,
        receipt: AgentTurnReceipt,
    ) -> AgentHarnessOutcome:
        """根据数据库持久化的历史运行收据重放缓存的执行结果，支持幂等重放。

        Args:
            session: 当前关联的会话对象。
            original_input: 用户原始请求文本。
            model_input: 脱敏后的文本。
            receipt: 已存在的业务收据记录实体。

        Returns:
            AgentHarnessOutcome: 还原构造的完整历史业务结果。
        """
        return AgentHarnessOutcome(
            session=session,
            original_input=original_input,
            model_input=model_input,
            intent=IntentType(receipt.intent),
            risk_level=receipt.risk,
            assessment=None,
            response_messages=[],
            response_model_route=default_response_route(self.settings),
            agent_steps=[],
            retrieved_knowledge=[],
            report_id=receipt.report_id,
            tool_plan=AgentToolPlan(receipt.report_id, receipt.risk),
            trace_id=receipt.trace_id,
            final_response=receipt.final_response,
            citations=json.loads(receipt.citations_json),
            run_id=receipt.run_id,
            trace_status=(load_answer_trace(self.db, session.user_id, session.id, receipt.run_id) or {}).get('status', 'completed'),
        )

    def save_assistant_message(
        self,
        user: UserAccount,
        session: ChatSession,
        content: str,
        run_id: str = "",
    ) -> None:
        """持久化存储智能体流式生成完毕的完整回复文本。

        Args:
            user: 当前用户实体。
            session: 目标会话实体。
            content: 助手端生成的自然语言回复内容。
            run_id: 关联的单轮运行 UUID。

        Raises:
            ValueError: 若传入的 run_id 所属用户或会话与当前调用不匹配。
        """
        receipt = self.db.get(AgentTurnReceipt, run_id) if run_id else None
        if receipt:
            if receipt.user_id != user.id or receipt.session_id != session.id:
                raise ValueError("Run owner mismatch")
            changed = self.db.query(AgentTurnReceipt).filter(
                AgentTurnReceipt.run_id == run_id,
                AgentTurnReceipt.user_id == user.id,
                AgentTurnReceipt.assistant_saved.is_(False),
            ).update(
                {"assistant_saved": True, "final_response": content},
                synchronize_session=False,
            )
            if not changed:
                self.db.rollback()
                return

        self._upsert_bound_message(user, session, content, run_id)

    def _save_interrupted_message(self, user, session, receipt, content):
        """保存中断的中间消息内容与执行收据，保留可见轨迹但不将其标记为成功缓存 (Save Interrupted Message)

        Args:
            user: 当前用户实体。
            session: 当前会话实体。
            receipt: 轮次执行收据实体。
            content: 中断前已生成的增量文本内容。
        """
        receipt.final_response = None
        receipt.assistant_saved = False
        self.db.add(receipt)
        self._upsert_bound_message(user, session, content, receipt.run_id)

    def _upsert_bound_message(self, user, session, content, run_id):
        """插入或更新绑定到特定 run_id 的助手回复消息 (Upsert Bound Assistant Message)

        若已存在对应 run_id 的消息记录则更新正文并使短期缓存失效；否则新建持久化消息与关联映射。

        Args:
            user: 当前用户实体。
            session: 当前会话实体。
            content: 消息内容文本。
            run_id: 运行追踪唯一 UUID。

        Returns:
            int: 消息主键 ID。
        """
        from app.models.entities import ChatMessageRun
        message = None
        if run_id:
            message = self.db.query(ChatMessage).join(
                ChatMessageRun, ChatMessageRun.message_id == ChatMessage.id
            ).filter(
                ChatMessageRun.run_id == run_id, ChatMessageRun.user_id == user.id,
                ChatMessage.user_id == user.id, ChatMessage.session_id == session.id,
                ChatMessage.role == MessageRole.ASSISTANT.value,
            ).first()
        if message is not None:
            message.content = content
            session.touch()
            self.db.add(message)
            self.db.add(session)
            self.db.commit()
            try:
                self.memory.evict(user.id, session.public_id)
            except Exception:
                logger.warning('Failed to evict interrupted history cache')
            return message.id
        message_id = self.save_message(user, session, MessageRole.ASSISTANT, content)
        if run_id:
            self.db.add(ChatMessageRun(message_id=message_id, user_id=user.id, run_id=run_id))
            self.db.commit()
        return message_id

    async def dispatch_tools(self, tool_plan: AgentToolPlan) -> list[str]:
        """根据工具计划分发并执行业务副作用操作（如建立危机个案与发送告警）。

        Args:
            tool_plan: 预先制定的工具执行计划。

        Returns:
            list[str]: 各工具调用的序列化返回值 JSON 字符串列表。
        """
        if tool_plan.report_id is None:
            return []

        from app.services.business_tools import ToolContext, business_registry, execute_registered
        from app.services.mcp import McpClient

        report = self.db.get(PsychologicalReport, tool_plan.report_id)
        if report is None:
            return []

        context = ToolContext(report.user_id, str(report.session_id), f"report:{report.id}")
        registry = business_registry(self.db, self.settings, context)

        # 收据处理句柄在请求主线程上执行；MCP 仅作为受控消息传输协议通道
        wrapped = type(registry)()
        for name, tool in registry.tools.items():
            wrapped.register(
                name,
                tool.description,
                tool.schema,
                lambda args, name=name: execute_registered(self.db, context, registry, name, args),
            )

        results = []
        with McpClient(wrapped) as connection:
            if tool_plan.risk_level in {"MEDIUM", "HIGH"}:
                from contextlib import nullcontext
                trace = getattr(self, 'public_trace', None)
                with trace.stage('case') if trace else nullcontext():
                    case = connection.call("create_risk_case", {"report_id": report.id})
                results.append(json.dumps(case, ensure_ascii=False))
                if tool_plan.risk_level == "HIGH":
                    with trace.stage('email') if trace else nullcontext({}) as email_step:
                        delivery = connection.call("send_risk_alert", {"case_id": case["case_id"]})
                        if delivery.get('status') == 'SUCCESS':
                            email_step['resultCode'] = 'smtp_accepted'
                        if delivery.get('status') == 'LOG_ONLY':
                            email_step.update(status='skipped', resultCode='log_only')
                        results.append(json.dumps(delivery, ensure_ascii=False))
        return results

    def save_message(
        self,
        user: UserAccount,
        session: ChatSession,
        role: MessageRole,
        content: str,
        receipt: AgentTurnReceipt | None = None,
    ) -> None:
        """持久化消息记录至关系型数据库，并同步追加至 Redis 短期记忆缓冲中。

        Args:
            user: 消息归属用户。
            session: 目标会话。
            role: 发送方角色 (user / assistant / system)。
            content: 消息纯文本内容。
            receipt: 可选的单轮执行收据实体。
        """
        message = ChatMessage(user_id=user.id, session_id=session.id, role=role.value, content=content)
        self.db.add(message)
        session.touch()
        self.db.add(session)
        if receipt is not None:
            receipt.user_message_saved = True
            self.db.add(receipt)
        self.db.commit()
        try:
            self.memory.evict(user.id, session.public_id)
        except Exception as exc:
            logger.warning("Failed to evict history cache: %s", exc)
        return message.id

    def _resolve_session(
        self,
        user: UserAccount,
        public_id: str | None,
        text: str,
    ) -> ChatSession:
        """解析已存在的会话实体或为用户创建全新会话。

        Args:
            user: 请求用户。
            public_id: 客户端传入的可选会话 UUID。
            text: 首轮输入文本（用于生成会话标题前缀）。

        Returns:
            ChatSession: 确定归属的会话实体。

        Raises:
            ValueError: 若传入了 public_id 但会话不存在或已归档隐藏。
        """
        if public_id:
            session = self.db.query(ChatSession).filter(
                ChatSession.public_id == public_id,
                ChatSession.user_id == user.id,
            ).first()
            if session is None:
                raise ValueError("Session not found")
            from app.services.conversations import ConversationService

            if ConversationService(self.db).is_hidden(user.id, session.id):
                raise ValueError("Session not found")
            return session

        session = ChatSession(public_id=uuid.uuid4().hex, user_id=user.id, title=text[:36])
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def _create_report(
        self,
        user: UserAccount,
        session: ChatSession,
        text: str,
        agent_run: Any,
        receipt: AgentTurnReceipt | None = None,
    ) -> PsychologicalReport | None:
        """持久化生成心理健康评估报告实体，并在同事务中原子关联执行收据。

        Args:
            user: 评估对象学生。
            session: 当前会话。
            text: 学生本轮原始输入。
            agent_run: 智能体运行输出。
            receipt: 可选的运行收据。

        Returns:
            PsychologicalReport | None: 若触发并生成了报告则返回其实体，否则返回 None。
        """
        if not agent_run.requires_report or agent_run.assessment is None:
            return None

        report = PsychologicalReport(
            user_id=user.id,
            session_id=session.id,
            content=text,
            intent=agent_run.intent.value,
            emotion=agent_run.assessment.emotion.value,
            emotion_score=agent_run.assessment.emotion_score,
            risk_level=agent_run.assessment.risk.value,
            summary=agent_run.assessment.summary,
        )
        self.db.add(report)
        self.db.flush()

        if receipt is not None:
            receipt.report_id = report.id
            self.db.add(receipt)

        self.db.commit()
        self.db.refresh(report)
        return report
