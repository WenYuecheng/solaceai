"""SolaceAI 对话编排与流式响应服务 (Conversation Orchestration & Streaming Service)

本模块作为驱动心理咨询交互流式体验的核心服务，承担以下关键职责：
1. 会话并发防重互斥 (`_ACTIVE_TURNS`)：防止同一用户在同一会话中并发连击发送多条消息导致状态混乱；
2. 跨线程异步调度：主事件循环接收 HTTP 流式请求，将重量级 Agent 运行时委托给独立工作线程
   (`_run_in_worker`)，通过线程安全的 `asyncio.Queue` 实现非阻塞事件流推送；
3. SSE 增量推送与协议适配：
   - 'meta': 推送会话 public_id 与单轮追踪 run_id；
   - 'token': 逐字推送 AI 生成文本块；
   - 'citation': 推送回复中引用的真实知识切片快照；
   - 'replace': 安全审核发现违规时，在前端原子替换为合规安全话术；
   - 'done': 推送单轮耗时、TTFT 首字延迟及持久化消息 ID；
4. 审计与度量指标落库：记录端到端耗时 (`elapsed_seconds`) 与首字流式延迟 (`first_token_seconds`)。
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from sqlalchemy.orm import Session

from app.agents.harness import SolaceAIAgentHarness
from app.core.config import Settings
from app.models.entities import UserAccount
from app.schemas.dtos import ChatRequest, ChatStreamEvent
from app.services.ai import AiClient, client_for_route
from app.services.citations import citation_payloads

logger = logging.getLogger(__name__)

# 活跃对话轮次互斥集合：存储 (user_id, session_key) 二元组
_ACTIVE_TURNS: set[tuple[int, str]] = set()
_ACTIVE_LOCK = threading.Lock()


class ChatService:
    """心理咨询与日常对话核心编排服务 (Chat Orchestration Service)

    统一协调大模型客户端、Agent 运行宿主 Harness 以及流式响应协议。
    """

    def __init__(self, db: Session, settings: Settings):
        """初始化对话编排服务实例

        Args:
            db (Session): SQLAlchemy 数据库会话。
            settings (Settings): 全局配置实例。
        """
        self.db = db
        self.settings = settings
        self.ai = AiClient(settings)
        self.agent_harness = SolaceAIAgentHarness(db, settings)

    async def stream_chat(self, user: UserAccount, request: ChatRequest):
        """流式对话入口生成器 (Streaming Chat Entrypoint)

        负责会话并发防重锁控制：若当前会话上一轮尚未生成结束，返回 SESSION_BUSY 错误。

        Args:
            user (UserAccount): 当前已认证用户实体。
            request (ChatRequest): 对话请求体。

        Yields:
            str: 格式化的 SSE 事件字符串。
        """
        key = (user.id, getattr(request, 'sessionId', None) or getattr(request, 'runId', None) or 'new')
        with _ACTIVE_LOCK:
            busy = key in _ACTIVE_TURNS
            if not busy:
                _ACTIVE_TURNS.add(key)
        if busy:
            yield sse('error', {'type': 'error', 'code': 'SESSION_BUSY', 'message': '本次对话正在回复，请等待完成。'})
            return
        try:
            async for event in self._stream_chat(user, request):
                yield event
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE_TURNS.discard(key)

    async def _stream_chat(self, user: UserAccount, request: ChatRequest):
        """端到端流式对话内部生成器 (Async Streaming Generator Implementation)

        执行步骤：
        1. 发送初始状态事件并启动工作线程执行 Agent 协同图；
        2. 消费事件队列，逐块向客户端产出 token 文本；
        3. 推送会话元数据与切片引用快照；
        4. 处理安全审核替换；
        5. 在独立线程保存助手消息与性能度量指标。

        Args:
            user (UserAccount): 当前登录用户。
            request (ChatRequest): 对话请求对象。

        Yields:
            str: 格式化的 SSE 事件报文。
        """
        # 1. 运行智能体沙箱，获取执行决策结果与准备就绪的响应 Prompt
        yield sse("status", {"type": "status", "message": "正在理解问题并整理依据"})
        cancel_event = threading.Event()
        loop = asyncio.get_running_loop()
        events = asyncio.Queue()
        streamed = []
        started = time.monotonic()
        first_token_seconds = None

        def emit(event):
            """线程安全的事件发送回调，将工作线程产生的事件安全压入异步事件队列。"""
            if not cancel_event.is_set():
                try:
                    loop.call_soon_threadsafe(events.put_nowait, event)
                except RuntimeError:
                    cancel_event.set()

        async def execute():
            """在工作线程池中异步调度 Agent 完整生命周期执行，并在结束时写入哨兵标记 None。"""
            try:
                return await asyncio.to_thread(self._run_in_worker, user, request, cancel_event, emit)
            finally:
                await events.put(None)

        task = asyncio.create_task(execute())
        try:
            while True:
                event = await events.get()
                if event is None:
                    break
                kind = event.get('type', 'status')
                if kind == 'token':
                    streamed.append(event.get('content', ''))
                    if first_token_seconds is None:
                        first_token_seconds = time.monotonic() - started
                yield sse(kind, event)
            outcome = await task
        except (asyncio.CancelledError, GeneratorExit):
            cancel_event.set()
            task.cancel()
            raise
        except Exception:
            logger.error('Agent run failed', exc_info=True)
            yield sse('error', {'type': 'error', 'code': 'RUN_FAILED', 'message': '这次回复未能完成，请稍后重试。'})
            return
        finally:
            cancel_event.set()

        # 2. 推送会话元数据（确保前端获取最新分配或延续的 sessionId 与 runId）
        yield sse("meta", {"type": "meta", "sessionId": outcome.session.public_id, "runId": getattr(outcome, 'run_id', '')})

        # 3. 组织并推送引用快照
        assistant = list(streamed)
        citations = getattr(outcome, "citations", None) or []
        if citations:
            yield sse("citation", {"type": "citation", "citations": citation_payloads(citations, getattr(outcome, "final_response", ""))})

        try:
            final = getattr(outcome, "final_response", None)
            if streamed:
                # 若已流式输出但最终安全审核修正了话术，发送 replace 事件指令前端整体替换
                if final is not None and ''.join(streamed) != final:
                    yield sse('replace', {'type': 'replace', 'content': final, 'sessionId': outcome.session.public_id})
            elif final is not None:
                # 图工作流生成的最终响应已在前置 review 节点完成合规审计，直接切片分块流式输出
                for start in range(0, len(final), 32):
                    token = final[start:start + 32]
                    assistant.append(token)
                    if first_token_seconds is None:
                        first_token_seconds = time.monotonic() - started
                    yield sse("token", ChatStreamEvent(type="token", sessionId=outcome.session.public_id, content=token).model_dump())
                    await asyncio.sleep(0)
            else:
                # 兼容旧模式：调用模型流式接口补全
                response_client = self._response_client(outcome)
                async for token in response_client.stream(outcome.response_messages):
                    assistant.append(token)
                    if first_token_seconds is None:
                        first_token_seconds = time.monotonic() - started
                    yield sse("token", ChatStreamEvent(type="token", sessionId=outcome.session.public_id, content=token).model_dump())
        except Exception:
            logger.error("Reply generation failed", exc_info=True)
            yield sse("error", {"type": "error", "message": "回复生成中断，请稍后重试。"})
            return

        # 4. 持久化助手完整回复
        message_id = None
        if assistant:
            message_id = await asyncio.to_thread(self._save_in_worker, user, outcome, final if final is not None else "".join(assistant))

        # 5. 持久化单轮性能指标 (TTFT 与端到端耗时)
        if isinstance(self.db, Session) and getattr(outcome, 'run_id', ''):
            await asyncio.to_thread(
                self._save_metrics,
                user.id,
                outcome.run_id,
                first_token_seconds,
                time.monotonic() - started,
            )

        # 6. 发送流式完成标记
        yield sse("done", {
            "type": "done",
            "sessionId": outcome.session.public_id,
            "runId": getattr(outcome, "run_id", ""),
            "messageId": message_id,
            "traceStatus": getattr(outcome, "trace_status", "completed"),
            "firstTokenSeconds": first_token_seconds,
            "elapsedSeconds": time.monotonic() - started,
        })

    def _run_in_worker(self, user, request, cancel_event=None, event_sink=None):
        """在独立线程/会话中执行智能体业务流程，避免跨线程共享 SQLAlchemy 会话。

        Args:
            user (UserAccount): 请求用户实体。
            request (ChatRequest): 对话请求对象。
            cancel_event (threading.Event | None, optional): 外部取消事件信号。
            event_sink (Callable | None, optional): 状态与中间增量推送接收器。

        Returns:
            AgentHarnessOutcome: 完整的多智能体业务执行结果。
        """
        if not isinstance(self.db, Session):
            self.agent_harness.cancel_event = cancel_event
            self.agent_harness.event_sink = event_sink
            return self.agent_harness.run(user, request)
        with Session(self.db.get_bind(), expire_on_commit=False) as db:
            worker_user = db.get(UserAccount, user.id)
            harness = SolaceAIAgentHarness(db, self.settings)
            harness.cancel_event = cancel_event
            harness.event_sink = event_sink
            return harness.run(worker_user, request)

    def _save_in_worker(self, user, outcome, content):
        """在独立线程会话中持久化流式回复消息与运行收据 (Save Assistant Message in Worker)

        Args:
            user (UserAccount): 接收回复的用户实体。
            outcome (AgentHarnessOutcome): 智能体执行结果包。
            content (str): 助手生成的最终文本。

        Returns:
            int | None: 成功落库的 ChatMessage 自增主键 ID。
        """
        if not isinstance(self.db, Session):
            return self.agent_harness.save_assistant_message(user, outcome.session, content)
        from app.models.entities import ChatMessage, ChatMessageRun, ChatSession
        with Session(self.db.get_bind(), expire_on_commit=False) as db:
            worker_user = db.get(UserAccount, user.id)
            session = db.get(ChatSession, outcome.session.id)
            SolaceAIAgentHarness(db, self.settings).save_assistant_message(
                worker_user, session, content, run_id=getattr(outcome, 'run_id', '')
            )
            run_id = getattr(outcome, 'run_id', '')
            if run_id:
                message = db.query(ChatMessage).join(
                    ChatMessageRun, ChatMessageRun.message_id == ChatMessage.id
                ).filter(
                    ChatMessageRun.user_id == user.id, ChatMessageRun.run_id == run_id,
                    ChatMessage.user_id == user.id, ChatMessage.session_id == session.id,
                    ChatMessage.role == 'ASSISTANT',
                ).first()
            else:
                message = db.query(ChatMessage).filter_by(
                    session_id=session.id, user_id=user.id, role='ASSISTANT'
                ).order_by(ChatMessage.id.desc()).first()
            return message.id if message else None

    def _save_metrics(self, user_id: int, run_id: str, first_token: float | None, elapsed: float) -> None:
        """在独立会话中持久化单轮问答交互的性能与耗时度量指标 (Save Performance Metrics)

        Args:
            user_id (int): 用户 ID。
            run_id (str): Agent 运行唯一标识 UUID。
            first_token (float | None): TTFT 首字延迟（秒）。
            elapsed (float): 整轮交互总端到端耗时（秒）。
        """
        from app.models.entities import RunMetrics
        with Session(self.db.get_bind()) as db:
            if db.get(RunMetrics, run_id) is None:
                db.add(RunMetrics(
                    run_id=run_id,
                    user_id=user_id,
                    first_token_seconds=first_token,
                    elapsed_seconds=elapsed,
                ))
                db.commit()

    def _response_client(self, outcome):
        """为没有最终答案的兼容运行结果按已有路由创建客户端。"""
        if self.settings.chat_backend == 'local':
            from app.services.chat_backends import LocalModelClient
            return LocalModelClient(self.settings.local_gateway_url, settings=self.settings)
        return client_for_route(self.settings, outcome.response_model_route)


def sse(event: str, data: dict) -> str:
    """格式化构造符合 W3C 标准的 Server-Sent Events (SSE) 协议报文 (Build SSE Frame)

    Args:
        event (str): 事件名称类型 (例如 'token', 'meta', 'citation', 'done', 'error')。
        data (dict): 事件携带的 JSON 负载字典。

    Returns:
        str: 带有 'event: ...\\ndata: ...\\n\\n' 换行终结符的标准 SSE 帧文本。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
