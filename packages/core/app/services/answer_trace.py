"""SolaceAI 学生端执行轨迹投影与安全审计分离服务 (Public Answer Trace & Stage Projection)

本模块专职负责在问答流式输出与历史回溯过程中，向学生安全展示 AI 思考与调用的执行步骤：
1. 深度隐私与指令防御隔离：
   - 严格杜绝将系统提示词、模型原始思考链 (Chain-of-Thought)、工具敏感出入参或未脱敏 PII 反射至前端；
   - 所有展示文案必须源自本模块静态白名单字典 (`STAGES`, `RESULTS`)，坚决不信任或直传模型生成的任意文本；
2. 标准化七阶段执行映射 (Phases: prepare, reason, act, observe, answer, review, deliver)；
3. 离线详细检索证据投影 (`project_retrieval_details`)：
   - 仅在离线纯本地模式 (`offline`) 下受限允许展示切片 ID、有限字符摘要及相关性分值，且对得分做有界与有限数值 (`isfinite`) 校验；
4. 运行时动态插桩装饰器 (`public_stage`)：
   - 零侵入装饰 LangGraph 状态图执行节点，自动感知工具调用、熔断降级、取消重试并记录耗时。
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import time
from typing import Any, Generator
import uuid
from app.models.answer_trace import PublicAnswerTrace

# 阶段文案白名单映射表：stage_key -> (title, why, default_result)
# 所有对外暴露的标题、意图说明和结果文案均硬编码定义于此，防范提示词泄露与模型输出投毒
STAGES = {
    'hospital_search': ('查询医院地点', '根据你提供的城市，查询高德地图中的医疗机构地点。', '医院地点查询已返回，接诊信息需向院方核实。'),
    'sanitize': ('输入隐私处理', '识别并处理输入中的个人信息，减少敏感信息进入后续处理。', '已完成输入隐私处理。'),
    'context': ('读取会话上下文', '在当前账号授权范围内整理历史对话和相关记忆。', '已整理当前会话的可用上下文。'),
    'risk': ('输入安全检查', '检查当前输入，选择适当的支持和安全处理方式。', '已完成安全检查并选择后续处理方式。'),
    'selection': ('选择支持方式', '结合当前问题选择适用的支持技能和授权上下文。', '已完成支持方式选择。'),
    'agent': ('Agent 决策摘要', '结合现有依据判断是否需要补充资料或生成回复。', '已完成本轮决策。'),
    'retrieval_policy': ('咨询检索策略', '咨询问题先查找参考资料，再进入模型决策。', '按咨询流程选择先调用 search_knowledge。'),
    'observation': ('观察工具结果', '整理本次工具执行结果，供后续 Agent 决策使用。', '已接收工具执行结果。'),
    'answer_ready': ('接收候选回答', '接收 Agent 已生成的回答，准备进行回复检查。', '候选回答已就绪，等待回复检查。'),
    'retrieval': ('检索参考资料', '寻找与当前问题有关的知识资料，为回复提供依据。', '已完成资料检索。'),
    'recall': ('多路召回与 Milvus RRF', '使用 BGE-M3 密集向量、学习型稀疏向量召回，再由 Milvus 执行 RRF 融合。', '已完成多路召回与 Milvus 融合。'),
    'fusion': ('跨查询 RRF 融合', '合并各查询的召回列表，去重并按倒数排名融合生成重排候选集。', '已生成重排候选集。'),
    'evidence_selection': ('最终证据选择', '按重排结果、父段去重和上下文预算选择最终参考资料。', '已选定回答使用的证据。'),
    'rerank': ('参考资料排序', '依据与问题的相关性重新排列候选资料。', '已完成候选资料排序。'),
    'memory_search': ('查找授权记忆', '从当前账号已授权保存的记忆中查找相关信息。', '已完成授权记忆查找。'),
    'memory_save': ('保存授权记忆', '按当前用户明确提出的要求保存记忆。', '已处理记忆保存请求。'),
    'memory_delete': ('删除授权记忆', '按当前用户明确提出的要求删除指定记忆。', '已处理记忆删除请求。'),
    'evidence_read': ('读取已检索资料', '进一步读取本轮已检索到的资料，核对可用依据。', '已读取本轮资料。'),
    'skill': ('加载支持技能', '加载本轮需要的受控支持技能。', '已加载支持技能。'),
    'tool': ('执行辅助操作', '按当前问题调用已授权的辅助能力。', '已完成辅助操作。'),
    'answer': ('生成回复', '依据当前问题、上下文和可用资料组织回复。', '已生成候选回复。'),
    'review': ('回复检查', '检查回复和引用；必要时修正不适当的内容。', '已完成回复和引用检查。'),
    'report': ('生成支持记录', '按本轮处理需要保存支持记录，便于后续跟进。', '已保存本轮支持记录。'),
    'case': ('建立跟进记录', '按本轮支持需要创建或复用后续跟进记录。', '跟进记录已处理。'),
    'email': ('发送跟进通知', '按本轮支持需要向已配置的接收方发送通知。', '通知发送操作已完成。'),
}
TOOL_STAGES = {
    'search_hospital': 'hospital_search',
    'search_knowledge': 'retrieval',
    'read_evidence': 'evidence_read',
    'load_skill': 'skill',
    'search_memory': 'memory_search',
    'save_memory': 'memory_save',
    'delete_memory': 'memory_delete',
}
ACT_STAGES = {*TOOL_STAGES.values(), 'tool', 'rerank', 'recall', 'fusion', 'evidence_selection'}
PHASES = {
    **{stage: 'prepare' for stage in STAGES},
    **{stage: 'act' for stage in ACT_STAGES},
    'agent': 'reason',
    'retrieval_policy': 'reason',
    'observation': 'observe',
    'answer': 'answer',
    'answer_ready': 'answer',
    'review': 'review',
    'report': 'deliver',
    'case': 'deliver',
    'email': 'deliver',
}
ACTORS = {
    'agent': 'MainAgent',
    'retrieval_policy': 'Runtime',
    'observation': 'MainAgent',
    'answer': 'MainAgent',
    'answer_ready': 'MainAgent',
    'review': 'SafetyAgent',
    'recall': 'RetrievalAgent',
    'fusion': 'RetrievalAgent',
    'evidence_selection': 'RetrievalAgent',
    'risk': 'SafetyAgent',
    'retrieval': 'RetrievalAgent',
    'rerank': 'RetrievalAgent',
    **{stage: 'MainAgent' for stage in TOOL_STAGES.values() if stage != 'retrieval'},
    'email': 'MailService',
    'case': 'CaseService',
    'report': 'ReportService',
}
STAGE_TOOLS = {stage: tool for tool, stage in TOOL_STAGES.items()}
STATUSES = {'running', 'completed', 'skipped', 'failed', 'degraded', 'cancelled'}
RESULTS = {
    'skipped': '本轮无需执行此步骤。',
    'failed': '此步骤未能完成。',
    'degraded': '部分能力暂不可用，已采用可用方式继续处理。',
    'cancelled': '此步骤已停止。',
    'reused': '已复用本轮已有结果，未重复执行。',
    'answered': '已完成本轮执行并生成候选回复。',
    'selected': '已选择下一项授权辅助操作。',
    'smtp_accepted': 'SMTP 服务器已接受预警邮件，请接收方检查收件箱及垃圾邮件。',
    'log_only': '通知已记入日志，未发送邮件。',
    'empty': '未找到适用的参考资料。',
    'observed': '工具结果已返回，将用于后续决策。',
}


def allowed_tool(value: Any) -> bool:
    """校验指定的工具名称是否属于受信任的受控工具白名单 (Check Allowed Tool)

    Args:
        value (Any): 工具标识名称。

    Returns:
        bool: 若在白名单中返回 True，否则返回 False。
    """
    return isinstance(value, str) and value in TOOL_STAGES


DETAIL_CHANNELS = {'dense', 'sparse', 'bm25', 'milvus_rrf', 'rrf', 'rerank', 'evidence'}
DETAIL_METHODS = {
    'cosine', 'inner_product', 'bm25', 'rrf', 'local_cross_encoder',
    'semantic_local_cross_encoder', 'semantic_api', 'lexical', 'selection',
}


def project_retrieval_details(value: Any) -> list[dict[str, Any]]:
    """安全清洗并投影多路召回与重排明细 (Sanitize and Project Retrieval Details)

    在防御纵深原则下，严格校验数值有限性（`math.isfinite`）、限制候选切片摘要长度（最长 600 字符），
    防范非法分值注入或超长恶意载荷导致前端渲染卡死。

    Args:
        value (Any): 原始检索通道与候选明细数据。

    Returns:
        list[dict[str, Any]]: 经过脱敏、截断与合法性审查后的检索详情数组。
    """
    if not isinstance(value, list):
        return []
    result = []
    for group in value[:4]:
        if not isinstance(group, dict) or group.get('channel') not in DETAIL_CHANNELS:
            continue
        rows = group.get('candidates')
        if not isinstance(rows, list):
            continue
        candidates = []
        for row in rows[:100]:
            if not isinstance(row, dict):
                continue
            rank, score, chunk = row.get('rank'), row.get('score'), row.get('chunkId')
            if type(rank) is not int or not 1 <= rank <= 100 or type(score) not in (float, int) or not math.isfinite(score):
                continue
            if chunk is not None and type(chunk) is not int and not isinstance(chunk, str):
                continue
            candidates.append(dict(
                rank=rank,
                chunkId=chunk if type(chunk) is int else str(chunk or '')[:120],
                source=str(row.get('source', ''))[:240],
                content=str(row.get('content', ''))[:600],
                score=score,
            ))
        total = group.get('total')
        result.append(dict(
            channel=group['channel'],
            queryIndex=max(1, min(4, group.get('queryIndex', 1))) if type(group.get('queryIndex', 1)) is int else 1,
            method=group.get('method') if group.get('method') in DETAIL_METHODS else 'selection',
            total=total if type(total) is int and 0 <= total <= 100000 else len(candidates),
            candidates=candidates,
        ))
    return result


def offline_details_enabled() -> bool:
    """检查当前系统运行模式是否允许输出深度检索过程指标 (Is Offline Details Enabled)

    Returns:
        bool: 仅在离线 Profile (`offline`) 下为 True。
    """
    from app.core.config import get_settings
    return get_settings().runtime_profile == 'offline'


def project_step(step: dict[str, Any], *, allow_details: bool = False) -> dict[str, Any] | None:
    """在读取边界对步骤数据重新执行白名单安全投影 (Project Step Under Allowlist Guard)

    执行逻辑：
    1. 校验阶段标识 `stage` 与状态 `status` 是否属于白名单合法枚举；
    2. 基于静态字典提取阶段标题、理由与执行结果描述；
    3. 组装七阶段标准字段（id, sequence, phase, actor, title, why, status, result 等）；
    4. 映射心理评估标签（意图识别、危机风险等级、情绪分类）；
    5. 若允许输出明细且属于检索阶段，附带投影清洗后的 `retrievalDetails`。

    Args:
        step (dict[str, Any]): 原始步骤字典。
        allow_details (bool, optional): 是否允许输出多路检索通道指标，默认为 False。

    Returns:
        dict[str, Any] | None: 脱敏规整后的步骤公开字典；若不合法返回 None。
    """
    stage = step.get('stage')
    if stage not in STAGES or step.get('status') not in STATUSES:
        return None
    title, why, result = STAGES[stage]
    status = step['status']
    result = RESULTS.get(status, result) if status in {'failed', 'cancelled', 'degraded'} else RESULTS.get(step.get('resultCode'), RESULTS.get(status, result))
    public = {key: step[key] for key in ('id', 'sequence', 'startedAt', 'durationMs') if key in step}
    public.update(
        stage=stage,
        phase=PHASES[stage],
        actor=ACTORS.get(stage, 'Runtime'),
        title=title,
        why=why,
        status=status,
        result='' if status == 'running' else result,
    )
    if status in {'completed', 'degraded'}:
        intent = {'CHAT': '日常交流', 'CONSULT': '心理咨询', 'RISK': '风险干预'}.get(step.get('intent'))
        risk = {'LOW': '低风险', 'MEDIUM': '中风险', 'HIGH': '高风险'}.get(step.get('risk'))
        emotion = {'NORMAL': '正常', 'ANXIETY': '焦虑', 'DEPRESSED': '低落', 'HIGH_RISK': '高风险'}.get(step.get('emotion'))
        if stage == 'selection' and intent:
            public['result'] = f'意图识别：{intent}。'
        if stage == 'risk' and risk:
            public['result'] = f'风险评估：{risk}。' + (f'情绪识别：{emotion}。' if emotion else '')
    if type(step.get('round')) is int and 1 <= step['round'] <= 10000:
        public['round'] = step['round']
    tool = step.get('tool', STAGE_TOOLS.get(stage))
    if allowed_tool(tool):
        public['tool'] = tool
    if allowed_tool(step.get('nextAction')) and status == 'completed':
        public['nextAction'] = step['nextAction']
        tool_title, tool_purpose, _ = STAGES[TOOL_STAGES[step['nextAction']]]
        public['result'] = f"下一步调用 {step['nextAction']}（{tool_title}），{tool_purpose}"
    if stage == 'observation' and allowed_tool(tool) and status == 'completed':
        public['result'] = f'已接收 {tool} 的执行结果，供后续决策使用。'
    if type(step.get('count')) is int and 0 <= step['count'] <= 100000:
        public['count'] = step['count']
    if allow_details and stage in {'recall', 'fusion', 'rerank', 'evidence_selection'}:
        details = project_retrieval_details(step.get('retrievalDetails'))
        if details:
            public['retrievalDetails'] = details
    return public


class TraceRecorder:
    """学生端问答执行轨迹记录器 (Student Trace Recorder)

    管理单次问答执行过程中各个阶段的生命周期、步骤序列累加、耗时统计、
    SSE 流式事件即时推送与数据库持久化。
    """

    def __init__(
        self,
        db: Any,
        user_id: int,
        session_id: int,
        public_id: str,
        run_id: str,
        sink: Any = None,
        cancel_event: Any = None,
        *,
        allow_details: bool | None = None,
    ):
        """初始化轨迹记录器。

        Args:
            db: 数据库会话。
            user_id (int): 所属用户主键 ID。
            session_id (int): 所属会话内部主键 ID。
            public_id (str): 会话对外暴露的 public_id。
            run_id (str): 本轮运行的全局唯一跟踪 UUID。
            sink (Any, optional): SSE 事件发射管道回调，默认为 None。
            cancel_event (Any, optional): 客户端主动断开或取消事件标志，默认为 None。
            allow_details (bool | None, optional): 是否允许输出多路召回指标，默认为 None（自动探测）。

        Raises:
            ValueError: 当数据库已有同 run_id 记录但归属不同用户/会话时抛出。
        """
        self.allow_details = offline_details_enabled() if allow_details is None else allow_details
        self.db, self.user_id, self.session_id = db, user_id, session_id
        self.public_id, self.run_id, self.sink, self.cancel_event = public_id, run_id, sink, cancel_event
        row = db.get(PublicAnswerTrace, run_id)
        if row and (row.user_id != user_id or row.session_id != session_id):
            raise ValueError('Run owner mismatch')
        self.row = row or PublicAnswerTrace(run_id=run_id, user_id=user_id, session_id=session_id, status='running')
        self.steps = json.loads(row.steps_json) if row else []
        self.round = max((step.get('round', 0) for step in self.steps if type(step.get('round', 0)) is int), default=0)
        self.previous_status = row.status if row else None
        self.row.status = 'running'
        # 重试已中断的任务时，将未完成的悬挂步骤标记为 cancelled
        for step in self.steps:
            if step['status'] == 'running':
                step['status'] = 'cancelled'
        self._save()

    def _save(self) -> None:
        """将当前内存中的步骤列表序列化并持久化写入数据库。"""
        self.row.steps_json = json.dumps(self.steps, ensure_ascii=False)
        self.db.add(self.row)
        self.db.commit()

    def _publish(self, step: dict[str, Any]) -> None:
        """持久化并将单个步骤作为 SSE 事件广播至前端客户端。"""
        self._save()
        if self.sink:
            self.sink({
                'type': 'step',
                'runId': self.run_id,
                'sessionId': self.public_id,
                'step': project_step(step, allow_details=self.allow_details),
            })

    @contextmanager
    def stage(self, name: str) -> Generator[dict[str, Any], None, None]:
        """轨迹步骤上下文管理器 (Trace Step Context Manager)

        进入时记录开始时间并向客户端广播 running 步骤，退出时计算执行毫秒耗时并广播完成/失败状态。

        Args:
            name (str): 阶段名称（必须在 STAGES 白名单内）。

        Yields:
            dict[str, Any]: 步骤元数据信息字典，业务代码可在其中写入 intent, risk, tool 等。

        Raises:
            ValueError: 传入未知的非白名单阶段名。
        """
        if name not in STAGES:
            raise ValueError('Unknown public stage')
        if PHASES[name] == 'reason':
            self.round += 1
        step = {
            'id': uuid.uuid4().hex,
            'sequence': len(self.steps) + 1,
            'stage': name,
            'status': 'running',
            'startedAt': datetime.now(timezone.utc).isoformat(),
            'durationMs': None,
        }
        if self.round:
            step['round'] = self.round
        self.steps.append(step)
        self._publish(step)
        started = time.monotonic()
        info = {}
        try:
            yield info
        except BaseException:
            info['status'] = 'cancelled' if self.cancel_event and self.cancel_event.is_set() else 'failed'
            raise
        finally:
            status = info.get('status', 'completed')
            step['status'] = status if status in STATUSES - {'running'} else 'completed'
            step['durationMs'] = round((time.monotonic() - started) * 1000, 2)
            if type(info.get('count')) is int and 0 <= info['count'] <= 100000:
                step['count'] = info['count']
            for field in ('intent', 'risk', 'emotion'):
                if isinstance(info.get(field), str):
                    step[field] = info[field]
            for field in ('tool', 'nextAction'):
                if allowed_tool(info.get(field)):
                    step[field] = info[field]
            if info.get('resultCode') in RESULTS:
                step['resultCode'] = info['resultCode']
            if self.allow_details:
                details = project_retrieval_details(info.get('retrievalDetails'))
                if details:
                    step['retrievalDetails'] = details
            self._publish(step)

    def finish(self, status: str = 'completed') -> None:
        """标记本轮整个问答轨迹执行生命周期结束 (Finalize Trace Lifecycle)

        Args:
            status (str, optional): 最终终态，默认为 'completed'。若中间包含 degraded 步骤则自动升级为 degraded。
        """
        if status == 'completed' and any(step['status'] in {'failed', 'degraded'} for step in self.steps):
            status = 'degraded'
        self.row.status = status if status in STATUSES - {'running'} else 'failed'
        for step in self.steps:
            if step['status'] == 'running':
                step['status'] = self.row.status
        self._save()


def load_answer_trace(
    db: Any,
    user_id: int,
    session_id: int,
    run_id: str,
    message_id: int | None = None,
) -> dict[str, Any] | None:
    """加载并投影特定问答的历史执行轨迹详情 (Load Answer Trace by Run ID)

    执行严格的归属校验（user_id 与 session_id 必须一致），防止越权嗅探。

    Args:
        db: 数据库会话。
        user_id (int): 当前用户 ID。
        session_id (int): 会话 ID。
        run_id (str): 目标运行 UUID。
        message_id (int | None, optional): 关联消息 ID。

    Returns:
        dict[str, Any] | None: 包含 runId, status, steps 的安全数据字典；不存在或越权时返回 None。
    """
    row = db.get(PublicAnswerTrace, run_id)
    if row is None or row.user_id != user_id or row.session_id != session_id:
        return None
    steps = [
        public for item in json.loads(row.steps_json)
        if (public := project_step(item, allow_details=offline_details_enabled()))
    ]
    return {'runId': run_id, 'messageId': message_id, 'status': row.status, 'steps': steps}


def public_stage(name: str):
    """LangGraph 运行时节点执行动态插桩装饰器工厂 (Instrument Runtime Graph Node)

    在不将图内部私有状态泄露给客户端的前提下，自动包装节点执行过程：
    - 自动开启与关闭当前步骤生命周期；
    - 捕获取消、降级、熔断与工具调用动作；
    - 记录工具执行观测与候选答案接收事件。

    Args:
        name (str): 节点对应的阶段名称。

    Returns:
        Callable: 包装后的节点函数装饰器。
    """
    from functools import wraps

    def decorate(method):
        @wraps(method)
        def wrapped(self, state, *args, **kwargs):
            recorder = getattr(self, 'public_trace', None)
            if recorder is None:
                return method(self, state, *args, **kwargs)
            stage = TOOL_STAGES.get(state.get('tool'), 'tool') if name == 'tool' else name
            info = {}
            starting_step = len(getattr(recorder, 'steps', []))
            try:
                with recorder.stage(stage) as info:
                    result = method(self, state, *args, **kwargs)
                    new_trace = result.get('trace', [])[len(state.get('trace', [])):]
                    actions = {item.get('action') for item in new_trace}
                    if any(str(action).startswith('cancelled') for action in actions):
                        info['status'] = 'cancelled'
                    elif actions & {'degraded', 'context_limit', 'privilege_demoted_to_draft', 'escalate_to_human'}:
                        info['status'] = 'degraded'
                    elif 'tool_result_reused' in actions:
                        info.update(status='skipped', resultCode='reused')
                    if stage == 'agent' and info.get('status', 'completed') == 'completed':
                        if result.get('answer'):
                            info['resultCode'] = 'answered'
                        elif allowed_tool(result.get('tool')):
                            info['nextAction'] = result['tool']
                        if 'evidence' in result or 'evidence' in state:
                            info['count'] = len(result.get('evidence', state.get('evidence', [])))
                    if name == 'tool':
                        if allowed_tool(state.get('tool')):
                            info['tool'] = state['tool']
                        observations = result.get('observations', [])
                        observation = observations[-1] if observations else {}
                        if observation.get('error') or str(observation.get('status', '')).startswith('circuit_open'):
                            info['status'] = 'degraded'
                        if stage == 'retrieval':
                            info['count'] = len(result.get('evidence', []))
                            if not info['count'] and 'status' not in info:
                                info['resultCode'] = 'empty'
                # 候选回答直接由决策节点返回时，安全记录其就绪事件
                if stage == 'agent' and result.get('answer') and info.get('status', 'completed') == 'completed':
                    generated = any(step['stage'] == 'answer' for step in getattr(recorder, 'steps', [])[starting_step:])
                    if not generated:
                        with recorder.stage('answer_ready'):
                            pass
                return result
            finally:
                if name == 'tool':
                    # 工具返回或异常退出边界，记录受限元数据
                    with recorder.stage('observation') as observation_info:
                        observation_info['status'] = info.get('status', 'completed')
                        observation_info['tool'] = state.get('tool')
                        if 'count' in info:
                            observation_info['count'] = info['count']
                        observation_info['resultCode'] = info.get('resultCode', 'observed')
        return wrapped
    return decorate
