"""SolaceAI 数据传输对象与模式定义模块 (Data Transfer Objects & Schemas)

本模块严格遵循 Pydantic V2 契约模式，定义全系统前后端交互与内部通信所使用的 DTO：
1. 客户端交互与流式输出契约 (ChatRequest, ChatStreamEvent)；
2. 知识库摄入、切片与多版本控制传输契约 (KnowledgeIngestRequest, KnowledgeIngestResponse)；
3. 心理评估报告与咨询历史聚合契约 (ReportResponse, ConversationResponse, ConversationMessageResponse)；
4. 危机个案跟踪、人工流转与审核日志契约 (RiskCaseResponse, CaseNoteResponse, ToolRecordResponse)；
5. 后台异步作业与死信排查数据契约 (ToolJobResponse, DeadLetterResponse)；
6. 智能体端到端追踪全景与治理策略审计 (AgentRunTraceResponse, ToolAuditResponse)；
7. 满意度评价与 LLM 消息通用格式 (AnswerFeedbackRequest, AiMessage)。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """客户端对话交互请求数据传输对象 (Chat Request DTO)

    用于学生或管理员在前端聊天界面发起一轮新的对话交互。
    """
    modelBackend: Literal['cloud', 'local'] | None = None
    message: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="用户本轮输入的文本消息（不能为空，上限 8192 字符）",
    )
    sessionId: Optional[str] = Field(
        default=None,
        description="已有会话的 public_id（若为空则由服务端自动创建新会话）",
    )
    runId: Optional[str] = Field(
        default=None,
        pattern=r"^[a-fA-F0-9]{32}$",
        description="重试同一轮对话时复用的 32 位 UUID hex 幂等标识",
    )


class ChatStreamEvent(BaseModel):
    """服务端事件流响应事件数据传输对象 (SSE Chat Stream Event DTO)

    用于遵循 Server-Sent Events (SSE) 协议向前端推送流式 Token、会话元数据或错误通知。
    """
    sessionId: Optional[str] = Field(
        default=None,
        description="当前绑定的会话 public_id",
    )
    content: Optional[str] = Field(
        default=None,
        description="当前增量生成的 Token 文本块",
    )
    message: Optional[str] = Field(
        default=None,
        description="系统状态变更通知或用户提示信息",
    )
    type: str = Field(
        ...,
        description="事件类型：'meta'(元数据) | 'token'(增量字词) | 'done'(完成) | 'error'(异常中断)",
    )


class KnowledgeIngestRequest(BaseModel):
    """知识库文档批量摄入与向量化请求数据传输对象 (Knowledge Ingest Request DTO)

    支持向知识库空间导入 Markdown 文本或纯文本，并触发多版本拆分与向量化。
    """
    document_id: Optional[str] = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
        description="稳定文档身份标识；同名文件需指定不同 ID 以支持版本独立流转",
    )
    source: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="知识文档来源文件名或标识 (例如 'student-support-01.md')",
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=500000,
        description="待导入的原始 Markdown 或纯文本内容（最长不超过 500,000 字符）",
    )


class KnowledgeIngestResponse(BaseModel):
    """知识库文档摄入完成响应数据传输对象 (Knowledge Ingest Response DTO)

    向调用方返回入库成功的文档身份与生成的物理切片数量。
    """
    document_id: Optional[str] = Field(
        default=None,
        description="成功持久化或更新的稳定文档全局唯一标识",
    )
    source: str = Field(
        ...,
        description="来源文件名或标识",
    )
    chunks: int = Field(
        ...,
        description="成功分块并写入关系表与向量索引的切片总数量",
    )


class ReportResponse(BaseModel):
    """学生心理健康评估报告详情响应数据传输对象 (Report Response DTO)

    用于辅导员控制台查看由大模型与多智能体系统研判生成的结构化评估结果。
    """
    id: int = Field(..., description="报告自增主键 ID")
    sessionId: str = Field(..., description="触发该报告的会话 public_id")
    username: str = Field(..., description="学生登录账号/学号")
    displayName: str = Field(..., description="学生真实姓名或系统显示名称")
    content: str = Field(..., description="触发本轮评估的学生原始输入文本")
    intent: str = Field(..., description="用户意图识别分类：'CHAT' | 'CONSULT' | 'RISK'")
    emotion: str = Field(..., description="情绪标签：'NORMAL' | 'ANXIETY' | 'DEPRESSED' | 'HIGH_RISK'")
    emotionScore: float = Field(..., description="情绪/风险量化得分，浮点区间 0.0 ~ 4.0")
    riskLevel: str = Field(..., description="研判风险等级：'LOW' | 'MEDIUM' | 'HIGH'")
    summary: str = Field(..., description="心理状态研判专业归因与关键特征摘要")
    createdAt: datetime = Field(..., description="评估报告生成时间戳")


class ConversationMessageResponse(BaseModel):
    """对话历史单条消息详情响应数据传输对象 (Conversation Message Response DTO)

    展示单轮对话中的发送者、文本内容以及发生时间。
    """
    role: str = Field(..., description="消息发送主体角色：'user' | 'assistant' | 'system'")
    content: str = Field(..., description="消息正文内容")
    createdAt: datetime = Field(..., description="消息创建时间戳")


class ConversationResponse(BaseModel):
    """会话多轮历史详情聚合响应数据传输对象 (Conversation Response DTO)

    包含会话基础元信息及全量按时间序排列的消息流。
    """
    sessionId: str = Field(..., description="会话唯一公共标识 public_id")
    title: str = Field(..., description="会话显示标题（通常由首轮输入或大模型自动提炼）")
    messages: list[ConversationMessageResponse] = Field(
        default_factory=list,
        description="按时序正序排列的完整对话历史消息列表",
    )


class ToolRecordResponse(BaseModel):
    """工具调用执行与审计记录响应数据传输对象 (Tool Record Response DTO)

    记录辅导员台账导出、紧急邮件通知等工具的执行状态与日志。
    """
    id: int = Field(..., description="工具执行审计记录主键 ID")
    reportId: int = Field(..., description="关联的心理健康评估报告 ID")
    status: str = Field(..., description="执行终态：'SUCCESS' | 'FAILED'")
    message: str = Field(..., description="执行详细说明或异常栈摘要")
    createdAt: datetime = Field(..., description="记录创建时间戳")
    channel: Optional[str] = Field(default=None, description="预警告警渠道（若为预警任务，如 'EMAIL' / 'LOG'）")
    recipient: Optional[str] = Field(default=None, description="告警接收人邮箱或目标列表")
    filePath: Optional[str] = Field(default=None, description="生成的物理台账文件路径（若为 Excel 导出）")


class RiskCaseResponse(BaseModel):
    """危机干预与个案流转跟踪响应数据传输对象 (Risk Case Response DTO)

    展示针对中高风险学生建立的线上线下闭环处置个案详情。
    """
    id: int = Field(..., description="个案自增主键 ID")
    reportId: int = Field(..., description="触发建立个案的心理评估报告 ID")
    riskLevel: str = Field(..., description="个案危机等级：'MEDIUM' | 'HIGH'")
    status: str = Field(..., description="个案流转状态：'OPEN' | 'ALERT_SENT' | 'ACKNOWLEDGED' | 'CLOSED'")
    owner: str = Field(..., description="当前责任辅导员或心理咨询师")
    summary: str = Field(..., description="危机个案核心诱因与风险表现摘要")
    handoffSummary: str = Field(..., description="标准化交接专业建议与线下干预行动指南")
    acknowledgedBy: Optional[str] = Field(default=None, description="线下确认接手辅导员姓名")
    acknowledgedAt: Optional[datetime] = Field(default=None, description="确认接手时间戳")
    createdAt: datetime = Field(..., description="个案创建时间戳")
    updatedAt: datetime = Field(..., description="个案最新状态更新时间戳")


class CaseNoteResponse(BaseModel):
    """个案流转与人工跟进备注响应数据传输对象 (Case Note Response DTO)

    辅导员对个案线下访谈、家长沟通或医疗转介情况的追加记录。
    """
    id: int = Field(..., description="跟进备注主键 ID")
    caseId: int = Field(..., description="所属个案主键 ID")
    actor: str = Field(..., description="添加该条备注的辅导员/咨询师姓名")
    note: str = Field(..., description="线下访谈详情、学生心理动向及干预措施文本")
    createdAt: datetime = Field(..., description="备注写入时间戳")


class ToolJobResponse(BaseModel):
    """异步工具任务作业状态响应数据传输对象 (Tool Job Response DTO)

    用于监控异步作业调度器的任务排队、执行与重试状态。
    """
    id: int = Field(..., description="异步任务作业自增 ID")
    reportId: int = Field(..., description="关联的心理健康评估报告 ID")
    kind: str = Field(..., description="作业类型：'EXCEL_REPORT' | 'CASE_CREATE' | 'ALERT_SEND'")
    status: str = Field(..., description="作业状态：'PENDING' | 'RUNNING' | 'SUCCESS' | 'DEAD'")
    attempts: int = Field(..., description="当前已尝试执行与重试的次数")
    maxAttempts: int = Field(..., description="允许的最大重试次数限制")
    dependsOnJobId: Optional[int] = Field(default=None, description="前置依赖作业 ID（需等待其成功后方可执行）")
    runAfter: datetime = Field(..., description="延迟调度时间戳（最早允许执行时间）")
    lastError: str = Field(..., description="最近一次执行失败的异常信息")
    createdAt: datetime = Field(..., description="作业创建入队时间戳")
    updatedAt: datetime = Field(..., description="作业状态最新更新时间戳")


class DeadLetterResponse(BaseModel):
    """死信队列记录响应数据传输对象 (Dead Letter Response DTO)

    针对超过最大重试次数仍然失败的任务进行持久化隔离，便于技术运维追溯定位。
    """
    id: int = Field(..., description="死信记录主键 ID")
    jobId: Optional[int] = Field(default=None, description="发生致命异常的原失败作业 ID")
    reportId: int = Field(..., description="关联的心理健康报告 ID")
    kind: str = Field(..., description="失败作业类型")
    reason: str = Field(..., description="最终致命失败归因与堆栈详情")
    payload: str = Field(..., description="作业触发时的原始参数快照 JSON")
    createdAt: datetime = Field(..., description="转入死信隔离表的时间戳")


class AgentRunTraceResponse(BaseModel):
    """多智能体协作链路全景追踪响应数据传输对象 (Agent Run Trace Response DTO)

    记录单次对话中意图识别、隐私脱敏、记忆提取、RAG检索及决策图的全程审计轨迹。
    """
    id: int = Field(..., description="链路追踪唯一主键 ID")
    sessionId: str = Field(..., description="关联会话 public_id")
    reportId: Optional[int] = Field(default=None, description="本轮产生的心理评估报告 ID（若有）")
    username: str = Field(..., description="发起对话的用户学号/账号")
    intent: str = Field(..., description="前端识别的用户意图分类")
    riskLevel: str = Field(..., description="研判确定的风险等级")
    originalInput: str = Field(..., description="脱敏前的用户原始提问文本")
    sanitizedInput: str = Field(..., description="执行敏感实体匿名化脱敏后的干净文本")
    memoryBrief: str = Field(..., description="检索注入的长期用户记忆摘要简报")
    agentSteps: list[dict[str, Any]] = Field(
        default_factory=list,
        description="智能体协作决策流与工具执行全流程步骤列表",
    )
    retrievedKnowledge: list[dict[str, Any]] = Field(
        default_factory=list,
        description="RAG 检索召回并精排后的知识切片清单",
    )
    responseMessages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="最终组装并投递至 LLM 推理的完整 Prompt 消息序列",
    )
    assessment: dict[str, Any] = Field(
        default_factory=dict,
        description="心理量化评估模型输出的原始打分字典",
    )
    createdAt: datetime = Field(..., description="追踪轨迹生成与持久化时间戳")


class ToolAuditResponse(BaseModel):
    """工具调用安全治理策略审计响应数据传输对象 (Tool Audit Response DTO)

    记录系统安全策略引擎对各工具调用请求的拦截审查决策。
    """
    id: int = Field(..., description="审计记录自增 ID")
    jobId: Optional[int] = Field(default=None, description="关联的异步作业 ID（若有）")
    reportId: Optional[int] = Field(default=None, description="关联的心理报告 ID（若有）")
    toolName: str = Field(..., description="被调用的业务工具函数名称")
    policy: str = Field(..., description="命中的安全治理规则/策略标识")
    allowed: bool = Field(..., description="策略是否放行该工具执行 (True 为放行，False 为阻断)")
    status: str = Field(..., description="审计终态判定")
    reason: str = Field(..., description="放行或拦截的具体决策理由说明")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="工具执行入参快照字典",
    )
    createdAt: datetime = Field(..., description="审计记录创建时间戳")
    updatedAt: datetime = Field(..., description="审计记录更新时间戳")


class AiMessage(BaseModel):
    """统一大模型上下文消息数据传输对象 (AiMessage DTO)

    用于在不同大模型后端（OpenAI / Ollama / 本地微调模型）之间统一消息格式。
    """
    role: str = Field(..., description="消息角色：'system' | 'user' | 'assistant'")
    content: str = Field(..., description="消息正文内容")


def authority(role: str) -> dict[str, Any]:
    """构造与 Spring Security 及前端 RBAC 鉴权规范兼容的角色权限字典

    Args:
        role (str): 角色字符串标识（如 'ROLE_ADMIN', 'ROLE_USER'）。

    Returns:
        dict[str, Any]: 包含 'authority' 键的权限映射字典。
    """
    return {"authority": role}


class AnswerFeedbackRequest(BaseModel):
    """用户问答质量评价与改进建议请求数据传输对象 (Answer Feedback Request DTO)

    用于终端学生或辅导员在对话气泡下方对 AI 回答质量打标（点赞/点踩与分类原因）。
    """
    messageId: int = Field(..., gt=0, description="被评价的助手回复消息主键 ID（必须大于 0）")
    runId: str = Field(..., min_length=1, max_length=64, description="关联的单轮 Agent 运行唯一标识 UUID")
    helpful: bool = Field(..., description="用户反馈布尔值：True 表示回答有帮助(点赞)，False 表示回答无帮助(点踩)")
    category: str = Field(
        default='general',
        pattern=r'^(general|irrelevant|evidence|unsafe|other)$',
        description=(
            "反馈问题分类标签：'general'(通用评价) | 'irrelevant'(答非所问) | "
            "'evidence'(知识引用不准) | 'unsafe'(不符合心理安全伦理) | 'other'(其他)"
        ),
    )
    note: str = Field(
        default='',
        max_length=1000,
        description="用户填写的详细建议或文本说明（最长 1000 字符）",
    )
