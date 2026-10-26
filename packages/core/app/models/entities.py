"""SolaceAI ORM 数据实体模型定义模块 (Domain Entities & Persistence Models)

该模块定义了 SolaceAI 系统中所有核心业务领域对象的 SQLAlchemy 2.0 ORM 映射模型，涵盖：
- 用户身份与权限体系 (UserAccount, SessionVisibility)
- 对话上下文与消息历史 (ChatSession, ChatMessage, ChatMessageRun, AgentTurnReceipt)
- 心理测评与危机干预流程 (PsychologicalReport, RiskCase, CaseNote, AlertRecord)
- 知识库多版本生命周期与切片 (KnowledgeChunk, KnowledgeChunkMetadata, KnowledgeDocumentVersion)
- 异步作业队列与死信治理 (ToolJob, DeadLetterRecord, ExcelRecord)
- 智能体端到端追踪、工具幂等性执行与指标统计 (AgentRunTrace, ToolAuditRecord, ToolExecution, AnswerFeedback, RunMetrics, AgentStoreItem)

设计原则：
1. 强类型映射：严格遵循 SQLAlchemy 2.0 `Mapped[...]` 和 `mapped_column` 现代类型标注。
2. 时间一致性：统一通过 `now()` 生成标准无时区 UTC naive datetime，规避 SQLite 与各数据库方言时间漂移。
3. 审计与幂等：针对 Agent 多轮执行与工具调用，提供强一致性的幂等键与执行收据追踪机制。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def now() -> datetime:
    """获取全局标准 UTC 当前时间戳。

    Returns:
        datetime: 无时区信息的 UTC naive datetime 对象，兼容 SQLite、PostgreSQL
            与 SQLAlchemy 日期比较，避免 Python 3.12+ `datetime.utcnow()` 废弃警告。
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


class UserAccount(Base):
    """用户账号与权限持久化模型。

    Attributes:
        id: 数据库自增主键。
        username: 全局唯一的登录用户名或学号工号。
        display_name: 用户真实姓名或友好昵称。
        password_hash: 基于 Argon2id 算法计算的安全散列密文。
        roles_csv: 逗号分隔的角色权限字符串（例如：ROLE_USER,ROLE_COUNSELOR,ROLE_ADMIN）。
        created_at: 账号注册创建时间戳。
        sessions: 用户发起的咨询对话会话列表。
    """

    __tablename__ = "user_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, doc="自增主键")
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, doc="唯一登录用户名/学号")
    email: Mapped[Optional[str]] = mapped_column(String(254), nullable=True, unique=True, index=True)
    requires_setup: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    display_name: Mapped[str] = mapped_column(String(128), doc="用户真实姓名或昵称")
    password_hash: Mapped[str] = mapped_column(String(128), doc="密码安全散列值 (Argon2id)")
    roles_csv: Mapped[str] = mapped_column(String(256), default="ROLE_USER", doc="逗号分隔的角色列表 (如 ROLE_USER,ROLE_ADMIN)")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="账号注册创建时间")

    # 关联用户的历史咨询对话会话
    sessions: Mapped[list["ChatSession"]] = relationship(back_populates="user")

    @property
    def roles(self) -> list[str]:
        """获取用户所拥有的角色权限标识列表。"""
        return [role for role in self.roles_csv.split(",") if role]

    @roles.setter
    def roles(self, value: list[str] | set[str]) -> None:
        """格式化并设置用户的角色权限列表。"""
        self.roles_csv = ",".join(sorted(value))


class ChatSession(Base):
    """咨询对话会话持久化模型。

    Attributes:
        id: 数据库自增主键。
        public_id: 对外公开的安全 UUID 格式会话标识符。
        title: 会话主题标题（通常由第一轮对话自动提取或摘要生成）。
        user_id: 所属用户账号 ID。
        created_at: 会话初始创建时间戳。
        updated_at: 会话最近活跃或产生新消息的时间戳。
        user: 所属的用户实体对象。
        messages: 会话下包含的时序对话消息列表（级联删除）。
    """

    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, doc="自增主键")
    public_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, doc="对外公开的 UUID 会话标识")
    title: Mapped[str] = mapped_column(String(160), doc="会话标题（通常由第一轮对话自动提取）")
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"), doc="所属用户主键 ID")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="会话发起时间")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="最近消息更新时间")

    user: Mapped[UserAccount] = relationship(back_populates="sessions")
    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session", cascade="all, delete-orphan")

    def touch(self) -> None:
        """更新当前会话的最近活跃时间戳为当前 UTC 时间。"""
        self.updated_at = now()


class ChatMessage(Base):
    """对话消息明细持久化模型。

    Attributes:
        id: 数据库自增主键。
        user_id: 消息所属的用户 ID。
        session_id: 关联的会话 ID。
        role: 消息发送方角色 (user / assistant / system)。
        content: 消息纯文本内容。
        created_at: 消息产生入库时间戳。
        session: 关联的会话实体对象。
    """

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, doc="自增主键")
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"), doc="所属用户 ID")
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), doc="所属会话 ID")
    role: Mapped[str] = mapped_column(String(32), doc="消息发送方角色 (USER / ASSISTANT / SYSTEM)")
    content: Mapped[str] = mapped_column(Text, doc="消息文本内容")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="消息产生时间")

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class KnowledgeChunk(Base):
    """知识库切片实体模型。

    Attributes:
        id: 数据库自增主键。
        document_id: 归属文档唯一标识。
        knowledge_base_id: 知识库标识空间（支持多知识库隔离）。
        tenant_id: 租户标识空间。
        version_id: 关联的文档版本记录 ID。
        source: 来源文件名称或外部标识符。
        source_index: 该切片在源文档中的顺序索引（从 0 开始）。
        content: 切片文本正文内容。
        embedding_json: 序列化为 JSON 字符串的浮点向量。
        created_at: 切片生成时间戳。
    """

    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, doc="自增主键")
    document_id: Mapped[str] = mapped_column(String(64), default='', index=True, doc="所属文档的全局唯一标识 SHA-256")
    knowledge_base_id: Mapped[str] = mapped_column(String(64), default='default', index=True, doc="知识库空间隔离标识")
    tenant_id: Mapped[str] = mapped_column(String(64), default='default', index=True, doc="多租户隔离标识")
    version_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True, doc="关联的文档版本主键 ID")
    source: Mapped[str] = mapped_column(String(256), index=True, doc="来源文件名或来源标识")
    source_index: Mapped[int] = mapped_column(Integer, doc="在来源文档中的切片序号")
    content: Mapped[str] = mapped_column(Text, doc="切片文本内容")
    embedding_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True, doc="预计算的 Embedding 浮点向量序列化 JSON")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="切片创建时间")


class SessionVisibility(Base):
    """用户端会话可见性控制模型。

    Attributes:
        user_id: 目标用户 ID。
        session_id: 关联会话 ID。
        hidden: 是否对用户列表视图隐藏（True 表示软隐藏或归档）。
    """

    __tablename__ = "session_visibility"
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"), primary_key=True, doc="所属用户 ID")
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), primary_key=True, doc="关联会话 ID")
    hidden: Mapped[bool] = mapped_column(Boolean, default=True, doc="是否对用户隐藏")


class AgentTurnReceipt(Base):
    """单轮对话智能体执行业务收据模型。

    用于保障对话状态机原子性，防止网络抖动导致的重复重放与业务副作用多次触发。

    Attributes:
        run_id: Agent 单轮运行的全局唯一 UUID 标识。
        user_id: 请求用户 ID。
        session_id: 关联的会话 ID。
        input_hash: 用户原始输入的 SHA-256 校验哈希。
        user_message_saved: 用户侧输入消息是否已成功写入持久层。
        assistant_saved: 助手侧回复消息是否已成功写入持久层。
        report_id: 关联生成的心理测评报告 ID。
        trace_id: 关联的多智能体执行链路追踪记录 ID。
        intent: 本轮对话识别意图（如 CHAT / CONSULT / CRISIS）。
        risk: 本轮对话安全研判风险等级。
        final_response: 安全审核通过的最终下发回复文本。
        citations_json: 回复中引用的真实知识切片 JSON 数组。
    """

    __tablename__ = "agent_turn_receipts"
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True, doc="单轮业务运行唯一标识 UUID")
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"), index=True, doc="所属用户 ID")
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), index=True, doc="关联会话 ID")
    input_hash: Mapped[str] = mapped_column(String(64), doc="用户输入文本 SHA256 哈希")
    user_message_saved: Mapped[bool] = mapped_column(Boolean, default=False, doc="用户消息是否已保存")
    assistant_saved: Mapped[bool] = mapped_column(Boolean, default=False, doc="助手回复是否已保存")
    report_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, doc="关联生成的心理评估报告 ID")
    trace_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, doc="关联的执行链路追踪 ID")
    intent: Mapped[str] = mapped_column(String(32), default="CHAT", doc="研判的对话意图")
    risk: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, doc="研判的心理风险等级")
    final_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True, doc="审核通过的最终回复内容")
    citations_json: Mapped[str] = mapped_column(Text, default="[]", doc="回复中引用的真实知识库切片 JSON")


class KnowledgeChunkMetadata(Base):
    """结构化知识切片扩展层级元数据模型。

    Attributes:
        chunk_id: 关联的基础知识切片主键 ID。
        version_id: 关联的文档版本 ID。
        document_id: 来源文档 SHA-256 唯一指纹。
        document_version: 文档内容版本哈希。
        section_path: Markdown 标题层级路径 JSON 数组。
        parent_id: 父级段落 SHA-256 哈希。
        parent_content: 父级段落完整原文（用于 Parent-Document 检索召回）。
        page_no: PDF 等分页文档的页码编号。
        contextual_prefix: LLM 提取的上下文导读摘要前缀。
        chunk_uid: 确定性切片业务全局唯一标识。
    """

    __tablename__ = "knowledge_chunk_metadata"
    chunk_id: Mapped[int] = mapped_column(ForeignKey("knowledge_chunks.id", ondelete="CASCADE"), primary_key=True, doc="关联的基础知识切片 ID")
    version_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True, doc="关联的文档版本主键 ID")
    document_id: Mapped[str] = mapped_column(String(64), doc="文档唯一 SHA256 哈希")
    document_version: Mapped[str] = mapped_column(String(64), doc="文档内容版本哈希")
    section_path: Mapped[str] = mapped_column(Text, default="[]", doc="Markdown 标题路径 JSON 数组")
    parent_id: Mapped[str] = mapped_column(String(64), doc="父级段落 SHA256 哈希")
    parent_content: Mapped[str] = mapped_column(Text, doc="完整的父级段落原文")
    page_no: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, doc="分页文档的页码编号 (PDF)")
    contextual_prefix: Mapped[str] = mapped_column(Text, default="", doc="生成的上下文摘要前缀")
    chunk_uid: Mapped[str] = mapped_column(String(64), default="", index=True, doc="确定性切片业务标识")


class PsychologicalReport(Base):
    """心理风险评估与分析报告模型。

    Attributes:
        id: 数据库自增主键。
        user_id: 接受评估的学生/用户 ID。
        session_id: 触发评估的会话 ID。
        content: 触发评估的学生本轮原始输入内容。
        intent: 识别的对话意图 (CHAT / CONSULT / CRISIS)。
        emotion: 识别的情绪类型标签 (NORMAL / ANXIETY / DEPRESSED / HIGH_RISK)。
        emotion_score: 情绪强度或风险量化得分 (0.0 ~ 4.0)。
        risk_level: 综合安全风险评级 (LOW / MEDIUM / HIGH)。
        _legacy_confidence: 历史兼容置信度字段（已废弃，保留仅为向后数据库兼容）。
        summary: 风险研判理由与心理状态诊断摘要。
        created_at: 报告生成时间戳。
    """

    __tablename__ = "psychological_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, doc="自增主键")
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"), doc="评估对象学生 ID")
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), doc="关联会话 ID")
    content: Mapped[str] = mapped_column(Text, doc="触发评估的学生本轮原始输入")
    intent: Mapped[str] = mapped_column(String(32), doc="意图分类 (CHAT / CONSULT / RISK)")
    emotion: Mapped[str] = mapped_column(String(32), doc="情绪标签 (NORMAL / ANXIETY / DEPRESSED / HIGH_RISK)")
    emotion_score: Mapped[float] = mapped_column(Float, doc="情绪强度/风险量化得分 (0.0~4.0)")
    risk_level: Mapped[str] = mapped_column(String(32), doc="综合风险评级 (LOW / MEDIUM / HIGH)")
    # 兼容历史版本的 NOT NULL 列：仅用于兼容已有数据库表的物理约束并保留插入兼容性
    _legacy_confidence: Mapped[float] = mapped_column("confidence", Float, default=0.0, doc="停用的历史兼容列，不参与业务")
    summary: Mapped[str] = mapped_column(Text, doc="评估摘要与研判理由")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="报告生成时间")


class RiskCase(Base):
    """危机干预个案流转管理模型。

    Attributes:
        id: 数据库自增主键。
        report_id: 关联触发的心理报告 ID（唯一约束）。
        risk_level: 个案风险评级 (MEDIUM / HIGH)。
        status: 个案流转状态 (OPEN / ALERT_SENT / ACKNOWLEDGED)。
        owner: 当前跟进责任辅导员或心理咨询师。
        summary: 个案核心问题简述。
        handoff_summary: 专业危机交接摘要（含建议处置方案与脱敏对话摘要）。
        acknowledged_by: 确认接管处理的教师/咨询师。
        acknowledged_at: 确认接管的时间戳。
        created_at: 个案建立时间戳。
        updated_at: 个案最近更新时间戳。
    """

    __tablename__ = "risk_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, doc="自增主键")
    report_id: Mapped[int] = mapped_column(Integer, unique=True, index=True, doc="关联的心理报告 ID")
    risk_level: Mapped[str] = mapped_column(String(32), index=True, doc="个案初始风险等级 (MEDIUM / HIGH)")
    status: Mapped[str] = mapped_column(String(32), index=True, doc="个案处理状态 (OPEN / ALERT_SENT / ACKNOWLEDGED)")
    owner: Mapped[str] = mapped_column(String(128), default="unassigned", doc="当前跟进责任人/辅导员")
    summary: Mapped[str] = mapped_column(Text, doc="个案核心摘要")
    handoff_summary: Mapped[str] = mapped_column(Text, default="", doc="专业交接摘要（含建议干预步骤与脱敏摘录）")
    acknowledged_by: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, doc="确认接手人")
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, doc="接手确认时间")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="个案创建时间")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="个案更新时间")


class CaseNote(Base):
    """危机个案跟进记录与流转日志模型。

    Attributes:
        id: 数据库自增主键。
        case_id: 归属的危机个案 ID。
        actor: 记录提交者姓名或工号。
        note: 跟进回访或干预措施详情。
        created_at: 记录添加时间戳。
    """

    __tablename__ = "case_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, doc="自增主键")
    case_id: Mapped[int] = mapped_column(Integer, index=True, doc="关联的个案 ID")
    actor: Mapped[str] = mapped_column(String(128), doc="操作人姓名/账号")
    note: Mapped[str] = mapped_column(Text, doc="跟进记录内容")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="记录产生时间")


class AlertRecord(Base):
    """紧急预警通知投递审计模型。

    Attributes:
        id: 数据库自增主键。
        report_id: 关联的心理测评报告 ID。
        channel: 触达通道 (email / sms / webhook / log)。
        recipient: 接收目标地址（邮箱、手机号或 Webhook URL）。
        status: 投递状态 (SUCCESS / FAILED)。
        message: 投递结果日志或异常错误摘要。
        created_at: 通知投递时间戳。
    """

    __tablename__ = "alert_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, doc="自增主键")
    report_id: Mapped[int] = mapped_column(Integer, index=True, doc="关联的心理报告 ID")
    channel: Mapped[str] = mapped_column(String(64), doc="预警渠道 (email / sms / webhook / log)")
    recipient: Mapped[str] = mapped_column(String(256), doc="通知接收人（邮箱/手机号）")
    status: Mapped[str] = mapped_column(String(32), doc="投递状态 (SUCCESS / FAILED)")
    message: Mapped[str] = mapped_column(Text, doc="发送明细日志或错误原因")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="发送时间")


class ExcelRecord(Base):
    """心理台账离线导出与归档记录模型。

    Attributes:
        id: 数据库自增主键。
        report_id: 关联归档的心理报告 ID。
        file_path: 导出的 Excel 物理文件存储路径。
        status: 归档写入状态 (SUCCESS / FAILED)。
        message: 写入结果说明或异常信息。
        created_at: 归档写入完成时间戳。
    """

    __tablename__ = "excel_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, doc="自增主键")
    report_id: Mapped[int] = mapped_column(Integer, index=True, doc="关联的心理报告 ID")
    file_path: Mapped[str] = mapped_column(String(512), doc="写入的目标 Excel 文件物理路径")
    status: Mapped[str] = mapped_column(String(32), doc="写入状态 (SUCCESS / FAILED)")
    message: Mapped[str] = mapped_column(Text, doc="写入结果说明")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="写入时间")


class ToolJob(Base):
    """异步工具执行任务作业模型 (Job Queue Entity)。

    Attributes:
        id: 数据库自增主键。
        report_id: 触发该作业的心理报告 ID。
        kind: 作业类型 (EXCEL_REPORT / CASE_CREATE / ALERT_SEND)。
        status: 当前执行状态 (PENDING / RUNNING / SUCCESS / DEAD)。
        attempts: 已重试执行次数。
        max_attempts: 允许的最大重试次数。
        depends_on_job_id: 前置依赖作业 ID（存在依赖时需等前置作业 SUCCESS 后调度）。
        run_after: 最早调度时间戳（用于指数退避重试控制）。
        last_error: 最近一次失败的错误异常堆栈。
        created_at: 作业创建入队时间戳。
        updated_at: 作业状态更新时间戳。
    """

    __tablename__ = "tool_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, doc="自增主键")
    report_id: Mapped[int] = mapped_column(Integer, index=True, doc="触发该作业的心理报告 ID")
    kind: Mapped[str] = mapped_column(String(64), index=True, doc="工具作业类型 (EXCEL_REPORT / CASE_CREATE / ALERT_SEND)")
    status: Mapped[str] = mapped_column(String(32), index=True, doc="任务状态 (PENDING / RUNNING / SUCCESS / DEAD)")
    attempts: Mapped[int] = mapped_column(Integer, default=0, doc="已尝试执行次数")
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, doc="最大允许重试次数")
    depends_on_job_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True, doc="前置依赖作业 ID")
    run_after: Mapped[datetime] = mapped_column(DateTime, default=now, index=True, doc="最早允许执行时间（用于退避重试与限流延迟）")
    last_error: Mapped[str] = mapped_column(Text, default="", doc="最后一次执行失败的异常信息")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="作业创建时间")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="作业状态更新时间")


class DeadLetterRecord(Base):
    """工具异步作业死信队列归档模型 (Dead Letter Queue)。

    Attributes:
        id: 数据库自增主键。
        job_id: 最终死亡的原作业 ID。
        report_id: 关联的心理报告 ID。
        kind: 作业类型名称。
        reason: 导致任务进入死信状态的终态异常堆栈。
        payload: 触发任务时的原始入参快照 JSON 字符串。
        created_at: 归档进入死信队列的时间戳。
    """

    __tablename__ = "dead_letter_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, doc="自增主键")
    job_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True, doc="失败的原作业 ID")
    report_id: Mapped[int] = mapped_column(Integer, index=True, doc="关联心理报告 ID")
    kind: Mapped[str] = mapped_column(String(64), index=True, doc="作业类型")
    reason: Mapped[str] = mapped_column(Text, doc="最终致死原因/堆栈信息")
    payload: Mapped[str] = mapped_column(Text, default="", doc="任务执行入参快照 (JSON)")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="进入死信队列时间")


class AgentRunTrace(Base):
    """多智能体协同运行端到端链路追踪模型 (Agent Run Trace)。

    Attributes:
        id: 数据库自增主键。
        user_id: 发起对话的用户 ID。
        session_id: 关联的会话 ID。
        report_id: 本轮运行生成的心理测评报告 ID。
        intent: 研判意图。
        risk_level: 最终安全风控等级。
        original_input: 用户原始请求文本。
        sanitized_input: 经 PII 隐私脱敏后的请求文本。
        memory_brief: 短期记忆压缩摘要。
        agent_steps_json: 各智能体决策、反思与流转步骤详情 (JSON 列表)。
        retrieved_knowledge_json: 检索召回的知识切片列表快照 (JSON 列表)。
        response_messages_json: 生成并下发的消息序列 (JSON 列表)。
        assessment_json: 心理评估详细指标与分数快照 (JSON 字典)。
        created_at: 链路追踪记录落库时间戳。
    """

    __tablename__ = "agent_run_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, doc="自增主键")
    user_id: Mapped[int] = mapped_column(ForeignKey("user_accounts.id"), index=True, doc="用户 ID")
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), index=True, doc="会话 ID")
    report_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True, doc="产生的心理报告 ID")
    intent: Mapped[str] = mapped_column(String(32), index=True, doc="识别的意图类型")
    risk_level: Mapped[str] = mapped_column(String(32), default="LOW", index=True, doc="最终风险研判等级")
    original_input: Mapped[str] = mapped_column(Text, doc="用户原始输入文本")
    sanitized_input: Mapped[str] = mapped_column(Text, doc="脱敏后的文本（去除 PII 个人隐私）")
    memory_brief: Mapped[str] = mapped_column(Text, default="", doc="短期记忆压缩摘要")
    agent_steps_json: Mapped[str] = mapped_column(Text, default="[]", doc="智能体执行步骤及协作事件序列 (JSON)")
    retrieved_knowledge_json: Mapped[str] = mapped_column(Text, default="[]", doc="RAG 召回的知识切片列表 (JSON)")
    response_messages_json: Mapped[str] = mapped_column(Text, default="[]", doc="生成回复的消息体列表 (JSON)")
    assessment_json: Mapped[str] = mapped_column(Text, default="{}", doc="心理评估详情字典 (JSON)")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="链路追踪记录创建时间")


class ToolAuditRecord(Base):
    """工具调用安全治理与访问控制审计模型 (Tool Audit Record)。

    Attributes:
        id: 数据库自增主键。
        job_id: 关联异步作业 ID。
        report_id: 关联心理测评报告 ID。
        tool_name: 被调用的工具名称。
        policy: 命中的安全治理策略名称。
        allowed: 策略检查是否允许放行。
        status: 审计状态 (AUTHORIZED / BLOCKED / SUCCESS / FAILED)。
        reason: 放行或拦截的研判依据说明。
        payload: 工具调用入参数据快照 (JSON 字典)。
        created_at: 审计记录创建时间戳。
        updated_at: 审计记录更新时间戳。
    """

    __tablename__ = "tool_audit_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, doc="自增主键")
    job_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True, doc="关联作业 ID")
    report_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True, doc="关联报告 ID")
    tool_name: Mapped[str] = mapped_column(String(64), index=True, doc="工具名称")
    policy: Mapped[str] = mapped_column(String(128), default="", doc="命中的安全治理策略名称")
    allowed: Mapped[bool] = mapped_column(Boolean, default=True, doc="策略检查是否放行通过")
    status: Mapped[str] = mapped_column(String(32), index=True, doc="审计状态 (AUTHORIZED / BLOCKED / SUCCESS / FAILED)")
    reason: Mapped[str] = mapped_column(Text, default="", doc="放行或拦截原因说明")
    payload: Mapped[str] = mapped_column(Text, default="{}", doc="工具调用入参快照 (JSON)")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="审计记录创建时间")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="审计更新时间")


class ToolExecution(Base):
    """工具调用执行记录与副作用幂等性控制模型 (Tool Execution & Idempotency Record)。

    Attributes:
        id: 数据库自增主键。
        execution_key: 全局确定性幂等键 (run_id + tool_name + args_hash)，保证幂等安全。
        run_id: 发起执行的 Agent 运行 UUID。
        user_id: 触发执行的用户 ID。
        session_id: 关联会话 public_id。
        tool_name: 调用的工具函数名称。
        status: 执行状态 (RUNNING / SUCCESS / FAILED)。
        arguments_json: 执行入参快照 JSON 字符串。
        result_json: 执行结果返回值 JSON 字符串。
        error: 异常错误信息文本。
        created_at: 工具启动执行时间戳。
        updated_at: 工具执行终态时间戳。
    """

    __tablename__ = 'tool_executions'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, doc="自增主键")
    execution_key: Mapped[str] = mapped_column(String(128), unique=True, doc="全局唯一确定性业务幂等键 (run_id + tool + args_hash)")
    run_id: Mapped[str] = mapped_column(String(64), index=True, doc="所属 Agent 单轮运行标识 UUID")
    user_id: Mapped[int] = mapped_column(Integer, index=True, doc="触发调用的用户主键 ID")
    session_id: Mapped[str] = mapped_column(String(64), default='', doc="关联对话会话 public_id")
    tool_name: Mapped[str] = mapped_column(String(100), doc="被执行的业务工具函数名称")
    status: Mapped[str] = mapped_column(String(24), default='RUNNING', doc="执行状态 (RUNNING / SUCCESS / FAILED)")
    arguments_json: Mapped[str] = mapped_column(Text, default='{}', doc="工具执行入参快照 JSON 字符串")
    result_json: Mapped[str] = mapped_column(Text, default='{}', doc="工具执行返回结果载荷 JSON 字符串")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True, doc="执行异常或失败原因详细信息")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="任务开始执行时间戳")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now, onupdate=now, doc="状态更新时间戳")


class ChatMessageRun(Base):
    """会话消息与 Agent 单轮运行强绑定关系模型 (Message-to-Run Association)。

    Attributes:
        message_id: 关联的对话消息主键 ID。
        user_id: 消息所属用户 ID。
        run_id: 产生该条消息的 Agent 运行 UUID。
    """

    __tablename__ = 'chat_message_runs'
    message_id: Mapped[int] = mapped_column(ForeignKey('chat_messages.id'), primary_key=True, doc="关联生成的对话消息主键 ID")
    user_id: Mapped[int] = mapped_column(Integer, index=True, doc="消息所有者用户 ID")
    run_id: Mapped[str] = mapped_column(String(64), index=True, doc="生成该消息的 Agent 运行 UUID")


class AnswerFeedback(Base):
    """用户对智能体回答质量与安全性的反馈评价模型 (Answer Quality Feedback)。

    Attributes:
        id: 数据库自增主键。
        user_id: 评价用户 ID。
        message_id: 被评价的助手回复消息 ID（联合唯一索引）。
        run_id: 产生该条回复的 Agent 运行 UUID。
        helpful: 是否有帮助 (True 为点赞，False 为点踩)。
        category: 反馈分类 (如 general / irrelevant / evidence / unsafe / other)。
        note: 用户补充填写的建议或文本说明。
        created_at: 反馈提交时间戳。
    """

    __tablename__ = 'answer_feedback'
    __table_args__ = (UniqueConstraint('user_id', 'message_id'),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, doc="自增主键")
    user_id: Mapped[int] = mapped_column(ForeignKey('user_accounts.id'), index=True, doc="提交反馈的用户主键 ID")
    message_id: Mapped[int] = mapped_column(ForeignKey('chat_messages.id'), doc="评价的目标对话消息主键 ID")
    run_id: Mapped[str] = mapped_column(String(64), default='', doc="生成该条回答时的 Agent Run 标识")
    helpful: Mapped[bool] = mapped_column(Boolean, doc="是否对用户有帮助 (True 为点赞，False 为点踩)")
    category: Mapped[str] = mapped_column(String(64), default='', doc="反馈类别标签 (general/irrelevant/evidence/unsafe/other)")
    note: Mapped[str] = mapped_column(Text, default='', doc="用户填写的详细建议或改进意见文本")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="反馈提交时间戳")


class KnowledgeDocumentVersion(Base):
    """知识库文档多版本流转与解析状态管理模型 (Knowledge Document Versioning)。

    Attributes:
        id: 数据库自增主键。
        document_id: 文档唯一标识 SHA-256。
        knowledge_base_id: 知识库空间标识。
        tenant_id: 租户隔离标识。
        active_key: 激活版本互斥键（保证每个文档在同一租户/库中仅有一个 ACTIVE 版本）。
        attempts: 解析构建尝试次数。
        error: 解析构建失败错误详情。
        source: 来源文件原始名称。
        content_hash: 文档二进制内容 SHA-256 指纹。
        parser: 解析器类型 (markdown / docling / pypdf)。
        parser_version: 解析器规则版本号。
        metadata_json: 文档扩展属性 JSON 字典。
        status: 版本生命周期状态 (PENDING / INDEXED / ACTIVE / FAILED / ARCHIVED / DELETE_PENDING / DELETED)。
        created_at: 版本记录创建时间戳。
    """

    __tablename__ = 'knowledge_document_versions'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, doc="自增主键")
    document_id: Mapped[str] = mapped_column(String(64), default='', index=True, doc="所属文档全局唯一标识 SHA-256")
    knowledge_base_id: Mapped[str] = mapped_column(String(64), default='default', index=True, doc="知识库空间隔离标识")
    tenant_id: Mapped[str] = mapped_column(String(64), default='default', index=True, doc="多租户隔离标识")
    active_key: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, unique=True, doc="全局唯一激活互斥键，保证同一租户库内仅有一个 ACTIVE 版本")
    attempts: Mapped[int] = mapped_column(Integer, default=0, doc="解析与向量化失败重试次数")
    error: Mapped[str] = mapped_column(Text, default='', doc="解析或构建失败的异常详细信息")
    source: Mapped[str] = mapped_column(String(256), index=True, doc="知识来源文件名或标识 (如 manual.pdf)")
    content_hash: Mapped[str] = mapped_column(String(64), index=True, doc="文档内容 SHA-256 校验指纹哈希")
    parser: Mapped[str] = mapped_column(String(64), default='markdown', doc="文档解析器类型 (markdown / docling / pypdf)")
    parser_version: Mapped[str] = mapped_column(String(64), default='1', doc="解析引擎协议或算法版本号")
    metadata_json: Mapped[str] = mapped_column(Text().with_variant(LONGTEXT(), "mysql"), default='{}', doc="文档扩展元数据字典序列化 JSON")
    status: Mapped[str] = mapped_column(String(24), default='PENDING', doc="PENDING / INDEXED / ACTIVE / FAILED / ARCHIVED / DELETE_PENDING / DELETED")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="版本记录创建时间戳")


class RunMetrics(Base):
    """单轮 Agent 执行耗时与性能度量统计模型 (Run Latency & Performance Metrics)。

    Attributes:
        run_id: Agent 单轮运行 UUID（主键）。
        user_id: 发起请求的用户 ID。
        first_token_seconds: Time to First Token (TTFT) 首字流式输出延迟（秒）。
        elapsed_seconds: 完整端到端交互总耗时（秒）。
        created_at: 度量记录创建时间戳。
    """

    __tablename__ = 'run_metrics'
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True, doc="Agent 单轮运行唯一标识 UUID")
    user_id: Mapped[int] = mapped_column(Integer, index=True, doc="请求触发用户主键 ID")
    first_token_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True, doc="首字流式输出时间延迟 (Time to First Token，秒)")
    elapsed_seconds: Mapped[float] = mapped_column(Float, doc="整轮完整交互端到端总执行耗时（秒）")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="指标生成记录时间戳")


class AgentStoreItem(Base):
    """智能体长期命名空间键值记忆存储模型 (Agent Persistent KV Memory Store)。

    Attributes:
        id: 数据库自增主键。
        namespace: 业务隔离命名空间（如 user_memories:1001）。
        key: 命名空间内唯一键名。
        value_json: 存储的数据 JSON 序列化字符串。
        created_at: 记录写入时间戳。
    """

    __tablename__ = 'agent_store_items'
    __table_args__ = (UniqueConstraint('namespace', 'key'),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, doc="自增主键")
    namespace: Mapped[str] = mapped_column(String(200), index=True, doc="业务隔离命名空间 (如 user_memories:1001)")
    key: Mapped[str] = mapped_column(String(100), doc="命名空间内唯一变量键名 (如 profile / preferences)")
    value_json: Mapped[str] = mapped_column(Text, default='{}', doc="存储的数据对象 JSON 序列化字符串")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, doc="键值创建或写入时间戳")
