"""SolaceAI 核心领域枚举定义模块 (Core Domain Enums)

本模块集中定义系统各业务子域中使用的标准状态、分类与类型常量，包含：
1. 消息角色类型 (MessageRole)；
2. 用户对话意图分类 (IntentType)；
3. 心理危机风险等级 (RiskLevel)；
4. 情绪细粒度状态分类 (EmotionLabel)；
5. 工具调用终态 (ToolStatus)；
6. 异步工具作业种类 (ToolJobKind)；
7. 异步任务生命周期状态 (ToolJobStatus)；
8. 危机干预个案流转状态 (RiskCaseStatus)。
"""

from enum import Enum


class MessageRole(str, Enum):
    """对话消息角色枚举 (Conversation Message Role)

    标识在人机心理咨询或日常对话流中每条消息的发出主体。
    """
    USER = "USER"
    """终端学生/用户发送的对话消息"""

    ASSISTANT = "ASSISTANT"
    """AI 心理助手/大语言模型生成的回复与干预话术"""

    SYSTEM = "SYSTEM"
    """系统级指令、角色设定提示词、安全合规边界或长期记忆摘要注入"""


class IntentType(str, Enum):
    """用户意图分类枚举 (User Intent Classification)

    由意图识别前置分类节点给出，决定后续调度进入哪一条业务处理链路。
    """
    CHAT = "CHAT"
    """普通日常闲聊、学业探讨、编程技术或事务性问答（不触发心理干预流程与风险报告评估）"""

    CONSULT = "CONSULT"
    """心理健康倾诉、情绪疏导、压力倾诉或人际关系困扰（触发 RAG 检索与共情对话策略）"""

    RISK = "RISK"
    """出现明确自残、自杀危机或伤害倾向信号（强制触发高危熔断保护与紧急干预工作流）"""


class RiskLevel(str, Enum):
    """心理危机风险等级枚举 (Psychological Risk Level)

    依据临床危机干预标准划分的三级风险阶梯，用于指导干预强度与线下告警分级。
    """
    LOW = "LOW"
    """低风险：存在轻微情绪波动或一般日常适应问题，采取常规倾听、共情与非侵入式疏导建议"""

    MEDIUM = "MEDIUM"
    """中风险：表现出中度抑郁、焦虑倾向或较长时间的负面情绪，自动生成个案台账供辅导员线下跟进"""

    HIGH = "HIGH"
    """高风险：存在明确自杀/自残计划、极度绝望或严重危机，立即触发紧急熔断、邮件预警与线下抢预干预"""


class EmotionLabel(str, Enum):
    """细粒度情绪状态标签枚举 (Fine-Grained Emotion Status)

    由微调情绪分类模型或大模型分析出的当前输入主要情绪特征。
    """
    NORMAL = "NORMAL"
    """情绪平稳、积极或无显著负面心理表征"""

    ANXIETY = "ANXIETY"
    """焦虑、紧张不安、恐慌、学业/考试压力或过度担忧"""

    DEPRESSED = "DEPRESSED"
    """情绪低落、抑郁、无精打采、自责内疚或无价值感"""

    HIGH_RISK = "HIGH_RISK"
    """极度绝望、严重心理危机或伴有死亡念头的高危状态"""


class ToolStatus(str, Enum):
    """工具调用执行终态枚举 (Tool Execution Terminal Status)

    记录智能体调用本地业务工具或外部 MCP 服务的最终执行结果。
    """
    SUCCESS = "SUCCESS"
    """工具成功执行完成，输出有效结构化结果"""

    FAILED = "FAILED"
    """工具执行中发生未捕获异常、超时或返回失败标识"""


class ToolJobKind(str, Enum):
    """异步工具作业类型枚举 (Asynchronous Tool Job Kind)

    用于多智能体系统在会话主链路之外触发的耗时或副作用后台作业调度。
    """
    EXCEL_REPORT = "EXCEL_REPORT"
    """异步将危机信息与评估结果追加记录到辅导员 Excel 风险台账文件"""

    CASE_CREATE = "CASE_CREATE"
    """自动创建或更新并流转中高风险心理预警个案 (RiskCase)"""

    ALERT_SEND = "ALERT_SEND"
    """发送高危紧急告警（执行邮件预警投递与日志紧急广播）"""

    RISK_ALERT = "RISK_ALERT"
    """兼容旧版本的直接告警任务类型"""


class ToolJobStatus(str, Enum):
    """异步工具任务生命周期状态枚举 (Async Job Lifecycle Status)

    用于作业调度器维护任务队列流转与幂等防重执行。
    """
    PENDING = "PENDING"
    """作业排队中，等待后台 Worker 调度执行"""

    RUNNING = "RUNNING"
    """作业已被 Worker 认领，正在执行中"""

    SUCCESS = "SUCCESS"
    """作业执行成功，已完成归档记录"""

    DEAD = "DEAD"
    """作业执行失败并已达到最大重试限制，转入死信表 (DeadLetterRecord) 供人工排查"""


class RiskCaseStatus(str, Enum):
    """危机个案流转生命周期枚举 (Risk Case Management Status)

    辅导员或心理咨询师线下介入跟进时的闭环状态标识。
    """
    OPEN = "OPEN"
    """新建待跟进：系统自动识别并立案，等待人工辅导员审阅与介入"""

    ALERT_SENT = "ALERT_SENT"
    """预警已发送：系统已成功外发紧急邮件通知或站内警报"""

    ACKNOWLEDGED = "ACKNOWLEDGED"
    """已确认接手：辅导员/咨询师已在后台确认该个案并在线下开展面对面干预"""

    CLOSED = "CLOSED"
    """个案已结案：危机已平息或已安全转介至专业精神卫生医疗机构，个案生命周期结束"""
