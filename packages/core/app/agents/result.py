"""SolaceAI 多智能体协同运行输出模型 (Multi-Agent Run Result Models)

本模块定义了多智能体系统完成单轮规划、工具调用、知识检索与安全审核后的标准输出模型：
1. `AgentStep`: 单步决策与执行追踪数据结构，记录每一步操作的智能体角色、动作、观测、耗时及 Token 消耗；
2. `AgentRunResult`: 单轮 Agent 执行的完整聚合结果包，包含识别的意图、风险等级、召回的知识切片、
   组装的 Prompt 消息流、模型路由元数据、多步追踪明细、记忆简报及最终生成的安全回复。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.enums import IntentType, RiskLevel
from app.schemas.dtos import AiMessage
from app.services.assessment import PsychologyAssessment
from app.services.ai import AgentModelRoute
from app.services.knowledge import SearchResult


@dataclass
class AgentStep:
    """单步智能体执行追踪数据点 (Agent Execution Step Trace)

    用于对状态图内部每一次模型决策、工具调用或安全审查进行微观度量与审计追踪。

    Attributes:
        step (int): 步骤在单轮执行中的递增序号（从 1 开始）。
        agent (str): 执行当前步骤的智能体角色名称（如 'MainAgent', 'SafetyAgent'）。
        action (str): 执行的具体动作或调用的工具函数名。
        observation (str): 动作执行后返回的观测结果文本或状态摘要。
        duration_ms (float | None, optional): 当前步骤耗时（毫秒）。
        model (str | None, optional): 触发当前步骤调用的底层大语言模型标识。
        tokens (dict | None, optional): 记录当前步骤消耗的 prompt_tokens 与 completion_tokens 统计。
    """
    step: int
    agent: str
    action: str
    observation: str
    duration_ms: float | None = None
    model: str | None = None
    tokens: dict | None = None


@dataclass
class AgentRunResult:
    """多智能体协同运行的完整输出聚合对象 (Multi-Agent Run Result Aggregate)

    汇聚单轮对话中各智能体协作产生的全量领域数据，供 Harness 持久化并向调用方回传。

    Attributes:
        intent (IntentType): 研判得到的用户对话意图分类 (CHAT / CONSULT / CRISIS)。
        risk_level (RiskLevel): 综合评估确定的心理危机风险等级 (LOW / MEDIUM / HIGH)。
        assessment (PsychologyAssessment | None): 心理量化评估与归因分析结构体。
        retrieved_knowledge (list[SearchResult]): 知识库检索召回并用于生成上下文的切片列表。
        response_messages (list[AiMessage]): 送入大模型最终推理的标准化上下文 Prompt 消息列表。
        response_model_route (AgentModelRoute): 最终生成阶段采用的模型提供方与路由配置。
        steps (list[AgentStep]): 本轮执行产生的所有决策与工具调用审计步骤。
        memory_brief (str): 提取并注入的当前用户历史工作记忆摘要简报。
        collaboration_events (list[Any]): 协同事件列表（用于前端可视化状态总线展示）。
        collaboration_tasks (list[Any]): 协同子任务状态列表。
        collaboration_artifacts (list[Any]): 运行时生成的中间工件列表。
        final_response (str | None, optional): 经过安全审核与合规改写后的最终下发回复文本。
        citations (list[dict]): 最终回复中引用的真实知识库文献切片元数据。
        run_id (str | None, optional): 本轮运行全局唯一 UUID 追踪标识。
        model_visible_evidence (list[dict]): 实际对大模型可见的精简知识切片内容快照。
    """
    intent: IntentType
    risk_level: RiskLevel
    assessment: PsychologyAssessment | None
    retrieved_knowledge: list[SearchResult]
    response_messages: list[AiMessage]
    response_model_route: AgentModelRoute
    steps: list[AgentStep]
    memory_brief: str
    collaboration_events: list[Any] = field(default_factory=list)
    collaboration_tasks: list[Any] = field(default_factory=list)
    collaboration_artifacts: list[Any] = field(default_factory=list)

    final_response: str | None = None
    citations: list[dict] = field(default_factory=list)
    run_id: str | None = None
    model_visible_evidence: list[dict] = field(default_factory=list)

    @property
    def requires_report(self) -> bool:
        """业务判定属性：是否需要生成并持久化心理评估报告

        领域规则：普通闲聊（CHAT）意图无需持久化心理评估报告；
        仅当用户属于心理咨询（CONSULT）或高危危机（RISK）时，才创建 `PsychologicalReport`。

        Returns:
            bool: 需生成评估报告返回 True，否则返回 False。
        """
        return self.intent != IntentType.CHAT
