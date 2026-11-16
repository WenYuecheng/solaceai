"""SolaceAI 线上轻量心理评估套件 (Online Lightweight Psychological Assessment Suite)

专职面向云端轻量化与低资源容器部署：
1. 基于公有云大语言模型 API 协议在线完成多维度心理状态研判；
2. 结构化 JSON 提示词输出契约校验（情绪、强度得分、风险等级、评估摘要）；
3. 高危信号拦截优先保证：遇到自伤或轻生极端词立即旁路触发规则兜底；
4. 异常自动降级机制：大模型超时或 JSON 解析异常自动回退至本地启发式规则。
"""

from __future__ import annotations

import json
from typing import Any

from app.core.config import Settings
from app.core.enums import EmotionLabel, RiskLevel
from app.domain.assessment import (
    PsychologyAssessment,
    heuristic,
    score_for_emotion,
)
from app.schemas.dtos import AiMessage
from app.services.ai import AiClient, PromptTemplates, has_high_risk_signal


def observe_emotion(settings: Settings, text: str, client: Any = None) -> dict[str, Any]:
    """收集在线模式下的情绪分类观测特征 (Online Emotion Observation Stub)

    在线模式禁用本地微调模型，因此返回禁用状态占位字典，避免对外部 API 泄露学生心理细节。

    Args:
        settings (Settings): 系统配置实例。
        text (str): 学生输入咨询文本。
        client (Any, optional): 情绪客户端实例，默认为 None。

    Returns:
        dict[str, Any]: 结构化情绪观测特征字典（标记 status 为 'disabled'）。
    """
    return dict(
        label=None,
        status='disabled',
        provider='ollama',
        model=settings.finetuned_model_name,
        clinicalValidated=False,
    )


class PsychologicalAssessmentService:
    """云端大语言模型心理评估核心服务 (Psychological Assessment Cloud Evaluator)

    通过系统提示词引导大语言模型输出规范的 JSON 格式评估对象。
    """

    def __init__(self, ai: AiClient):
        """初始化云端心理评估服务。

        Args:
            ai (AiClient): 大模型交互客户端。
        """
        self.ai = ai
        self.last_source = "cloud"

    def assess(
        self, text: str, history: list[AiMessage] | None = None
    ) -> PsychologyAssessment:
        """对学生咨询文本执行云端大模型结构化研判 (Assess Psychological State via Cloud LLM)

        执行逻辑：
        1. 高危硬规则拦截：命中强危机关键词直接判定为 HIGH 风险并返回，不调用模型；
        2. 提示词组装与调用：要求模型输出 JSON 格式（包含 emotion, emotionScore, risk, summary）；
        3. 容错解析与范围钳位：提取 JSON 字符串并进行数值范围合法性校验；
        4. 异常降级：发生网络超时或模型拒绝时，回退至本地确定性词法规则 `heuristic`。

        Args:
            text (str): 学生本轮输入的咨询文本。
            history (list[AiMessage] | None, optional): 历史对话上下文。

        Returns:
            PsychologyAssessment: 结构化心理评估领域值对象。
        """
        self.last_source = "cloud"
        if has_high_risk_signal(text):
            self.last_source = "high_risk_rule"
            return PsychologyAssessment(
                EmotionLabel.HIGH_RISK, 4.0, RiskLevel.HIGH, "检测到明确高风险表达"
            )

        try:
            raw = self.ai.complete(
                PromptTemplates.psychology_prompt(history or [], text),
                response_format="json_object",
            )
            start = raw.find("{")
            end = raw.rfind("}")
            data = json.loads(raw[start : end + 1] if start >= 0 and end > start else raw)

            emotion = EmotionLabel(data.get("emotion", "NORMAL").upper())
            score = max(
                0.0,
                min(
                    4.0,
                    float(data.get("emotionScore", score_for_emotion(emotion))),
                ),
            )
            risk = RiskLevel(data.get("risk", "LOW").upper())

            return PsychologyAssessment(
                emotion, score, risk, data.get("summary", "模型评估结果")
            )
        except Exception:
            self.last_source = "local_heuristic"
            return heuristic(text)


class OnlineAssessmentService:
    """线上轻量心理评估门面服务 (Online Assessment Facade Service)

    整合大模型研判与状态观测元数据。
    """

    def __init__(self, settings: Settings, ai_client: AiClient | None = None):
        """初始化线上评估服务。

        Args:
            settings (Settings): 系统配置实例。
            ai_client (AiClient | None, optional): 大语言模型客户端实例。
        """
        self.settings = settings
        self.ai_client = ai_client or AiClient(settings)
        self.evaluator = PsychologicalAssessmentService(self.ai_client)

    def assess(
        self,
        text: str,
        history: list[AiMessage] | None = None,
        emotion_client: Any = None,
    ) -> tuple[PsychologyAssessment, dict[str, Any]]:
        """执行端到端线上心理评估并聚合来源观测数据 (Assess and Collect Observation Metadata)

        Args:
            text (str): 学生输入的咨询文本。
            history (list[AiMessage] | None, optional): 历史对话记录。
            emotion_client (Any, optional): 情绪分析客户端。

        Returns:
            tuple[PsychologyAssessment, dict[str, Any]]: (评估结果值对象, 溯源及降级状态元数据字典)。
        """
        # 1. 规则硬拦截
        if has_high_risk_signal(text):
            return heuristic(text), {
                "label": None,
                "status": "skipped_high_risk_rule",
                "assessmentSource": "high_risk_rule",
                "degraded": False,
            }

        # 2. 云端结构化模型评估
        assessment = self.evaluator.assess(text, history or [])

        # 3. 收集观测元数据
        observation = observe_emotion(self.settings, text, emotion_client)
        observation.update(
            assessmentSource=self.evaluator.last_source,
            degraded=self.evaluator.last_source == "local_heuristic",
        )
        return assessment, observation
