"""SolaceAI 线下心理风险评估套件 (Offline Psychological Risk Assessment Suite).

专职面向校园/内网本地离线部署环境：
1. 本地微调 Qwen3-4B 双 LoRA 模型 (MLX 4-bit / Metal 原生加速，分类 LoRA 与降级回答 LoRA) 推理；
2. 确定性映射规则：将分类标签映射为业务风险等级（正常->LOW, 焦虑->LOW, 低落->MEDIUM, 高风险->HIGH）；
3. 严格规则优先防御：检测到自杀自残等高危信号跳过模型直接触发最高等级干预；
4. 故障闭环降级：模型超时或输出异常时回退至本地词法启发式规则兜底；
5. 全流程物理隔离：严格杜绝任何外部云端大模型 API 网络调用，杜绝隐私数据泄露。
"""

from __future__ import annotations

from app.domain.assessment import PsychologyAssessment, heuristic, assessment_from_local_label, score_for_emotion, risk_from_score, risk_order
from typing import Any

from app.core.config import Settings
from app.core.enums import EmotionLabel, RiskLevel
from app.schemas.dtos import AiMessage
from app.services.ai import has_consult_signal, has_high_risk_signal
from apps.offline.emotion_classifier import observe_emotion


class OfflineAssessmentService:
    """线下心理风险评估服务 (Offline Assessment Service)."""

    def __init__(self, settings: Settings, emotion_client: Any = None):
        """初始化线下心理评估服务。

        Args:
            settings: 系统配置实例。
            emotion_client: 可选注入的情绪分类客户端（用于单元测试或自定义模型）。
        """
        self.settings = settings
        self.emotion_client = emotion_client

    def assess(
        self,
        text: str,
        history: list[AiMessage] | None = None,
        emotion_client: Any = None,
    ) -> tuple[PsychologyAssessment, dict]:
        """对用户当前输入执行纯离线心理健康评估与危机研判。

        Args:
            text: 用户输入的原始文本。
            history: 对话历史消息列表（可选）。
            emotion_client: 运行时注入的情绪分类客户端（可选）。

        Returns:
            tuple[PsychologyAssessment, dict]: 标准化心理评估实体及观测元数据字典。
        """
        client = emotion_client or self.emotion_client

        # 1. 优先拦截明确的高危自伤/自杀危机信号
        if has_high_risk_signal(text):
            return heuristic(text), {
                "label": None,
                "status": "skipped_high_risk_rule",
                "assessmentSource": "high_risk_rule",
                "degraded": False,
            }

        # 2. 强制开启本地分类器设置进行离线情绪感知
        local_settings = self.settings.model_copy(
            update={"emotion_classifier_enabled": True}
        )
        observation = observe_emotion(local_settings, text, client)

        if observation.get("status") == "ok" and observation.get("label"):
            assessment = assessment_from_local_label(observation["label"])
            observation.update(assessmentSource="local_qwen", degraded=False)
        else:
            assessment = heuristic(text)
            assessment.summary = "本地分类不可用，规则兜底。" + assessment.summary
            observation.update(assessmentSource="local_heuristic", degraded=True)

        return assessment, observation


