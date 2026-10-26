"""SolaceAI 心理评估跨运行时统一门面与兼容导出模块 (Psychological Assessment Facade)

本模块作为整个系统心理状态研判与危机识别的核心入口门面 (Facade Pattern)：
1. 极端高危信号零延迟前置拦截：对于自残、轻生等明确高危表达，直接旁路跳过模型推理，
   秒级触发确定性规则防护 (`skipped_high_risk_rule`)；
2. 跨后端（Online / Offline）动态多态路由：依据运行配置中的 `risk_assessment_source`
   以及分类器启用状态，自动委托给对应的离线本地模型后端或云端大模型后端执行评估；
3. 历史兼容导出：通过模块级 `__getattr__` 保持对旧版代码引用类名的无缝兼容。
"""
from typing import Any

from app.domain.assessment import (
    PsychologyAssessment,
    assessment_from_local_label,
    heuristic,
    risk_from_score,
    risk_order,
    score_for_emotion,
)
from app.services.ai import has_high_risk_signal, has_consult_signal
from app.core.backends import backend


def assess_with_source(
    settings: Any,
    safety_client: Any,
    text: str,
    history: list[Any] | None = None,
    emotion_client: Any = None,
) -> tuple[PsychologyAssessment, dict[str, Any]]:
    """根据系统配置与安全规则执行多维度心理状态研判 (Execute Assessment with Source Routing)

    执行逻辑：
    1. 规则前置审查：利用关键词与正则快速检测是否包含自杀自残等高危信号。
       若命中，立即返回确定性高危评估结果，标记来源为 'high_risk_rule' 并旁路跳过深度模型；
    2. 后端路由决策：
       - 若配置为 'local'，或配置为 'auto' 且启用了本地微调分类器，则分发至 'offline' 后端；
       - 否则分发至 'online' 云端大模型后端；
    3. 调用对应后端的 `assess` 接口获取研判结果与观测上下文；
    4. 针对 legacy 兼容模式，若需要同时观测情绪标签，自动合并离线情绪分类特征。

    Args:
        settings: 系统全局配置实例。
        safety_client: 安全/通用大模型推理客户端。
        text (str): 学生本轮输入的原始文本。
        history (list[Any] | None, optional): 历史对话消息上下文，默认为 None。
        emotion_client (Any, optional): 本地微调情绪分类客户端，默认为 None。

    Returns:
        tuple[PsychologyAssessment, dict[str, Any]]: 
            - PsychologyAssessment: 包含情绪标签、量化分值、风险等级与摘要的评估实体；
            - dict[str, Any]: 包含 assessmentSource、degraded 等溯源和观测属性的字典。
    """
    if has_high_risk_signal(text):
        return heuristic(text), dict(
            label=None,
            status='skipped_high_risk_rule',
            assessmentSource='high_risk_rule',
            degraded=False,
        )

    local = settings.risk_assessment_source == 'local' or (
        settings.risk_assessment_source == 'auto'
        and settings.emotion_classifier_enabled
        and settings.ai_provider != 'mock'
    )
    result, observation = backend('offline' if local else 'online').assess(
        settings, safety_client, text, history, emotion_client
    )

    if not local and settings.runtime_profile == 'legacy' and settings.emotion_classifier_enabled:
        legacy = backend('offline').observe_emotion(settings, text, emotion_client)
        legacy.update(
            assessmentSource=observation['assessmentSource'],
            degraded=observation['degraded'],
        )
        observation = legacy

    return result, observation


def __getattr__(name: str) -> Any:
    """动态模块属性解析，兼容旧版本对具体服务类的直接导入 (Dynamic Attribute Resolver)

    Args:
        name (str): 导入的目标符号名称。

    Returns:
        Any: 目标后端中注册的对应类或对象。

    Raises:
        AttributeError: 当请求的符号既不是兼容类名也不存在时抛出。
    """
    # 历史调用方可能直接由此处导入这些类名；正常运行时仅按需加载选定的适配器
    if name in ('PsychologicalAssessmentService', 'OnlineAssessmentService'):
        return getattr(backend('online'), name)
    if name == 'OfflineAssessmentService':
        return getattr(backend('offline'), name)
    raise AttributeError(name)
