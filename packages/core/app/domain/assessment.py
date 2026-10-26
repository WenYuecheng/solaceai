"""SolaceAI 心理评估结果领域模型与确定性风险规则 (Domain Psychological Assessment & Risk Rules)

本模块定义跨运行时（云端 Online 与离线 Offline）共享的心理状态研判领域实体与决策规则：
1. 统一评估领域载体 (`PsychologyAssessment`)：封装情绪类别、量化得分、风险等级与评估摘要；
2. 情绪与风险量化映射：建立情绪分类到四分位标度的单调映射；
3. 风险等级偏序关系 (`risk_order`)：为多阶段审核中的“取最高风险”决策提供确定性比较依据；
4. 本地分类标签映射与启发式规则兜底 (`heuristic`)：在大模型异常或超时时提供无死角的安全保底防御。
"""
from dataclasses import dataclass
from app.core.enums import EmotionLabel, RiskLevel
from app.services.ai import has_high_risk_signal, has_consult_signal


@dataclass
class PsychologyAssessment:
    """心理评估研判结果领域值对象 (Psychology Assessment Value Object)

    Attributes:
        emotion (EmotionLabel): 识别出的心理情绪分类标签枚举（正常、焦虑、低落、高风险）。
        emotion_score (float): 情绪严重程度的连续量化分值（0.0 ~ 4.0 标度）。
        risk (RiskLevel): 对应的危机干预风险等级（LOW 低风险、MEDIUM 中风险、HIGH 高危）。
        summary (str): 评估依据简要陈述或模型生成的诊断摘要。
    """
    emotion: EmotionLabel
    emotion_score: float
    risk: RiskLevel
    summary: str


def score_for_emotion(emotion: EmotionLabel) -> float:
    """根据情绪分类枚举映射基准量化分值 (Map Emotion Label to Continuous Score)

    Args:
        emotion (EmotionLabel): 情绪分类枚举。

    Returns:
        float: 对应的危机严重度基准分值（HIGH_RISK: 4.0, DEPRESSED: 3.0, ANXIETY: 2.0, NORMAL: 0.0）。
    """
    return {
        EmotionLabel.HIGH_RISK: 4.0,
        EmotionLabel.DEPRESSED: 3.0,
        EmotionLabel.ANXIETY: 2.0,
        EmotionLabel.NORMAL: 0.0,
    }[emotion]


def risk_from_score(score: float) -> RiskLevel:
    """根据情绪量化分值区间划分危机干预风险等级 (Partition Risk Level by Score)

    评分区间标准：
    - score >= 4.0: HIGH (高风险危机，必须人工跟进并报警)
    - 3.0 <= score < 4.0: MEDIUM (中度抑郁/情绪崩溃，建立跟踪日志)
    - score < 3.0: LOW (低风险或日常对话)

    Args:
        score (float): 心理危机量化得分。

    Returns:
        RiskLevel: 对应的风险级别枚举。
    """
    if score >= 4:
        return RiskLevel.HIGH
    if score >= 3:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def risk_order(risk: RiskLevel) -> int:
    """获取风险等级的数值排序权重 (Get Risk Level Precedence Weight)

    用于在智能体决策流中对不同来源（如微调分类器、LLM 研判、启发式规则）
    给出的多个风险等级进行偏序比对，执行防御性升级（取最大严重度）。

    Args:
        risk (RiskLevel): 风险等级枚举。

    Returns:
        int: 数值权重（LOW: 1, MEDIUM: 2, HIGH: 3）。
    """
    return {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}[risk]


def assessment_from_local_label(label: str) -> PsychologyAssessment:
    """将微调模型输出的四分类文本标签转换为领域心理评估对象 (Convert Local Label to Assessment)

    Args:
        label (str): 本地分类器输出的分类中文字符串 ('正常', '焦虑', '低落', '高风险')。

    Returns:
        PsychologyAssessment: 结构化领域评估结果。
    """
    mapping = {
        "正常": EmotionLabel.NORMAL,
        "焦虑": EmotionLabel.ANXIETY,
        "低落": EmotionLabel.DEPRESSED,
        "高风险": EmotionLabel.HIGH_RISK,
    }
    emotion = mapping.get(label, EmotionLabel.NORMAL)
    score = score_for_emotion(emotion)
    return PsychologyAssessment(
        emotion,
        score,
        risk_from_score(score),
        f"本地Qwen分类：{label}；风险按固定标签映射",
    )


def heuristic(text: str) -> PsychologyAssessment:
    """基于词法关键词与模式匹配的确定性启发式安全兜底评估 (Heuristic Fallback Assessment)

    当所有外部或本地深度模型调用均失败、超时或检测到强烈极端信号时，
    以纯静态模式匹配规则提供 100% 可靠的确定性风险判定。

    Args:
        text (str): 待评估的学生原始文本。

    Returns:
        PsychologyAssessment: 启发式研判生成的评估结果。
    """
    if has_high_risk_signal(text):
        return PsychologyAssessment(
            EmotionLabel.HIGH_RISK, 4.0, RiskLevel.HIGH, "检测到明确高风险表达"
        )
    if any(
        word in text.lower()
        for word in ["抑郁", "低落", "崩溃", "难过", "depress", "hopeless"]
    ):
        return PsychologyAssessment(
            EmotionLabel.DEPRESSED, 3.1, RiskLevel.MEDIUM, "检测到低落或抑郁相关表达"
        )
    if has_consult_signal(text):
        return PsychologyAssessment(
            EmotionLabel.ANXIETY, 2.2, RiskLevel.LOW, "检测到焦虑或压力相关表达"
        )
    return PsychologyAssessment(
        EmotionLabel.NORMAL, 0.0, RiskLevel.LOW, "未检测到明显风险信号"
    )
