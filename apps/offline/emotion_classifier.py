"""SolaceAI 本地微调情绪分类与风险决策观测服务 (Emotion Classifier Service)

本模块结合本地部署的领域微调模型，为多智能体状态图提供轻量级的情绪初筛观测特征：
1. 四分类情绪标签空间：'正常', '焦虑', '低落', '高风险'；
2. 本地推理隔离：强制使用本地 Ollama 端点，防御向外部付费 API 泄露学生心理隐私；
3. 纯观测辅助定位：情绪分类结果作为 `MainAgent` 决策时的输入观测特征之一，
   但不替代 `SafetyAgent` 独立决定最终的风险等级；
4. 容错降级机制：当本地分类器不可用或超时时，安全标记为 'unavailable' 并由启发式规则与大模型兜底。
"""

import time
import httpx

from app.schemas.dtos import AiMessage
from app.services.privacy import PrivacySanitizer

# 系统支持的四分类标准情绪输出标签元组
LABELS = ('正常', '焦虑', '低落', '高风险')


class EmotionClassifier:
    """情绪分类器核心逻辑包装器 (Emotion Classifier Core Wrapper)

    组装受限提示词，驱动轻量推理模型输出指定受限集合内的情绪分类标签。
    """

    def __init__(self, client):
        """初始化情绪分类器

        Args:
            client: 具备 `complete(messages)` 接口的模型推理客户端实例。
        """
        self.client = client

    def classify(self, text: str) -> str:
        """对给定用户输入文本执行单标签情绪分类 (Classify Emotion Label)

        强制注入脱敏前处理与单标签限制指令。

        Args:
            text (str): 待分析的用户输入原始文本。

        Returns:
            str: 匹配为 '正常' | '焦虑' | '低落' | '高风险' 的标签字符串。

        Raises:
            ValueError: 当模型输出不在预设合法标签集合内时抛出。
        """
        result = self.client.complete([
            AiMessage(role='system', content='你是一个严格遵守指令的中文文本分类助手。'),
            AiMessage(
                role='user',
                content='分析用户文本情绪，只能输出：正常、焦虑、低落、高风险\n\n用户文本：' + PrivacySanitizer().sanitize(text),
            ),
        ]).strip()
        if result not in LABELS:
            raise ValueError('Invalid emotion classifier output')
        return result


class LocalEmotionClient:
    """基于本地 Ollama 服务的轻量分类推理客户端 (Local Ollama Emotion Client)

    封装对本地 Ollama 服务的 HTTP 调用，并做本地端点安全校验。
    """

    def __init__(self, settings):
        """初始化本地情绪推理客户端

        Args:
            settings: 系统配置对象。
        """
        self.settings = settings

    def complete(self, messages: list[AiMessage]) -> str:
        """向本地 Ollama HTTP 接口同步发送非流式推理请求 (Inference via Ollama)

        Args:
            messages (list[AiMessage]): 输入上下文消息序列。

        Returns:
            str: Ollama 生成的返回文本。

        Raises:
            ValueError: 若 Ollama 地址未指向本地地址（禁止云端中转）。
            HTTPStatusError: 网络请求非 200 时抛出。
        """
        from app.services.api_budget import is_local_endpoint

        # 安全审查：确保端点严格为本地回环或局域网私有地址，禁止通过外部公网传输
        if not is_local_endpoint(self.settings.ollama_base_url):
            raise ValueError('Emotion inference must use a configured local endpoint')

        response = httpx.post(
            self.settings.ollama_base_url.rstrip('/') + '/api/chat',
            json={
                'model': self.settings.finetuned_model_name,
                'messages': [m.model_dump() for m in messages],
                'stream': False,
                'keep_alive': 0,  # 本地演示环境下在 BGE 与 Cross-Encoder 推理前立即释放分类器显存与内存
                'options': {'temperature': 0, 'num_predict': 16},
            },
            timeout=self.settings.emotion_classifier_timeout_seconds,
            trust_env=False,
        )
        response.raise_for_status()
        return response.json()['message']['content']


def observe_emotion(settings, text: str, client=None) -> dict:
    """安全观察输入文本的情绪特征并度量推理开销 (Observe Emotion Features)

    非侵入式辅助观测函数，具备全套超时保护、离线安全与毫秒级耗时追踪。

    Args:
        settings: 系统全局配置实例。
        text (str): 待研判的用户输入文本。
        client (optional): 依赖注入的测试 Mock 客户端或自定义推理客户端。

    Returns:
        dict: 包含 'label'(标签), 'status'('ok'/'disabled'/'offline'/'unavailable'),
            'elapsedMs'(耗时毫秒) 等度量属性的字典。
    """
    observed_provider = getattr(client, 'provider', None) if client is not None else None
    observed_model = getattr(client, 'model', None) if client is not None else None
    result = {
        'label': None,
        'status': 'disabled',
        'provider': observed_provider if isinstance(observed_provider, str) else 'ollama',
        'model': observed_model if isinstance(observed_model, str) else settings.finetuned_model_name,
        'clinicalValidated': False,
    }

    # 1. 功能总开关检查
    if not settings.emotion_classifier_enabled:
        return result

    # 2. 离线 Mock 环境支持
    if settings.ai_provider == 'mock' and client is None:
        result['status'] = 'offline'
        return result

    # 3. 执行推理与耗时统计
    start = time.perf_counter()
    try:
        classifier_client = client or LocalEmotionClient(settings)
        result['label'] = EmotionClassifier(classifier_client).classify(text)
        result['status'] = 'ok'
    except Exception as exc:
        result['status'] = 'unavailable'
        result['errorType'] = type(exc).__name__

    result['elapsedMs'] = (time.perf_counter() - start) * 1000
    return result
