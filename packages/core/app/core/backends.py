"""SolaceAI 运行时后端动态加载与解耦适配器 (Dynamic Backend Loader & SPI Discovery)

本模块定义了系统的后端驱动服务提供者接口 (Service Provider Interface, SPI)：
1. 架构解耦与物理隔离：`solaceai-core` 核心业务逻辑不直接强依赖 Chroma 或 Milvus/PyTorch；
2. 基于 `entry_points` 的动态服务发现：通过 `solaceai.backends` 命名空间，在独立安装
   `solaceai-online` 或 `solaceai-offline` 分发包后自动解析并实例化对应后端；
3. 统一契约接口 `Backend`：强制要求所有具体后端实现向量存储构建、多路混合检索执行、
   心理危机风险评估、知识库索引构建与切片版本增删改查生命周期管理。
"""
from functools import lru_cache
from importlib.metadata import entry_points
from typing import Protocol, Any


class Backend(Protocol):
    """SolaceAI 运行时后端契约协议 (Backend SPI Protocol)

    规范所有具体后端（如云端轻量 OnlineBackend、纯离线高安全 OfflineBackend）必须履行的业务接口。
    """

    def vector_store(self, settings: Any) -> Any:
        """根据系统配置构建并返回对应后端的向量数据库存储驱动实例。

        Args:
            settings: 运行时系统配置对象。

        Returns:
            Any: 具备 `upsert_chunks`, `delete_chunks`, `rebuild` 等接口的向量库适配器对象。
        """
        ...

    def retrieval_engine(self, settings: Any) -> Any:
        """构建并返回当前后端的多路召回与重排检索执行引擎实例。

        Args:
            settings: 运行时系统配置对象。

        Returns:
            Any: 具备 `get_recall_lanes` 等接口的多路检索引擎实例。
        """
        ...

    def assess(
        self,
        settings: Any,
        safety_client: Any,
        text: str,
        history: Any,
        emotion_client: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """执行端到端心理状态研判与危机风险评估。

        Args:
            settings: 运行时系统配置对象。
            safety_client: 安全大模型推理客户端。
            text: 用户本轮输入的咨询文本。
            history: 多轮历史对话上下文。
            emotion_client: 辅助情绪分类客户端。

        Returns:
            tuple[Any, dict[str, Any]]: (心理评估结果对象 PsychologyAssessment, 观测指标与溯源元数据字典)。
        """
        ...

    def prepare_index(self, service: Any) -> int:
        """在服务启动或配置就绪时预热和补全知识切片向量索引。

        Args:
            service: 知识库服务实例 (KnowledgeService)。

        Returns:
            int: 本次预处理或补全索引的切片总数量。
        """
        ...

    def rebuild(self, service: Any, rows: list[Any]) -> int:
        """清空已有向量集合并针对全量切片全量重建向量索引。

        Args:
            service: 知识库服务实例 (KnowledgeService)。
            rows: 待全量建立索引的切片实体列表。

        Returns:
            int: 成功重建索引的切片总数。
        """
        ...

    def index(self, service: Any, rows: list[Any]) -> None:
        """对增量或更新的切片批量执行向量化并写入索引库。

        Args:
            service: 知识库服务实例 (KnowledgeService)。
            rows: 待写入向量索引库的切片实体列表。
        """
        ...

    def delete(self, service: Any, ids: list[str]) -> None:
        """从底层向量数据库中硬删除指定的向量切片。

        Args:
            service: 知识库服务实例 (KnowledgeService)。
            ids: 待移除的切片唯一外部标识列表。
        """
        ...


@lru_cache
def backend(name: str) -> Backend:
    """根据分发包 entry_points 动态加载并缓存后端实现实例 (Load Backend by Entry Point)

    通过 Python 标准包元数据扫描 `solaceai.backends` 组下的扩展点，
    确保环境中严格安装且仅安装了一个目标后端实现，避免多重安装发生冲突。

    Args:
        name (str): 后端标识名称（通常为 'online' 或 'offline'）。

    Returns:
        Backend: 实现了 `Backend` 协议的后端单例对象。

    Raises:
        RuntimeError: 当未安装对应包或发现多个冲突的同名后端时抛出。
    """
    matches = list(entry_points(group='solaceai.backends', name=name))
    if len(matches) != 1:
        raise RuntimeError(f'Install exactly one solaceai-{name} package (found {len(matches)})')
    return matches[0].load()()


def knowledge_backend(settings: Any) -> Backend:
    """根据系统配置的知识库后端类型，自动解析并返回对应的后端实例 (Resolve Knowledge Backend)

    Args:
        settings: 系统配置实例，用于读取 `knowledge_backend` 配置项。

    Returns:
        Backend: 对应的 Online 或 Offline 后端实例。
    """
    return backend('offline' if getattr(settings, 'knowledge_backend', 'chroma_bm25') == 'milvus_bge' else 'online')
