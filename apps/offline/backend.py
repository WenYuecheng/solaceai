"""SolaceAI 线下高安全运行模式后端实现 (Offline Backend SPI Implementation)

本模块作为系统在私有化、纯内网及高安全高校环境中运行的核心装配器：
1. 向量存储层：采用 Milvus 向量数据库与本地 BGE-M3 密集/稀疏联合索引 (`MilvusBGEStore`)，
   彻底规避 Chroma 与外部云端 Embedding API 依赖；
2. 检索重排层：采用本地 Cross-Encoder 语义精排与多路倒数排名融合 (`OfflineRetrievalEngine`)；
3. 心理评估层：纯内网本地微调大模型 (Qwen3-4B 双 LoRA / MLX Metal 原生加速) 辅助研判与情绪特征观测 (`OfflineAssessmentService`)；
4. 严格隐私边界：物理阻断任何公网付费大模型调用，确保学生隐私数据不出校园网物理机房。
"""
from __future__ import annotations

import hashlib
from typing import Any

from .assessment import OfflineAssessmentService
from .retrieval import OfflineRetrievalEngine
from .emotion_classifier import observe_emotion


class DisabledVectorStore:
    """离线模式占位禁用向量存储类 (Disabled Chroma Vector Store Stub)

    在线下 Profile 下，Chroma 依赖被物理剔除，该占位类确保在意外调用在线向量接口时安全阻断。

    Attributes:
        can_embed (bool): 恒定为 False，标识在线嵌入不可用。
        error (str): 错误描述字符串。
    """
    can_embed: bool = False
    error: str = ''


class OfflineBackend:
    """线下高安全模式运行时后端驱动 (Offline High-Security Runtime Backend)

    实现 `app.core.backends.Backend` 接口协议，全流程装配 Milvus、本地 BGE 与离线评估组件。
    """

    OfflineAssessmentService = OfflineAssessmentService
    observe_emotion = staticmethod(observe_emotion)

    def vector_store(self, settings: Any) -> DisabledVectorStore:
        """在线下模式下返回禁用的 Chroma 占位对象（线下模式强制使用 Milvus）。

        Args:
            settings: 系统配置实例。

        Returns:
            DisabledVectorStore: 占位禁用对象。
        """
        return DisabledVectorStore()

    def retrieval_engine(self, settings: Any) -> OfflineRetrievalEngine:
        """构建并返回当前线下环境专用的多路神经混合检索引擎。

        Args:
            settings: 系统配置实例。

        Returns:
            OfflineRetrievalEngine: 线下多路检索引擎实例。
        """
        return OfflineRetrievalEngine(settings)

    def assess(
        self,
        settings: Any,
        safety_client: Any,
        text: str,
        history: Any,
        emotion_client: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """委托线下评估服务完成纯本地心理状态研判与情绪分类观测。

        Args:
            settings: 系统配置实例。
            safety_client: 安全审核客户端。
            text (str): 学生输入咨询文本。
            history: 历史对话记录。
            emotion_client: 本地微调情绪分类客户端。

        Returns:
            tuple[Any, dict[str, Any]]: (心理评估实体, 观测指标字典)。
        """
        return OfflineAssessmentService(settings, emotion_client).assess(text, history, emotion_client)

    def milvus_store(self, service: Any) -> Any:
        """根据租户与知识库标识计算集合名称，构建并返回 Milvus 存储驱动实例 (Get Milvus Store)

        通过 SHA-256 计算多租户集合隔离后缀，连接本地或私有部署的 Milvus 服务。

        Args:
            service: 知识库服务实例 (KnowledgeService)。

        Returns:
            MilvusBGEStore: 绑定的 Milvus 向量存储适配器。
        """
        from .milvus_bge import get_milvus_store
        s = service.settings
        scope = (
            '' if (service.tenant_id, service.knowledge_base_id) == ('default', 'default')
            else '_' + hashlib.sha256(f'{service.tenant_id}:{service.knowledge_base_id}'.encode()).hexdigest()[:12]
        )
        return get_milvus_store(
            s.milvus_uri,
            s.milvus_token,
            s.milvus_collection + scope,
            s.local_rag_url,
            s.local_rag_timeout_seconds,
        )

    def prepare_index(self, service: Any) -> int:
        """预处理并确保 Milvus 密集与稀疏向量索引构建完成 (Prepare Milvus Index)

        Args:
            service: 知识库服务实例。

        Returns:
            int: 知识切片总数量。
        """
        rows = service._all_chunks()
        store = service._milvus_store()
        store.ensure_index(rows, service._chunk_metadata)
        return len(rows)

    def rebuild(self, service: Any, rows: list[Any]) -> int:
        """清空 Milvus 集合并全量重新计算本地向量并建索引 (Rebuild Milvus Index)

        Args:
            service: 知识库服务实例。
            rows (list[Any]): 待重建索引的切片 ORM 实体列表。

        Returns:
            int: 成功重建的切片数量。

        Raises:
            RuntimeError: 当知识向量库未启用时抛出。
        """
        if not service.settings.knowledge_vector_enabled:
            raise RuntimeError('Milvus向量库未启用')
        return service._milvus_store().rebuild(rows, metadata=service._chunk_metadata)

    def index(self, service: Any, rows: list[Any]) -> None:
        """将增量切片送入本地 BGE 推理模型提取向量并写入 Milvus (Index Chunks via Local BGE)

        Args:
            service: 知识库服务实例。
            rows (list[Any]): 待增量插入的切片实体列表。
        """
        if service.settings.knowledge_vector_enabled:
            service._milvus_store().upsert_chunks(rows, metadata=service._chunk_metadata)

    def delete(self, service: Any, ids: list[str]) -> None:
        """从 Milvus 集合中根据外部 ID 物理删除向量记录 (Delete Chunks from Milvus)

        Args:
            service: 知识库服务实例。
            ids (list[str]): 待删除的切片外部标识列表。
        """
        if service.settings.knowledge_vector_enabled:
            service._milvus_store().delete_chunks(ids)

    def local_reranker(self, settings: Any) -> Any:
        """构建本地 Cross-Encoder 语义精排器实例。

        Args:
            settings: 系统配置实例。

        Returns:
            Any: 本地语义精排器。
        """
        return self.retrieval_engine(settings).build_reranker()

    def vector_status(self, service: Any) -> tuple[bool, str, int | None]:
        """探查本地 RAG 推理 Worker 与 Milvus 集合的就绪状态及记录行数 (Probe Milvus & Worker Status)

        执行逻辑：
        1. 检查配置是否启用向量库；
        2. 发起 HTTP 健康探测请求至 `local_rag_url/health`；
        3. 连通 Milvus 并查询对应 collection 的实时行数统计；
        4. 异常时安全捕获并返回不可用标识。

        Args:
            service: 知识库服务实例。

        Returns:
            tuple[bool, str, int | None]: (可用性标识, 错误描述, 集合记录总数)。
        """
        if not service.settings.knowledge_vector_enabled:
            return False, '', None
        try:
            import httpx
            health = httpx.get(service.settings.local_rag_url + '/health', timeout=3)
            health.raise_for_status()
            store = service._milvus_store()
            store.client.list_collections()
            count = int(store.client.get_collection_stats(collection_name=store.collection_name)['row_count']) if store.collection_name else None
            return True, '', count
        except Exception as exc:
            return False, type(exc).__name__, None
