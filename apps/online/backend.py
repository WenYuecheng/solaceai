"""SolaceAI 云端轻量运行模式后端实现 (Online Backend SPI Implementation)

本模块作为系统在云端低内存容器环境中运行的核心装配器：
1. 向量存储层：采用基于 SQLite 嵌入式文件的 Chroma 向量库 (`ChromaKnowledgeStore`)，
   无需常驻外部 Milvus 分布式集群；
2. 检索重排层：采用在线 API 嵌入特征 + 本地 BM25 倒数排名融合 (RRF) 检索引擎 (`OnlineRetrievalEngine`)；
3. 心理评估层：由云端大语言模型 API 驱动的结构化评估服务 (`OnlineAssessmentService`)；
4. 资源轻量化约束：严格禁止在在线模式下拉起本地 Milvus 或 CrossEncoder 重排器，规避高内存消耗。
"""
from __future__ import annotations

import json
from typing import Any

from .vector_store import ChromaKnowledgeStore
from .assessment import OnlineAssessmentService, PsychologicalAssessmentService
from .retrieval import OnlineRetrievalEngine


class OnlineBackend:
    """云端轻量模式运行时后端驱动 (Online Lightweight Runtime Backend)

    实现 `app.core.backends.Backend` 接口协议，为应用提供无依赖外部重型中间件的独立服务装配。
    """

    OnlineAssessmentService = OnlineAssessmentService
    PsychologicalAssessmentService = PsychologicalAssessmentService

    def vector_store(self, settings: Any) -> ChromaKnowledgeStore:
        """构建并返回当前在线环境专用的 Chroma 嵌入式向量库实例。

        Args:
            settings: 系统配置实例。

        Returns:
            ChromaKnowledgeStore: Chroma 向量存储适配器。
        """
        return ChromaKnowledgeStore(settings)

    def retrieval_engine(self, settings: Any) -> OnlineRetrievalEngine:
        """构建并返回当前在线环境专用的混合检索执行引擎。

        Args:
            settings: 系统配置实例。

        Returns:
            OnlineRetrievalEngine: 在线轻量混合检索执行器。
        """
        return OnlineRetrievalEngine(settings)

    def assess(
        self,
        settings: Any,
        safety_client: Any,
        text: str,
        history: Any,
        emotion_client: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """委托云端评估服务完成心理状态研判与危机风险评估。

        Args:
            settings: 系统配置实例。
            safety_client: 安全审核模型推理客户端。
            text (str): 学生输入咨询文本。
            history: 历史对话记录。
            emotion_client: 辅助情绪分类客户端。

        Returns:
            tuple[Any, dict[str, Any]]: (心理评估实体, 观测及来源字典)。
        """
        return OnlineAssessmentService(settings, safety_client).assess(text, history, emotion_client)

    def prepare_index(self, service: Any) -> int:
        """检查并补全 Chroma 向量库中缺失的切片向量索引 (Prepare Chroma Index)

        比对关系统计表与 Chroma 集合中的现有 ID，对尚未向量化的历史切片执行批量嵌入并同步提交。

        Args:
            service: 知识库服务实例 (KnowledgeService)。

        Returns:
            int: 本次增量写入 Chroma 的切片数量。

        Raises:
            RuntimeError: 当在线向量嵌入服务不可用时抛出。
        """
        if not service.vector_store.can_embed:
            raise RuntimeError(service.vector_store.error or 'Vector store unavailable')
        current = set(service.vector_store.collection.get().get('ids', []))
        missing = [row for row in service._all_chunks() if service.vector_store._id(row.id) not in current]
        if missing:
            service._index_vector_chunks(missing)
            service.db.commit()
        return len(missing)

    def rebuild(self, service: Any, rows: list[Any]) -> int:
        """全量重新计算切片嵌入向量并重置重建 Chroma 向量集合 (Rebuild Chroma Index)

        Args:
            service: 知识库服务实例。
            rows (list[Any]): 待全量重建索引的切片 ORM 实体列表。

        Returns:
            int: 成功重建索引的切片总数。

        Raises:
            RuntimeError: 当向量服务未就绪时抛出。
        """
        if not service.vector_store.can_embed:
            raise RuntimeError(getattr(service.vector_store, 'error', '') or 'Chroma向量库不可用')
        for row in rows:
            row.embedding_json = None
        embeddings = service._embeddings_for_chunks(rows)
        for row, embedding in zip(rows, embeddings, strict=True):
            row.embedding_json = json.dumps(embedding, separators=(',', ':'))
        service.vector_store.rebuild(rows, embeddings)
        service.db.commit()
        return len(rows)

    def index(self, service: Any, rows: list[Any]) -> None:
        """为增量摄入的切片生成在线向量并写入 Chroma 向量库 (Index Chunks Incrementally)

        Args:
            service: 知识库服务实例。
            rows (list[Any]): 增量切片实体列表。

        Raises:
            ValueError: 向量维度或数量与切片数不匹配。
            RuntimeError: 向量服务不可用且启用了向量检索。
        """
        if service.vector_store.can_embed:
            embeddings = service._embeddings_for_chunks(rows)
            if len(embeddings) != len(rows):
                raise ValueError('Embedding数量与切片不一致')
            for row, embedding in zip(rows, embeddings, strict=True):
                row.embedding_json = json.dumps(embedding)
            service.vector_store.upsert_chunks(rows, embeddings)
        elif service.settings.knowledge_vector_enabled:
            raise RuntimeError('向量服务不可用；版本保留待重试')

    def delete(self, service: Any, ids: list[str]) -> None:
        """从 Chroma 向量库中物理移除指定切片的向量记录 (Delete Chunks from Chroma)

        Args:
            service: 知识库服务实例。
            ids (list[str]): 切片唯一标识列表。
        """
        if service.vector_store.can_embed:
            service.vector_store.delete_chunks(ids)

    def milvus_store(self, service: Any) -> Any:
        """在线 Profile 禁止调用 Milvus，防范环境误配。"""
        raise RuntimeError('Milvus is unavailable in online profile')

    def local_reranker(self, settings: Any) -> Any:
        """在线 Profile 禁止加载本地 CrossEncoder 重排模型，规避内存溢出。"""
        raise ValueError('Local reranker is unavailable in online profile')

    def vector_status(self, service: Any) -> tuple[bool, str, int | None]:
        """探查当前在线 Chroma 向量存储的健康状态与记录总数 (Probe Chroma Status)

        Args:
            service: 知识库服务实例。

        Returns:
            tuple[bool, str, int | None]: (可用性标识, 错误描述, 向量总条数)。
        """
        available = service.vector_store.can_embed
        error = getattr(service.vector_store, 'error', '')
        count = None
        if available:
            try:
                count = service.vector_store.count()
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        return available, error, count
