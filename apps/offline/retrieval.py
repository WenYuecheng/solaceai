"""SolaceAI 线下神经检索与重排套件 (Offline Neural Retrieval & Cross-Encoder Suite)

专职服务本地离线环境：
1. BGE-M3 密集向量 (Dense 1024d) + 词元级学习型稀疏特征 (Learned Sparse Lexical)；
2. Milvus 分布式/本地向量存储与双路召回；
3. 服务端/客户端 RRF (Reciprocal Rank Fusion) 倒数排名融合；
4. 本地 Cross-Encoder (bge-reranker-large) 深度交互语义重排序。
"""

from __future__ import annotations

from typing import Any, Callable

from app.core.config import Settings
from app.services.retrieval import (
    LaneRetriever,
    LocalCrossEncoderReranker,
    RRFFusion,
)


class OfflineRetrievalEngine:
    """线下混合检索执行引擎 (Offline Retrieval Engine)

    调度本地深度学习特征推理服务与 Milvus 向量集合，执行密集+稀疏双路召回与重排。
    """

    def __init__(self, settings: Settings):
        """初始化线下检索引擎。

        Args:
            settings (Settings): 系统配置实例。
        """
        self.settings = settings

    def get_recall_lanes(
        self,
        service: Any,
        query: str,
        recall_k: int,
        chunks: list[Any],
        reasons: list[str],
    ) -> tuple[dict[str, Any], set[str], bool]:
        """获取线下 Milvus + BGE-M3 检索通道字典与参与融合的通道集 (Get Offline Recall Lanes)

        执行逻辑：
        1. 尝试调用 `service._retrieve_bge` 发起本地 BGE-M3 向量嵌入与 Milvus 多路召回；
        2. 若 Milvus 结果为空或出现调用异常，记录原因并自动优雅降级为本地内存 BM25 词法检索通道；
        3. 返回通道字典、参与融合的通道标识集合以及 BGE 是否调用成功的布尔标志。

        Args:
            service (Any): 知识库服务实例 (KnowledgeService)。
            query (str): 检索查询文本。
            recall_k (int): 单通道召回 Top-K 数量。
            chunks (list[Any]): 当前租户与知识库范围内的全量可用切片列表。
            reasons (list[str]): 检索降级原因记录追踪列表。

        Returns:
            tuple[dict[str, Any], set[str], bool]: (通道字典, 参与融合的通道集合, BGE是否执行成功标志)。
        """
        bge_ok = False
        try:
            lanes = service._retrieve_bge(query, recall_k, chunks)
            bge_ok = True
            fusion_lanes = {"milvus_fused"}
            if chunks and not lanes.get('milvus_fused'):
                reasons.append('milvus_no_results')
                lanes['bm25'] = lambda q, k: service._retrieve_bm25(q, k, chunks)
                fusion_lanes = {'bm25'}
        except Exception as exc:
            reasons.append("milvus_bge_failed:" + type(exc).__name__)
            lanes = {"bm25": lambda q, k: service._retrieve_bm25(q, k, chunks)}
            fusion_lanes = {"bm25"}
        return lanes, fusion_lanes, bge_ok

    def build_reranker(self, local_rag_client: Any = None) -> Any:
        """构建线下本地 CrossEncoder 语义精排器 (Build Local Semantic Reranker)

        若未显式注入客户端，自动初始化 `LocalBGEClient` 连接本地 Worker 微服务。

        Args:
            local_rag_client (Any, optional): 本地 RAG HTTP 客户端实例，默认为 None。

        Returns:
            LocalCrossEncoderReranker: 深度学习交叉注意力精排适配器。
        """
        if local_rag_client is not None:
            return LocalCrossEncoderReranker(local_rag_client)
        from apps.offline.local_rag import LocalBGEClient
        client = LocalBGEClient(
            self.settings.local_rag_url, self.settings.local_rag_timeout_seconds
        )
        return LocalCrossEncoderReranker(client)
