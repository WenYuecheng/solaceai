"""SolaceAI 线上轻量混合检索套件 (Online Lightweight Hybrid Retrieval Suite).

专职服务云端轻量化与低内存容器部署：
1. Chroma 嵌入式向量引擎 + 在线 Embedding API (text-embedding-3-small, 1024d)；
2. 本地内存 BM25 中文分词字面检索；
3. 客户端标准 RRF (Reciprocal Rank Fusion) 倒数排名融合；
4. 轻量级词面精排或 API Reranker 适配器。
"""

from __future__ import annotations

from typing import Any

from app.core.config import Settings
from app.services.retrieval import (
    LaneRetriever,
    RRFFusion,
    SemanticReranker,
)


class OnlineRetrievalEngine:
    """线上轻量混合检索执行引擎 (Online Retrieval Engine)."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def get_recall_lanes(
        self,
        service: Any,
        query: str,
        recall_k: int,
        chunks: list[Any],
        reasons: list[str],
    ) -> tuple[dict[str, Any], set[str], bool]:
        """获取线上 Chroma 密集向量 + BM25 词法通道字典与融合集合。

        Returns:
            tuple: (lanes_dict, fusion_lanes_set, is_ok_flag)
        """
        lanes = {
            "dense": lambda q, k: (
                service._retrieve_vector(q, k)
                if service.vector_store.can_embed
                else []
            ),
            "bm25": lambda q, k: service._retrieve_bm25(q, k, chunks),
        }
        fusion_lanes = {"dense", "bm25"}
        return lanes, fusion_lanes, True

    def build_reranker(self) -> Any:
        """构建线上重排器（API 或词面重排）。"""
        return SemanticReranker(self.settings)
