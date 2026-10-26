"""SolaceAI Milvus 与 BGE-M3 混合向量检索知识库存储引擎 (Milvus BGE-M3 Hybrid Vector Store)

本模块作为系统面向大规模分布式环境的向量检索存储引擎，深度整合 Milvus 与 BGE-M3：
1. 双路混合特征索引 Schema：
   - dense: 1024 维 BGE-M3 稠密向量，基于 FLAT 索引与 COSINE 余弦度量空间；
   - sparse: BGE-M3 学习型词元级稀疏特征，基于 SPARSE_INVERTED_INDEX 与内积 (IP) 空间；
2. 服务端倒数排名融合 (RRF)：直接在 Milvus 集群服务端执行 `hybrid_search` 与 `RRFRanker(60)`，
   避免多路结果在应用端传输产生额外反序列化与内存开销；
3. 模型权重与版本哈希隔离：集合名自动绑定 BGE-M3 权重哈希指纹（`{prefix}_v2_{digest}`），
   模型权重升级时自动创建新集合，杜绝不同权重特征混合导致检索退化；
4. 并发一致性与游标查询：结合全局可重入锁 `_lock` 与 `query_iterator`，防范并发重复建表与游标截断。
"""

from __future__ import annotations

import hashlib
import re
import threading
from functools import lru_cache
from typing import Any, Callable

from apps.offline.local_rag import LocalBGEClient

# 全局可重入线程锁，防止多线程并发检查或重复建表时的竞态冲突
_lock = threading.RLock()


class MilvusBGEStore:
    """基于 Milvus 向量引擎与 BGE-M3 混合特征的检索存储核心类 (Milvus BGE-M3 Store)

    负责切片的混合特征提取入库、集合动态版本维护以及服务端 RRF 多路召回。
    """

    def __init__(
        self,
        uri: str,
        token: str,
        collection: str,
        model_url: str,
        timeout: int | float = 180,
        client: Any = None,
        encoder: LocalBGEClient | None = None,
    ):
        """初始化 Milvus BGE 存储引擎

        Args:
            uri (str): Milvus 远程服务连接 URI (例如 'http://127.0.0.1:19530')。
            token (str): Milvus 鉴权 Token（若未启用可留空）。
            collection (str): 集合名称前缀标识符。
            model_url (str): 本地 BGE 推理 Worker 的服务访问地址。
            timeout (int | float, optional): 客户端交互超时阈值（秒），默认 180。
            client (Any, optional): 注入的预连接 MilvusClient 实例（便于测试 Mock）。
            encoder (LocalBGEClient | None, optional): 注入的特征编码器客户端实例。

        Raises:
            ValueError: 当集合名称前缀包含非法字符时不符合 Milvus 命名规范。
        """
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,100}', collection):
            raise ValueError('Invalid Milvus collection prefix')
        if client is None:
            from pymilvus import MilvusClient
            client = MilvusClient(uri=uri, token=token, timeout=30)
        self.client = client
        self.encoder = encoder or LocalBGEClient(model_url, timeout)
        self.prefix = collection
        self.collection_name = ''

    def _ensure_collection(self) -> str:
        """确保绑定当前 BGE-M3 静态模型指纹的专属版本集合已就绪 (Ensure Versioned Collection)

        若集合不存在，则按强一致性级别 (consistency_level='Strong') 创建包含 dense 与 sparse 的复合 Schema。

        Returns:
            str: 格式为 `{prefix}_v2_{digest}` 的物理集合名称。
        """
        if not self.collection_name:
            digest = hashlib.sha256(self.encoder.identity().encode()).hexdigest()[:16]
            self.collection_name = f'{self.prefix}_v2_{digest}'
        name = self.collection_name
        if self.client.has_collection(collection_name=name):
            return name

        from pymilvus import DataType

        # 构建复合向量 Schema
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 构建稠密与稀疏双索引
        indexes = self.client.prepare_index_params()
        indexes.add_index(field_name='dense', index_type='FLAT', metric_type='COSINE')
        indexes.add_index(field_name='sparse', index_type='SPARSE_INVERTED_INDEX', metric_type='IP')

        self.client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=indexes,
            consistency_level='Strong',
        )
        return name

    def _ids(self) -> set[int]:
        """流式迭代获取集合中已持久化的全量主键 ID 集合 (Iterate All Document IDs)

        使用 `query_iterator` 规避服务器端默认 16384 条拉取上限，不依赖粗略的 row_count。

        Returns:
            set[int]: 当前集合中包含的所有整型主键 ID 集合。
        """
        iterator = self.client.query_iterator(
            collection_name=self.collection_name,
            batch_size=1000,
            filter='',
            output_fields=['id'],
        )
        ids = set()
        try:
            while batch := iterator.next():
                ids.update(int(row['id']) for row in batch)
        finally:
            iterator.close()
        return ids

    def upsert_chunks(
        self,
        chunks: list[Any],
        embeddings=None,
        *,
        metadata: Callable[[Any], dict] | None = None,
    ) -> int:
        """向 Milvus 集合批量计算特征并写入切片向量 (Batch Upsert Knowledge Chunks)

        将标题大纲路径面包屑拼接至正文前执行联合特征计算，保持长文本分层上下文。

        Args:
            chunks (list[Any]): 知识切片实体列表。
            embeddings (optional): 预置特征向量（若未提供则调用 BGE-M3 实时特征提取）。
            metadata (Callable[[Any], dict] | None, optional): 提取元数据的回调函数。

        Returns:
            int: 成功写入的切片数量。

        Raises:
            ValueError: 当生成的特征数量与切片数不一致时抛出。
        """
        metadata = metadata or (lambda row: {})
        with _lock:
            name = self._ensure_collection()
            for start in range(0, len(chunks), 8):
                batch = chunks[start:start + 8]
                # 拼接标题面包屑以增强切片的独立语义
                texts = [' > '.join(metadata(c).get('section_path', [])) + '\n' + c.content for c in batch]
                dense, sparse = self.encoder.embed(texts)
                if len(batch) != len(dense) or len(batch) != len(sparse):
                    raise ValueError('Embedding数量与切片不一致')
                self.client.upsert(
                    collection_name=name,
                    data=[
                        {'id': c.id, 'dense': d, 'sparse': sparse[i]}
                        for i, (c, d) in enumerate(zip(batch, dense, strict=True))
                    ],
                )
            self.client.flush(collection_name=name)
            return len(chunks)

    def delete_chunks(self, ids: list[int]) -> None:
        """分批从 Milvus 集合中物理删除指定主键切片向量 (Batch Delete Chunks)

        Args:
            ids (list[int]): 待删除的切片主键 ID 列表。
        """
        if not ids:
            return
        with _lock:
            name = self._ensure_collection()
            for start in range(0, len(ids), 1000):
                self.client.delete(collection_name=name, ids=ids[start:start + 1000])
            self.client.flush(collection_name=name)

    def rebuild(
        self,
        chunks: list[Any],
        embeddings=None,
        *,
        metadata: Callable[[Any], dict] | None = None,
    ) -> int:
        """全量重建向量索引并清理历史孤儿向量 (Rebuild Vector Index with Orphan Cleanup)

        Args:
            chunks (list[Any]): 关系数据库当前有效的切片列表。
            embeddings (optional): 预置特征。
            metadata (Callable[[Any], dict] | None, optional): 元数据提取函数。

        Returns:
            int: 成功重建的切片数量。
        """
        with _lock:
            count = self.upsert_chunks(chunks, metadata=metadata)
            self.delete_chunks(sorted(self._ids() - {c.id for c in chunks}))
            return count

    def ensure_index(self, chunks: list[Any], metadata: Callable[[Any], dict]) -> str:
        """查询前检查并增量补全缺失的切片索引 (Ensure Index Ready for Query)

        仅增量补录缺失切片，不删除处于 PENDING 状态的新切片，保证高并发导入安全。

        Args:
            chunks (list[Any]): 预期索引的切片列表。
            metadata (Callable[[Any], dict]): 元数据提取函数。

        Returns:
            str: 处于就绪状态的物理集合名称。
        """
        with _lock:
            name = self._ensure_collection()
            missing = {c.id for c in chunks} - self._ids()
            if missing:
                self.upsert_chunks([c for c in chunks if c.id in missing], metadata=metadata)
            return name

    def legacy_collections(self) -> list[str]:
        """列出旧模型内容哈希历史遗留集合 (List Legacy Version Collections)

        供运维工具核查并确认无其他运行中服务依赖后，安全回收旧物理集合。

        Returns:
            list[str]: 匹配前缀的历史版本集合名称列表。
        """
        pattern = re.compile(re.escape(self.prefix) + r'_[a-f0-9]{16}')
        return [name for name in self.client.list_collections() if pattern.fullmatch(name)]

    def cleanup_legacy_collections(self, names: list[str]) -> int:
        """安全物理删除指定的旧模型集合 (Drop Legacy Collections)

        Args:
            names (list[str]): 待清理的历史集合名称列表。

        Returns:
            int: 成功删除的集合数量。

        Raises:
            ValueError: 试图删除不属于当前集合前缀作用域的集合时抛出。
        """
        allowed = set(self.legacy_collections())
        if any(name not in allowed for name in names):
            raise ValueError('只能清理当前作用域的旧哈希集合')
        for name in names:
            self.client.drop_collection(collection_name=name)
        return len(names)

    def recall(
        self,
        query: str,
        k: int,
        chunks: list[Any],
        metadata: Callable[[Any], dict],
    ) -> dict[str, list[Any]]:
        """对用户检索词执行密集、稀疏与服务端 RRF 混合融合的多路召回 (Multi-Lane Recall & Fusion)

        调用流程：
        1. 读取已准备的集合索引，不在查询中批量建库；
        2. 生成 query 的 dense 向量与 sparse 稀疏权重；
        3. 分别发起 dense 向量检索通道与 sparse 倒排索引检索通道；
        4. 构造 `AnnSearchRequest` 发起 `hybrid_search`，由 Milvus 服务端通过 RRFRanker(60)
           加权计算并返回最优候选集；
        5. 将各路命中映射回结构化 `SearchResult` 实体列表。

        Args:
            query (str): 用户检索提问。
            k (int): 各单路期望召回切片数量。
            chunks (list[Any]): 活跃知识切片全集。
            metadata (Callable[[Any], dict]): 元数据提取函数。

        Returns:
            dict[str, list[SearchResult]]: 包含 'dense'、'sparse' 与 'milvus_fused' 三个通道的结果字典。
        """
        from pymilvus import AnnSearchRequest, RRFRanker
        from app.services.knowledge import SearchResult

        if not chunks:
            return {'dense': [], 'sparse': [], 'milvus_fused': []}

        # 知识库全量向量化应在显式预处理或导入时完成，绝不在实时聊天请求中执行耗时建库
        name = self._ensure_collection()
        if not self._ids().intersection(c.id for c in chunks):
            return {'dense': [], 'sparse': [], 'milvus_fused': []}
        dense, sparse = self.encoder.embed([query])
        active_filter = 'id in [' + ','.join(str(int(c.id)) for c in chunks) + ']'
        dense_params = {'metric_type': 'COSINE', 'params': {}}
        sparse_params = {'metric_type': 'IP', 'params': {}}

        # 执行单路检索并分别记录轨迹
        lanes = {}
        for field, data, params in [('dense', dense, dense_params), ('sparse', sparse, sparse_params)]:
            lanes[field] = self.client.search(
                collection_name=name,
                data=data,
                anns_field=field,
                search_params=params,
                limit=k,
                filter=active_filter,
            )[0]

        # 构造双路融合请求，由服务端基于 RRF 算法加权融合
        requests = [
            AnnSearchRequest(data=dense, anns_field='dense', param=dense_params, limit=k, expr=active_filter),
            AnnSearchRequest(data=sparse, anns_field='sparse', param=sparse_params, limit=k, expr=active_filter),
        ]
        lanes['milvus_fused'] = self.client.hybrid_search(
            collection_name=name,
            reqs=requests,
            ranker=RRFRanker(60),
            limit=min(30, k * 2),
        )[0]

        # 映射回完整的领域实体对象
        by_id = {c.id: c for c in chunks}
        return {
            lane: [
                SearchResult(
                    by_id[h['id']].id,
                    by_id[h['id']].source,
                    by_id[h['id']].content,
                    float(h['distance']),
                )
                for h in hits
                if h['id'] in by_id
            ]
            for lane, hits in lanes.items()
        }


@lru_cache(maxsize=8)
def get_milvus_store(
    uri: str,
    token: str,
    collection: str,
    model_url: str,
    timeout: int | float,
) -> MilvusBGEStore:
    """获取全局复用的 MilvusBGEStore 单例实例（基于 LRU 缓存）(Cached Milvus Store Singleton)

    Args:
        uri (str): Milvus 服务连接 URI。
        token (str): 鉴权 Token。
        collection (str): 集合前缀。
        model_url (str): 模型推理 Worker 地址。
        timeout (int | float): 超时秒数。

    Returns:
        MilvusBGEStore: 全局单例对象。
    """
    return MilvusBGEStore(uri, token, collection, model_url, timeout)
