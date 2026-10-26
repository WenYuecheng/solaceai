"""SolaceAI 本地深度学习模型服务进程 (Isolated Local BGE & Cross-Encoder FastAPI Worker)

本模块作为独立运行的微服务进程，专职承载计算密集型的深度学习语义检索模型推理：
1. 进程级资源隔离：将 HuggingFace / PyTorch 权重加载与推理隔离在独立子进程，
   规避主 Web 服务进程发生内存碎片堆积（Memory Bloat）与 Python GIL 锁冲突；
2. 双模型统一托管：
   - BGE-M3 (FlagEmbedding): 密集向量 (Dense Vector) 与词元级稀疏权重 (Sparse Lexical Weights) 联合特征提取；
   - BGE-Reranker-Large (CrossEncoder): 基于全注意力的 (Query, Document) 交互交叉精排打分；
3. 确定性指纹鉴伪：通过扫描权重文件大小、修改时间与配置文本计算 SHA-256 静态特征哈希；
4. 并发与内存安全：结合线程互斥锁 (threading.Lock) 与最大批处理尺寸硬限制，确保 CPU/GPU 推理不爆内存。
"""

import argparse
import hashlib
import math
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def create_worker(
    embedding_path: str,
    reranker_path: str,
    device: str = 'cpu',
) -> FastAPI:
    """构建并配置本地 RAG 模型的 FastAPI 路由应用 (Local RAG Worker Application Factory)

    延迟按需加载 PyTorch 权重，注入线程锁控制并发推理，挂载健康探测与推理接口。

    Args:
        embedding_path (str): 本地 BGE-M3 嵌入模型权重的存放目录。
        reranker_path (str): 本地 BGE-Reranker-Large 交叉重排模型权重的存放目录。
        device (str, optional): 推理硬件计算设备标识 ('cpu', 'cuda', 'mps')，默认为 'cpu'。

    Returns:
        FastAPI: 承载模型特征提取与重排服务的 FastAPI 实例。
    """
    import torch
    # 限制 PyTorch 内部 CPU 线程数，防止高并发打满主机全部 CPU 核心造成系统响应停顿
    torch.set_num_threads(4)

    # 线程互斥锁，保证多请求并发时底层 PyTorch 推理串行执行，避免显存/内存竞争
    lock = threading.Lock()
    # 延迟加载缓存字典：{'embedding': BGEM3FlagModel, 'reranker': CrossEncoder}
    loaded: dict[str, object] = {}

    # 计算本地模型权重的静态内容指纹（包括文件大小与纳秒时间戳），供健康探针校验版本一致性
    files = sorted([*Path(embedding_path).glob('*.bin'), *Path(embedding_path).glob('*.safetensors')])
    config_file = Path(embedding_path) / 'config.json'
    config_text = config_file.read_text(encoding='utf-8') if config_file.is_file() else ''
    identity_payload = (
        str(Path(embedding_path).resolve())
        + config_text
        + str([(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files])
    )
    identity = hashlib.sha256(identity_payload.encode()).hexdigest()

    class Texts(BaseModel):
        """批量文本特征提取请求体数据契约 (Batch Texts Request DTO)"""
        texts: list[str] = Field(
            ...,
            min_length=1,
            max_length=32,
            description='待批量处理的文本列表，受最大批次 32 限制以保护显存/内存',
        )

    class Ranking(Texts):
        """语义重排序请求体数据契约 (Semantic Ranking Request DTO)"""
        query: str = Field(
            ...,
            min_length=1,
            max_length=2000,
            description='用户检索的查询词，最长不超过 2000 字符',
        )

    app = FastAPI(title='SolaceAI Local BGE')

    def validate(texts: list[str]) -> None:
        """防御性校验输入文本批次的非空性与长度上限 (Defensive Batch Validation)

        Args:
            texts (list[str]): 输入待校验的字符串列表。

        Raises:
            HTTPException(422): 文本为空、包含纯空白或超过 10000 字符限制时抛出。
        """
        if any(not isinstance(t, str) or not t.strip() or len(t) > 10000 for t in texts):
            raise HTTPException(422, 'Invalid text batch: texts must be non-empty and under 10000 characters')

    @app.get('/health')
    def health() -> dict:
        """服务健康与就绪探针接口 (Health Check Probe)

        Returns:
            dict: 包含服务名称、当前就绪状态、推理计算设备、已装载模型列表及模型哈希标识的健康元数据。
        """
        return {
            'service': 'solaceai-local-bge',
            'status': 'UP',
            'device': device,
            'loadedModels': list(loaded),
            'embeddingIdentity': identity,
        }

    @app.post('/embed')
    def embed(body: Texts) -> dict:
        """执行 BGE-M3 密集向量与词元级稀疏权重联合特征提取 (BGE-M3 Joint Embedding)

        首次调用时触发 BGE-M3 权重懒加载。

        Args:
            body (Texts): 包含文本列表的请求体。

        Returns:
            dict: 包含 'dense'(稠密浮点向量矩阵) 与 'sparse'(词元权重哈希字典列表) 的结构体。

        Raises:
            RuntimeError: 模型文件缺失或推理失败时抛出。
        """
        validate(body.texts)
        with lock:
            # 懒加载初始化 BGE-M3 模型
            if 'embedding' not in loaded:
                if not Path(embedding_path).is_dir():
                    raise RuntimeError('Local BGE-M3 files missing')
                from FlagEmbedding import BGEM3FlagModel
                loaded['embedding'] = BGEM3FlagModel(
                    embedding_path,
                    use_fp16=device != 'cpu',
                    devices=device,
                )
            model = loaded['embedding']
            # 执行特征编码计算
            result = model.encode(
                body.texts,
                batch_size=4,
                max_length=512,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            return {
                'dense': result['dense_vecs'].tolist(),
                'sparse': [{str(k): float(v) for k, v in row.items()} for row in result['lexical_weights']],
            }

    @app.post('/rerank')
    def rerank(body: Ranking) -> dict:
        """调用 CrossEncoder 交叉编码器对 (query, text) 对执行深层全注意力相关度打分 (Cross-Encoder Reranking)

        首次调用时触发 CrossEncoder 权重懒加载。

        Args:
            body (Ranking): 包含检索查询 query 与候选文本列表 texts 的请求体。

        Returns:
            dict: 降序排列的候选相关度评分列表，格式为 [{'index': int, 'relevance_score': float}]。

        Raises:
            RuntimeError: 模型目录缺失、权重损坏或推理出现非有限值（NaN/Inf）时抛出。
        """
        validate(body.texts)
        with lock:
            # 懒加载初始化 CrossEncoder 交叉重排模型
            if 'reranker' not in loaded:
                if not Path(reranker_path).is_dir():
                    raise RuntimeError('Local reranker files missing')
                from sentence_transformers import CrossEncoder
                loaded['reranker'] = CrossEncoder(
                    reranker_path,
                    device=device,
                    max_length=512,
                    local_files_only=True,
                )
                if device != 'cpu':
                    loaded['reranker'].model.half()
            model = loaded['reranker']
            # 执行全交互相关度打分预测
            scores = model.predict([(body.query, t) for t in body.texts], batch_size=4, show_progress_bar=False)
            rows = [{'index': i, 'relevance_score': float(v)} for i, v in enumerate(scores)]

            # 校验重排得分数值稳定性，防范数值溢出产生 NaN
            if any(not math.isfinite(row['relevance_score']) for row in rows):
                raise RuntimeError('Nonfinite rerank score')

            # 按相关度得分从高到低排序返回
            return {'results': sorted(rows, key=lambda row: row['relevance_score'], reverse=True)}

    return app


def main() -> None:
    """本地 RAG Worker 微服务命令行主启动函数 (Main CLI Entrypoint)

    解析命令行模型权重路径参数，创建 FastAPI 应用实例并通过 Uvicorn 绑定环回地址监听。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embedding', required=True, help='BGE-M3 本地模型权重目录路径')
    parser.add_argument('--reranker', required=True, help='bge-reranker-large 本地模型权重目录路径')
    parser.add_argument('--device', default='cpu', help='推理计算设备 (cpu/cuda/mps)')
    parser.add_argument('--port', type=int, default=8091, help='本地监听端口 (默认 8091)')
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        create_worker(args.embedding, args.reranker, args.device),
        host='127.0.0.1',
        port=args.port,
    )


if __name__ == '__main__':
    main()
