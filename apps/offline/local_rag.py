"""SolaceAI 本地神经检索与重排服务基础设施 (Local Neural Retrieval & Reranker Service).

本模块实现面向高安全性高校场景的纯本地离线深度学习神经检索支撑：
1. `LocalBGEClient`: 封装对本地独立工作进程 (BGE-M3 密集/稀疏混合向量模型与 Cross-Encoder
   语义精排模型) 的标准 HTTP 协议通信，内置维度安全性校验与稠密/稀疏向量清洗；
2. `start_local_rag` & `stop_local_rag`: 负责监督工作器独立子进程 (`app/rag_worker.py`)
   的生命周期，提供幂等启动探测、离线环境变量锁定 (`HF_HUB_OFFLINE=1`) 与优雅退出管理。

Architecture Note:
    高内聚低耦合设计：工作器采用独立 Python 进程或专用环境隔离 PyTorch/CUDA 内存开销，
    防止深度学习框架的 GIL 争用与大内存占用干扰 FastAPI 异步主事件循环。
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from app.core.config import Settings
from app.services.api_budget import is_local_endpoint
from app.services.retrieval import validate_rerank_response


class LocalBGEClient:
    """本地 BGE-M3 混合向量嵌入与 Cross-Encoder 语义精排模型 HTTP 客户端。

    通过环回地址与专用的本地推理工作子进程通信，获得 BGE-M3 的 1024 维稠密特征向量、
    词元级学习型稀疏权重以及 Cross-Encoder 的全局注意力语义精排得分。

    Attributes:
        url (str): 本地 RAG 推理微服务的标准 HTTP 基准地址（必须为 loopback 环回地址）。
        timeout (int | float): 批量向量推理调用的最大超时时间阈值（秒）。
    """

    def __init__(self, url: str, timeout: int | float = 180) -> None:
        """初始化本地 BGE 神经模型通信客户端。

        Args:
            url: 推理服务的基础 URL，强制校验必须属于本机回环地址。
            timeout: 单次推理请求的最大等待超时（秒）。

        Raises:
            ValueError: 当配置的服务端点并非本地回环地址（如 127.0.0.1 或 localhost）时抛出。
        """
        if not is_local_endpoint(url):
            raise ValueError("Local RAG endpoint must be loopback")
        self.url = url.rstrip("/")
        self.timeout = timeout

    def identity(self) -> str:
        """探测本地 RAG 服务的健康状态并获取模型文件的一致性指纹标识。

        Returns:
            str: 结合模型权重路径与 SHA-256 生成的模型版本标识字符串，用于 Milvus 集合校验。

        Raises:
            httpx.HTTPError: 当无法与本地推理子进程建立连接或返回非 200 响应时抛出。
        """
        response = httpx.get(f"{self.url}/health", timeout=10, trust_env=False)
        response.raise_for_status()
        return response.json()["embeddingIdentity"]

    def embed(self, texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]]]:
        """批量调用本地 BGE-M3 服务，同步生成文本的密集向量与词元级稀疏权重向量。

        Args:
            texts: 待计算特征向量的文本列表。

        Returns:
            tuple[list[list[float]], list[dict[int, float]]]:
                - 第一个元素为稠密向量列表，每个稠密向量固定为 1024 维浮点数；
                - 第二个元素为稀疏词元权重映射列表，键为词元 ID (int)，值为正浮点权重。

        Raises:
            ValueError: 当返回的批次大小不符、稠密向量维度非法/包含非有限浮点数、
                或稀疏词表字典格式异常时抛出。
            httpx.HTTPError: 当底层 HTTP 通信失败时抛出。
        """
        response = httpx.post(f"{self.url}/embed", json={"texts": texts}, timeout=self.timeout, trust_env=False)
        response.raise_for_status()
        data = response.json()
        dense, sparse = data["dense"], data["sparse"]

        # 校验响应列表长度一致性
        if len(dense) != len(texts) or len(sparse) != len(texts):
            raise ValueError("Embedding batch size mismatch")

        # 校验密集向量维度 (BGE-M3 标准输出为 1024 维) 与有限浮点安全性
        for vector in dense:
            if len(vector) != 1024 or any(
                isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
                for v in vector
            ):
                raise ValueError("Invalid BGE dense embedding")

        # 校验并净化稀疏词元权重字典
        clean: list[dict[int, float]] = []
        for weights in sparse:
            parsed: dict[int, float] = {}
            if not isinstance(weights, dict):
                raise ValueError("Invalid sparse embedding")
            for key, value in weights.items():
                index = int(key)
                if (
                    index < 0
                    or isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError("Invalid sparse weight")
                parsed[index] = float(value)
            if not parsed:
                raise ValueError("Empty sparse embedding")
            clean.append(parsed)

        return dense, clean

    def rerank(self, query: str, texts: list[str]) -> list[tuple[int, float]]:
        """调用本地 Cross-Encoder 交叉编码器对多候选切片文本执行全注意力交互重排。

        Args:
            query: 用户当前检索查询语句。
            texts: 待重排的候选切片正文列表。

        Returns:
            list[tuple[int, float]]: 降序排序的元组列表 `(original_index, rerank_score)`。

        Raises:
            ValueError: 当重排模型返回异常结构时抛出。
            httpx.HTTPError: 当网络通信超时或出错时抛出。
        """
        response = httpx.post(
            f"{self.url}/rerank",
            json={"query": query, "texts": texts},
            timeout=self.timeout, trust_env=False,
        )
        response.raise_for_status()
        return validate_rerank_response(response.json(), len(texts))


def start_local_rag(settings: Settings) -> subprocess.Popen | None:
    """根据系统配置检查并在需要时启动本地 RAG 深度学习模型工作子进程。

    包含三重工程防线：
    1. 幂等就绪探针：先探测目标端口是否已有健康存活的工作进程，避免多进程重复创建；
    2. 严格离线隔离：强制设置 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`，
       严防 PyTorch/HuggingFace 产生外部网络轮询导致学生隐私泄露或挂起；
    3. 异步就绪轮询：使用指数间隔健康轮询等待推理服务完成权重冷启动加载（最长等待 30 秒）。

    Args:
        settings: 系统全局配置对象。

    Returns:
        subprocess.Popen | None: 新建的工作子进程句柄；若已复用现有实例或未启用则返回 None。

    Raises:
        ValueError: 当配置的模型 URL 不属于环回地址时抛出。
        RuntimeError: 当模型存储路径不存在、工作进程过早崩溃或冷启动健康轮询超时时抛出。
    """
    if not settings.local_rag_autostart:
        return None
    if not is_local_endpoint(settings.local_rag_url):
        raise ValueError("Worker must bind loopback")

    # 1. 探测已有健康实例，避免重复创建进程
    try:
        response = httpx.get(settings.local_rag_url + "/health", timeout=2, trust_env=False)
        if response.status_code == 200 and response.json().get("service") == "solaceai-local-bge":
            return None
    except httpx.HTTPError:
        pass

    root = settings.project_root
    python = settings.local_rag_python or sys.executable
    if not Path(python).is_absolute():
        python = str(root / python)

    def absolute(path: str) -> str:
        p = Path(path)
        return str(p if p.is_absolute() else root / p)

    # 2. 校验模型权重持久化目录
    for path in (settings.local_bge_model_path, settings.local_reranker_model_path):
        if not Path(absolute(path)).is_dir():
            raise RuntimeError("本地RAG模型目录不存在")

    port = urlsplit(settings.local_rag_url).port or 8091
    command = [
        python,
        "-m", "apps.rag_worker.main",
        "--embedding",
        absolute(settings.local_bge_model_path),
        "--reranker",
        absolute(settings.local_reranker_model_path),
        "--device",
        settings.local_rag_device,
        "--port",
        str(port),
    ]

    (root / "data").mkdir(exist_ok=True)
    env = os.environ.copy()
    # 强制离线运行，严防 HuggingFace 网络调用导致阻塞或泄漏
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
    )

    with (root / "data/local-rag-worker.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=log,
            env=env,
            start_new_session=True,
        )

    # 3. 轮询等待健康就绪
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("本地RAG服务启动失败，查看data/local-rag-worker.log")
        try:
            response = httpx.get(settings.local_rag_url + "/health", timeout=2, trust_env=False)
            if response.status_code == 200 and response.json().get("service") == "solaceai-local-bge":
                return process
        except httpx.HTTPError:
            pass
        time.sleep(0.2)

    process.terminate()
    raise RuntimeError("本地RAG服务启动超时")


def stop_local_rag(process: subprocess.Popen | None) -> None:
    """两阶段安全终止本地 RAG 模型后台子进程 (Graceful Shutdown Supervisor)。

    首先尝试发送 SIGTERM 触发优雅退出并等待 10 秒；若子进程超时未能释放 GPU/内存，
    则强制升级为 SIGKILL 确保无僵尸进程残留。

    Args:
        process: 待停止的子进程对象。
    """
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
