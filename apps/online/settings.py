"""SolaceAI 云端轻量运行模式配置规格 (Online Application Profile Settings)

本模块定义并管理面向云端无本地 GPU 环境的特化应用配置 (Profile):
1. 明确的模式隔离：服务端点默认绑定 8080 端口，锁定 `runtime_profile = 'online'`；
2. 资源轻量化硬限制：
   - 锁定 `knowledge_backend = 'chroma_bm25'`，杜绝尝试连接 Milvus；
   - 锁定 `risk_assessment_source = 'cloud'`，心理研判使用公有云大模型 API；
   - 显式关闭 `emotion_classifier_enabled` 与 `local_rag_autostart`，杜绝拉起本地深度学习推理 Worker；
3. 隔离持久化存储路径：数据库、Chroma、Checkpointer 与记忆 SQLite 均隔离在 `data/online/` 目录。
"""
from pathlib import Path
from typing import Any
from pydantic import Field
from app.core.config import Settings
from app.core.paths import PROJECT_ROOT
from app.core.profiles import load_profile_settings


class ProfileSettings(Settings):
    """云端轻量模式特化配置模型 (Online Profile Settings Schema)

    锁定云端低内存容器运行所需的基础参数与路径映射。

    Attributes:
        server_host (str): 监听主机地址，默认 127.0.0.1。
        server_port (int): 监听端口，默认 8080。
        runtime_profile (str): 运行模式，恒定为 'online'。
        knowledge_backend (str): 知识库检索后端，固定为 'chroma_bm25'。
        risk_assessment_source (str): 心理风险研判来源，固定为 'cloud'。
        emotion_classifier_enabled (bool): 是否启用本地微调分类器，固定为 False。
        local_rag_autostart (bool): 是否自动启动本地 RAG Worker，固定为 False。
        reranker_provider (str): 重排算法提供者，固定为 'lexical'。
        database_url (str): 隔离存储的 SQLite 数据库连接 URL。
    """
    server_host: str = '127.0.0.1'
    server_port: int = 8080
    runtime_profile: str = 'online'
    agent_memory_path: str = 'data/online/agent-memory.sqlite'
    agent_checkpoint_path: str = 'data/online/agent-checkpoints.sqlite'
    api_budget_path: str = 'data/online/api-budget.sqlite'
    api_daily_budget_cny: float = Field(default=1.0, ge=0, le=1.0, allow_inf_nan=False)
    excel_path: str = 'data/online/solaceai-risk-ledger.xlsx'
    chroma_persist_dir: str = 'data/online/chroma'
    chroma_snapshot_dir: str = 'data/online/chroma-snapshots'
    chroma_collection_name: str = 'solaceai_online'
    milvus_collection: str = 'solaceai_online'
    redis_namespace: str = 'solaceai:online'
    knowledge_backend: str = 'chroma_bm25'
    risk_assessment_source: str = 'cloud'
    emotion_classifier_enabled: bool = False
    local_rag_autostart: bool = False
    reranker_provider: str = 'lexical'
    embedding_timeout_seconds: float = 8.0
    database_url: str = 'sqlite:///' + str(PROJECT_ROOT / 'data/online/solaceai.sqlite')


def load_settings(**overrides: Any) -> ProfileSettings:
    """加载、校验并组装在线云端运行配置实例 (Load Online Profile Settings)

    执行逻辑：
    1. 优先读取 `apps/online/.env` 环境变量文件，并与项目通用 `.env` 执行隔离合并；
    2. 防御性合规校验：若检测到启用了 Milvus、本地评估源、本地分类器或本地 Worker，立即抛出 ValueError 阻断启动；
    3. 规范化路径计算：将各持久化文件路径自动解析为基于 `PROJECT_ROOT` 的绝对路径。

    Args:
        **overrides (Any): 运行时动态覆盖的配置参数。

    Returns:
        ProfileSettings: 校验完备且路径规范化的在线配置单例对象。

    Raises:
        ValueError: 当配置试图违规启用离线模式组件时抛出。
    """
    if '_env_file' in overrides:
        settings = ProfileSettings(**overrides)
    else:
        settings = load_profile_settings(
            ProfileSettings,
            PROJECT_ROOT,
            _env_file=PROJECT_ROOT / 'apps/online/.env',
            **overrides,
        )
    if (
        settings.knowledge_backend != 'chroma_bm25'
        or settings.risk_assessment_source != 'cloud'
        or settings.emotion_classifier_enabled
        or settings.local_rag_autostart
        or settings.reranker_provider == 'local_cross_encoder'
    ):
        raise ValueError('online profile cannot enable offline backends, classifier or worker')

    paths = {}
    for key in (
        'agent_memory_path',
        'agent_checkpoint_path',
        'api_budget_path',
        'excel_path',
        'chroma_persist_dir',
        'chroma_snapshot_dir',
    ):
        value = Path(getattr(settings, key))
        paths[key] = str(value if value.is_absolute() else PROJECT_ROOT / value)

    if settings.database_url.startswith('sqlite:///') and ':memory:' not in settings.database_url:
        value = Path(settings.database_url.removeprefix('sqlite:///'))
        paths['database_url'] = 'sqlite:///' + str(value if value.is_absolute() else PROJECT_ROOT / value)

    return settings.model_copy(update=paths)
