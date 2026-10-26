"""SolaceAI 线下高安全运行模式配置规格 (Offline Application Profile Settings)

本模块定义并管理面向私有化部署与高校机房纯内网的特化应用配置 (Profile):
1. 明确的模式隔离：服务端点默认绑定 8082 端口，锁定 `runtime_profile = 'offline'`；
2. 离线技术栈约束：
   - 锁定 `knowledge_backend = 'milvus_bge'`，使用 Milvus 与本地 BGE-M3；
   - 锁定 `risk_assessment_source = 'local'`，心理评估强制使用本地模型；
   - 启用 `emotion_classifier_enabled = True` 与 `local_rag_autostart = True`，服务启动时自举拉起本地推理子进程；
   - 锁定 `reranker_provider = 'local_cross_encoder'`，采用本地全注意力 CrossEncoder 语义精排；
3. 隔离持久化存储路径：数据库与状态文件统一沉淀在 `data/offline/` 目录。
"""
from pathlib import Path
from typing import Any
from app.core.config import Settings
from app.core.paths import PROJECT_ROOT
from app.core.profiles import load_profile_settings


class ProfileSettings(Settings):
    """线下高安全模式特化配置模型 (Offline Profile Settings Schema)

    定义高校私有化环境运行所需的基础参数与路径映射。

    Attributes:
        server_host (str): 监听主机地址，默认 127.0.0.1。
        server_port (int): 监听端口，默认 8082。
        runtime_profile (str): 运行模式，恒定为 'offline'。
        knowledge_backend (str): 知识库检索后端，固定为 'milvus_bge'。
        risk_assessment_source (str): 心理风险研判来源，固定为 'local'。
        emotion_classifier_enabled (bool): 是否启用本地微调分类器，固定为 True。
        local_rag_autostart (bool): 是否随应用自启动本地 RAG Worker 进程，固定为 True。
        reranker_provider (str): 重排算法提供者，固定为 'local_cross_encoder'。
        database_url (str): 隔离存储的 SQLite/MySQL 数据库连接 URL。
    """
    server_host: str = '127.0.0.1'
    server_port: int = 8082
    runtime_profile: str = 'offline'
    document_converter_python: str = '.venv-rag/bin/python'
    agent_memory_path: str = 'data/offline/agent-memory.sqlite'
    agent_checkpoint_path: str = 'data/offline/agent-checkpoints.sqlite'
    api_budget_path: str = 'data/offline/api-budget.sqlite'
    excel_path: str = 'data/offline/solaceai-risk-ledger.xlsx'
    chroma_persist_dir: str = 'data/offline/chroma'
    chroma_snapshot_dir: str = 'data/offline/chroma-snapshots'
    chroma_collection_name: str = 'solaceai_offline'
    milvus_collection: str = 'solaceai_offline_mysql'
    redis_namespace: str = 'solaceai:offline'
    knowledge_backend: str = 'milvus_bge'
    risk_assessment_source: str = 'local'
    emotion_classifier_enabled: bool = True
    local_rag_autostart: bool = True
    reranker_provider: str = 'local_cross_encoder'


def load_settings(**overrides: Any) -> ProfileSettings:
    """加载、校验并组装线下高安全运行配置实例 (Load Offline Profile Settings)

    执行逻辑：
    1. 优先读取 `apps/offline/.env` 环境变量文件，并与项目通用 `.env` 执行隔离合并；
    2. 防御性合规校验：若配置项未能同时满足 Milvus 检索与本地评估，立即抛出 ValueError 阻断启动；
    3. 规范化路径计算：将各持久化文件路径自动解析为基于 `PROJECT_ROOT` 的绝对路径。

    Args:
        **overrides (Any): 运行时动态覆盖的配置参数。

    Returns:
        ProfileSettings: 校验完备且路径规范化的线下配置单例对象。

    Raises:
        ValueError: 当配置未能满足线下模式必选项时抛出。
    """
    if '_env_file' in overrides:
        settings = ProfileSettings(**overrides)
    else:
        settings = load_profile_settings(
            ProfileSettings,
            PROJECT_ROOT,
            _env_file=PROJECT_ROOT / 'apps/offline/.env',
            **overrides,
        )
    if settings.knowledge_backend != 'milvus_bge' or settings.risk_assessment_source != 'local':
        raise ValueError('offline profile requires Milvus retrieval and local assessment')

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
