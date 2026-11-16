"""SolaceAI 全局配置管理模块 (Global Application Settings)

本模块基于 Pydantic-Settings (`BaseSettings`) 实现系统配置的集中定义、类型校验与动态加载：
1. 环境变量读取：自动从项目根目录 `.env` 文件与操作系统环境变量中加载配置项；
2. 模块化划分：涵盖 Agent 协作系统、大语言模型推理、混合检索 RAG、数据库连接池、
   Redis 记忆存储与 SMTP 告警通知等全链路配置；
3. 单例缓存：提供 `get_settings()` 函数结合 `@lru_cache` 实现单例生命周期管理，
   避免重复解析环境变量带来的 I/O 与计算开销。
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录绝对路径（根据当前文件定位向上一级推导至项目根）
from app.core.paths import PROJECT_ROOT


class Settings(BaseSettings):
    """SolaceAI 系统全局配置项类

    继承自 Pydantic `BaseSettings`，提供强类型校验与默认值回退机制。
    所有的字段名均对应小写环境变量或 `.env` 文件中的配置键。
    """

    # =========================================================================
    # 1. 多智能体系统 (Event-Driven Multi-Agent Framework) 配置
    # =========================================================================
    server_host: str = "127.0.0.1"
    server_port: int = 8080
    runtime_profile: str = "legacy"
    redis_namespace: str = "solaceai"
    static_dir: str = ""

    agent_framework: str = "langgraph"
    """智能体执行引擎驱动类型：默认为 'langgraph'，提供基于状态图的有向多轮规划调度"""

    agent_checkpoint_path: str = "data/agent-checkpoints.sqlite"
    """LangGraph 状态图检查点持久化存储 SQLite 数据库路径，支持断点恢复与多轮状态回溯"""

    agent_memory_path: str = "data/agent-memory.sqlite"
    """智能体长期语义记忆与用户信息持久化 SQLite 路径"""

    agent_context_tokens: int = 8192
    """智能体运行期上下文 Token 预算上限，超过后触发分级记忆压缩机制"""

    agent_compaction_headroom_tokens: int = 512
    """上下文压缩的安全余量 Token 数，在触发硬限制前提前启动平滑衰减或摘要压缩"""

    agent_max_decisions: int = 6
    """单轮对话内智能体决策与工具调用的最大跳数（防止循环调用导致死循环）"""

    agent_max_retrieval_rounds: int = 2
    """单轮对话内允许执行知识库检索与重试的最大轮数"""

    paid_api_enabled: bool = False
    """是否启用付费商业 API 计费监控开关"""

    api_budget_cny: float | None = None
    """系统 API 调用的全局预算上限（单位：人民币元），超过后触发熔断保护"""

    api_daily_budget_cny: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    """按 Asia/Shanghai 自然日计算的 API 预算；None 保留原有不限制每日额度的行为"""

    chat_backend: Literal['cloud', 'local'] = 'cloud'
    local_model_switch_enabled: bool = False
    local_gateway_url: str = 'http://127.0.0.1:18181'
    local_gateway_token: str | None = None

    emotion_classifier_enabled: bool = True
    """是否开启前置微调情绪分类器（对学生输入进行情绪倾向初筛）"""

    risk_assessment_source: Literal['auto', 'local', 'cloud'] = 'auto'
    """心理风险评估模型路由策略：'auto'(自动降级) | 'local'(本地微调模型) | 'cloud'(云端大模型)"""

    emotion_classifier_timeout_seconds: float = 20.0
    """情绪分类服务 HTTP 请求超时时间（秒）"""

    api_budget_path: str = "data/api-budget.sqlite"
    """API 调用计量与成本追踪数据存储 SQLite 文件路径"""

    api_price_chat_input_cny_per_million: float = 0.0
    """大模型对话 Prompt 输入单价（元/百万 Tokens）"""

    api_price_chat_output_cny_per_million: float = 0.0
    """大模型对话 Completion 输出单价（元/百万 Tokens）"""

    api_price_embedding_cny_per_million: float = 0.0
    """Embedding 嵌入向量服务单价（元/百万 Tokens）"""

    api_price_rerank_cny_per_million: float = 0.0
    """Rerank 重排序检索服务单价（元/百万次调用或百万 Tokens）"""

    knowledge_retrieval_mode: str = "upgraded"
    """知识检索管线运行模式：'upgraded'(增强混合检索) | 'legacy'(传统基线检索)"""

    knowledge_backend: str = "chroma_bm25"
    """知识库底层检索后端架构：'chroma_bm25'(Chroma 向量与本地 BM25 融合) | 'milvus_bge'(Milvus 分布式向量)"""

    knowledge_parent_chunk_size: int = 1200
    """父子文档检索结构中父分块的字符长度（用于补充完整上下文语义）"""

    local_rag_url: str = "http://127.0.0.1:8091"
    """本地独立 BGE-M3 / Cross-Encoder 神经推理 Worker 进程的 HTTP 访问端点"""

    local_rag_timeout_seconds: float = 180.0
    """本地 RAG 推理 Worker 进程请求超时时间（秒，考虑批量特征计算耗时）"""

    local_rag_autostart: bool = False
    """主服务启动时是否自动在后台守护拉起 local_rag 子进程"""

    local_rag_python: str = ""
    """运行本地 RAG Worker 专用的独立 Python 虚拟环境可执行文件路径"""

    local_bge_model_path: str = "models/bge-m3"
    """本地 BGE-M3 嵌入模型权重文件目录"""

    local_reranker_model_path: str = "models/bge-reranker-large"
    """本地 BGE-Reranker-Large 重排序模型权重文件目录"""

    local_rag_device: str = "cpu"
    """本地神经模型推理计算设备：'cpu' | 'cuda' | 'mps'"""

    milvus_uri: str = "http://127.0.0.1:19530"
    """分布式 Milvus 向量数据库连接 URI"""

    milvus_token: str = ""
    """Milvus 数据库访问鉴权 Token（若启用安全验证）"""

    milvus_collection: str = "solaceai_bge"
    """Milvus 知识切片存储的集合名称"""

    reranker_provider: str = "api"
    """重排序提供方类型：'api'(外部标准 API) | 'local'(本地 CrossEncoder)"""

    knowledge_rewrite_enabled: bool = False
    """是否在检索前调用 LLM 对用户口语化查询进行查询重写（Query Rewrite）"""

    knowledge_rrf_k: int = 60
    """倒数排名融合 (Reciprocal Rank Fusion, RRF) 平滑常数 k"""

    knowledge_recall_k: int = 20
    """多路召回（向量与词法）初筛阶段各路保留的最大候选切片数量"""

    knowledge_rerank_candidate_k: int = 10
    """送入 Cross-Encoder 重排序模型的候选切片截断数量"""

    knowledge_evidence_token_budget: int = 2000
    """最终拼装并注入 Prompt 作为参考证据的知识切片最大 Token 预算"""

    knowledge_contextual_enabled: bool = False
    """是否开启上下文感知切片（Contextual Retrieval）以强化切片独立语义"""

    reranker_enabled: bool = False
    """是否全局启用语义重排序模型阶段"""

    reranker_base_url: str = ""
    """外部 Rerank API 服务 Base URL"""

    reranker_api_key: str = ""
    """外部 Rerank API 鉴权密钥"""

    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    """外部 Rerank 调用的模型名称标识"""

    reranker_timeout_seconds: float = 20.0
    """外部 Rerank 请求超时时间（秒）"""

    # =========================================================================
    # 2. 大语言模型推理核心 (LLM Inference Provider) 配置
    # =========================================================================
    ai_provider: str = "openai"
    """主对话大语言模型驱动提供商：'openai'(OpenAI 兼容协议) | 'ollama'(本地 Ollama 服务)"""

    ai_temperature: float = 0.35
    """生成采样温度，心理场景推荐 0.3~0.4 保障输出稳定、严谨与共情安全"""

    ai_max_tokens: int = 4096
    """单次模型推理生成的最大 Token 数量限制"""

    openai_reasoning_effort: str | None = None
    """针对具备推理思考链模型（如 o1/o3-mini）的思考深度设置：'low' | 'medium' | 'high'"""

    ollama_base_url: str = "http://localhost:11434"
    """本地 Ollama 守护进程 HTTP API 基础访问地址"""

    ollama_model: str = "solaceai-qwen2.5-7b-ft:latest"
    """Ollama 本地微调模型名称标签（历史基线兼容配置；候选高安全入口已升级至 Qwen3-4B 双 LoRA 架构）"""

    # 微调模型资产配置 (LoRA / GGUF 微调权重与 Qwen3 演进说明)
    finetuned_model_name: str = "solaceai-qwen2.5-7b-ft:latest"
    """心理领域微调适配模型全称（原 Qwen2.5 基线，最新方案已在 experiments/qwen3_bonus 内落地为 Qwen3-4B 双 LoRA 体系）"""

    finetuned_model_dir: str = "models/solaceai-qwen2.5-7b-ft"
    """微调权重文件本地存储目录（保留目录结构兼容；项目内重复 GGUF 副本已完成清理并由 Ollama 注册模型管理）"""

    finetuned_model_file: str = "solaceai-qwen2.5-7b-ft-q4_k_m.gguf"
    """GGUF 量化格式权重文件名（适用于 llama.cpp / Ollama 历史基线）"""

    # 兼容 OpenAI API 接口配置 (支持 DeepSeek / 通义千问 / OpenAI / Azure)
    openai_base_url: str = "https://api.deepseek.com/v1"
    """OpenAI 兼容协议服务的 Base URL"""

    openai_api_key: str = ""
    """OpenAI 兼容协议服务的访问凭据 API Key"""

    openai_model: str = "deepseek-flash"
    """云端主推理模型名称"""

    openai_thinking_enabled: bool | None = False
    """是否在对话输出中启用或展示深度思考链输出"""

    # =========================================================================
    # 3. 向量检索与 RAG 知识库 (Embedding & Vector Store) 配置
    # =========================================================================
    embedding_base_url: str = "https://api.chatanywhere.tech/v1"
    """在线向量嵌入服务 Base URL"""

    embedding_api_key: str = ""
    """在线向量嵌入服务鉴权 API Key"""

    embedding_model: str = "text-embedding-3-small"
    """在线向量嵌入模型名称"""

    embedding_dimensions: int = 1024
    """嵌入向量输出维度（如 1024 / 1536 / 384）"""

    embedding_timeout_seconds: float = 30.0
    """在线向量生成请求超时时间（秒）"""

    knowledge_top_k: int = 5
    """最终精选并注入 Prompt 的 Top-K 心理知识切片数量"""

    knowledge_candidate_k: int = 16
    """混合检索粗排阶段召回的候选文档切片总数"""

    knowledge_chunk_size: int = 512
    """知识库文档切片的基础字符长度"""

    knowledge_chunk_overlap: int = 64
    """切片滑动窗口重叠字符量（保持段落边界与语义连贯性）"""

    knowledge_hybrid_vector_weight: float = 0.65
    """混合检索中语义向量相似度的加权系数（与 BM25 权重互补）"""

    knowledge_hybrid_bm25_weight: float = 0.35
    """混合检索中 BM25 词法字面匹配度的加权系数"""

    knowledge_rerank_enabled: bool = True
    """是否启用语义重排序与覆盖率精打分流程"""

    knowledge_vector_enabled: bool = True
    """是否开启 Chroma 向量索引引擎"""

    knowledge_vector_required: bool = False
    """是否强制要求向量库就绪（False 时在网络或向量不可用时平滑降级至 BM25）"""

    knowledge_quality_precheck_enabled: bool = True
    """是否开启知识入库前的文档质量前置清洗校验（过滤乱码与低质文档）"""

    knowledge_quality_threshold: float = 0.80
    """文档有效字符质量置信度阈值（低于此比例拒绝入库）"""

    docling_models_offline: bool = False
    """模型已缓存的本机演示可禁止 Docling 联网检查。"""

    document_converter_python: str = ""
    """独立文档解析环境的 Python 路径；为空时使用当前环境。"""

    docling_ocr_mode: str = "pdf_aware"
    """Docling 版面文档解析 OCR 模式：'pdf_aware'(智能图文感知) | 'full_page' | 'layout_regions'"""

    docling_document_timeout: float = 120.0
    """Docling 复杂多模态文档解析超时阈值（秒）"""

    docling_heading_hierarchy_enabled: bool = True
    """是否开启 Docling 对文档标题层级大纲 (H1~H6) 的自动推断与维护"""

    chroma_persist_dir: str = "data/chroma"
    """Chroma 向量数据库本地持久化存储目录"""

    chroma_collection_name: str = "solaceai_te3small_1024"
    """Chroma 向量集合唯一标识名称"""

    chroma_snapshot_dir: str = "data/chroma-snapshots"
    """Chroma 向量索引备份快照归档存储目录"""

    chroma_snapshot_keep: int = 5
    """Chroma 快照保留的最大历史版本数量"""

    # RAG 离线质量评测参数
    rag_eval_dataset: str = "evaluation/data/rag_cases.json"
    """RAG 离线问答质量评测黄金测试集路径"""

    rag_eval_output: str = "target/rag-eval-report.json"
    """评测结果指标报告生成的输出 JSON 文件路径"""

    # =========================================================================
    # 4. 关系型数据库与存储 (Database & Ledger) 配置
    # =========================================================================
    database_url: str = "mysql+pymysql://solaceai:solaceai@127.0.0.1:3306/solaceai?charset=utf8mb4"
    """业务关系型数据库连接 URL (默认 MySQL 8.0，亦支持 SQLite)"""

    conversation_timeout_seconds: float = Field(default=180, ge=1, le=300, allow_inf_nan=False)
    """Absolute admitted conversation deadline, including every paid upstream call."""

    chat_history_limit: int = 10
    """单轮对话默认加载的最近历史交互消息条数"""

    excel_path: str = "data/solaceai-risk-ledger.xlsx"
    """辅导员危机预警个案线下同步 Excel 台账文件本地存储路径"""

    # =========================================================================
    # 5. Redis 短期记忆与上下文压缩 (Short-Term Memory & Compaction) 配置
    # =========================================================================
    redis_url: str = "redis://127.0.0.1:6379/0"
    """Redis 缓存与短期工作记忆连接 URL"""

    redis_memory_ttl_seconds: int = 86400
    """对话短期缓存在 Redis 中的默认生存过期时间（秒，默认 24 小时）"""

    redis_memory_max_messages: int = 40
    """单个会话在 Redis 维护的最大消息滑动窗口数量上限"""

    redis_socket_timeout_seconds: float = 0.2
    """Redis 网络 I/O 套接字交互超时熔断阈值（秒，历史缓存默认 200ms）"""

    redis_history_cache_enabled: bool = True
    """是否启用基于 Redis 的 Agent 历史消息快照缓存（默认开启）"""

    # =========================================================================
    # 6. SMTP 邮件预警与通知服务 (Alert & Notification Service) 配置
    # =========================================================================
    smtp_host: str = ""
    """邮件服务器 SMTP 主机域名 (例如 smtp.exmail.qq.com)"""

    smtp_port: int = 587
    """邮件服务器通信端口号 (通常 587 用于 STARTTLS, 465 用于 SSL)"""

    smtp_username: str = ""
    """SMTP 认证账号 / 发件人邮箱地址"""

    smtp_password: str = ""
    """SMTP 认证授权密码或独立应用专用安全口令"""

    smtp_use_tls: bool = True
    """是否在连接建立后启用 STARTTLS 加密升级"""

    smtp_use_ssl: bool = False
    """是否在连接建立初期直接采用隐式 SSL/TLS 加密"""

    smtp_timeout_seconds: float = 10.0
    """SMTP 建立网络连接与传输交互的超时阈值（秒）"""

    alert_email_delivery_mode: str = "log"
    """预警通知实际投递模式：'log'(仅写入本地日志，用于离线研发) | 'smtp'(真实发送邮件通知)"""

    alert_email_from: str = ""
    """发件人显示邮箱地址（若为空则默认使用 smtp_username）"""

    alert_email_to: str = ""
    """危机预警邮件接收人邮箱列表，多邮箱通过半角逗号分隔"""

    alert_email_subject_prefix: str = "[SolaceAI 高风险预警]"
    """系统外发预警告警邮件的主题前缀标识"""

    alert_email_rate_limit_per_minute: int = 30
    """高危告警邮件每分钟发送的最大频率限制，防范警报风暴"""

    # =========================================================================
    # 7. Pydantic 行为规则与配置环境映射
    # =========================================================================
    amap_enabled: bool = False
    amap_api_key: str = Field(default='', repr=False)
    amap_timeout_seconds: float = Field(default=8.0, ge=1, le=30)
    amap_daily_call_limit: int = Field(default=100, ge=0, le=10000)
    amap_cache_ttl_seconds: int = Field(default=900, ge=0, le=86400)
    amap_usage_path: str = 'data/amap-usage.sqlite'

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # 忽略未定义的额外环境变量，增强配置向前向后兼容性
        env_parse_none_str="null",
    )

    @property
    def project_root(self) -> Path:
        """获取当前项目的绝对物理根目录路径 (Project Root Path)

        Returns:
            Path: 项目根目录绝对路径对象。
        """
        return PROJECT_ROOT


@lru_cache
def get_settings() -> Settings:
    """获取系统全局配置单例 (Thread-Safe Cached Settings Singleton)

    利用 `functools.lru_cache` 保证只在进程启动或首次访问时解析环境与实例化一次，
    后续调用均直接命中内存缓存，具备极高的读取吞吐性能。

    Returns:
        Settings: 全局已校验的配置实例。
    """
    return _configured_settings if _configured_settings is not None else Settings()


_configured_settings = None

def configure_settings(settings: Settings) -> None:
    """在数据库与路由模块导入之前，为当前进程显式配置单例运行参数 (Configure Process-Wide Settings)

    执行逻辑：
    1. 进程级单模式约束：每个操作系统进程仅允许配置一个 Profile，若检测到已有配置且与新配置冲突，
       抛出 RuntimeError 强制要求在线与离线环境分进程部署；
    2. 导入时序安全性校验：检测 `app.core.database` 是否已被导入，若已被导入且其持有的旧配置与当前配置不一致，
       立即抛出异常，防止数据库连接池以未更新的旧配置错误初始化；
    3. 全局生效：将 `_configured_settings` 锁定为当前传入的配置单例。

    Args:
        settings (Settings): 待锁定的全局系统配置实例。

    Raises:
        RuntimeError: 当试图在同一进程内混用不同 Profile 或在数据库导入后修改配置时抛出。
    """
    global _configured_settings
    import sys
    if _configured_settings is not None and _configured_settings != settings:
        raise RuntimeError('Run online and offline profiles in separate processes')
    if 'app.core.database' in sys.modules:
        existing = sys.modules['app.core.database'].settings
        if existing != settings:
            raise RuntimeError('Configure application settings before importing database services')
    _configured_settings = settings
    get_settings.cache_clear()
