# 心栖 · SolaceAI

<div align="center">

 **基于智能体编排与混合检索增强（RAG）的工业级端云协同心理健康支持系统**
*An Industrial-Grade Edge-Cloud Collaborative Mental Health Agent Powered by LangGraph, Multi-Stage Hybrid RAG, and MLX Dual-LoRA*

[![Python Version](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-State_Machine-orange.svg)](https://github.com/langchain-ai/langgraph)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Multi__Stage_RAG-purple.svg)](https://www.llamaindex.ai/)
[![Model Context Protocol](https://img.shields.io/badge/Protocol-MCP-black.svg)](https://modelcontextprotocol.io/)
[![PERM-GRPO](https://img.shields.io/badge/RL_Algorithm-PERM--GRPO-blue.svg)](docs/guide.md#第三章端侧-qwen3-4b-共享基座双-lora-微调与-metal-加速体系)
[![Dual-LoRA](https://img.shields.io/badge/Fine--tuning-Qwen3__Dual__LoRA-green.svg)](docs/guide.md#第三章端侧-qwen3-4b-共享基座双-lora-微调与-metal-加速体系)
[![Tests](https://img.shields.io/badge/Regression-592_Passed-brightgreen.svg)](docs/evaluation.md)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](deploy/)

[🚀 在线演示](https://mindbridge.wenyuecheng.cn) · [📖 系统架构指南 (核心白皮书)](docs/guide.md) · [📊 性能基准报告](docs/evaluation.md) · [⚡ 极速上手](#-极速上手与部署-quick-start) · [📂 官方文档中心](docs/README.md)

</div>

---

## 📸 实际运行截图

以下截图于 **2026-11-16** 启动本地 Vue 前端与 FastAPI 后端后，通过浏览器实际操作拍摄。使用独立演示账号、全新 SQLite 演示库和示例提问，管理端报告由本次对话自动生成；截图不包含真实用户资料。

### 登录与学生工作台

邮箱与密码登录入口，以及包含历史会话、情绪快捷入口和个人记忆管理的学生首页。

![SolaceAI 登录页：邮箱验证码与密码登录入口](docs/images/runtime/01-login.png)

![SolaceAI 学生首页：心情手账、对话区与小憩角落](docs/images/runtime/02-student-home.png)

### 真实对话与执行过程

向系统提交毕业设计与面试压力的示例问题，展示 DeepSeek Flash 实际生成的回复，以及可展开的 Agent 决策、工具调用和结果记录。

![SolaceAI 实际对话：示例提问与模型回复](docs/images/runtime/03-student-chat.png)

![SolaceAI 回答执行过程：实际步骤、状态与耗时](docs/images/runtime/04-execution-trace.png)

本次演示使用本地 Ollama 分类与 DeepSeek Flash 回答，检索使用 BM25 和词法重排；未启动 Milvus/BGE 向量服务，界面如实显示检索降级状态。完整混合检索的配置与历史评测见[系统架构指南](docs/guide.md)和[评测说明](docs/evaluation.md)。

### 管理工作台与知识库

管理端读取本次演示产生的咨询报告，展示报告数量、风险分布、工具执行统计和知识库维护入口。

![SolaceAI 管理工作台：本次演示报告与统计图表](docs/images/runtime/05-admin-overview.png)

![SolaceAI 知识库维护：片段统计、文件上传与索引维护](docs/images/runtime/06-admin-knowledge.png)

<details>
<summary>查看移动端登录效果</summary>

<img src="docs/images/runtime/07-mobile-login.png" alt="SolaceAI 移动端登录页实拍" width="390" />

</details>

---

## 🌟 旗舰级量化性能看板 (Flagship Performance Metrics)

下表汇总既有实验记录：123 题检索/生成评测、400 条合成分类回归及 80 题四模型回答评测。不同实验的样本、配置和指标范围分别见评测文档。

| 维度 | 评测指标 | 达成表现 | 架构与算法保障 |
| :--- | :--- | :---: | :--- |
| **检索命中率** | **Hit@4** | **100.00%** | 父子分块 + BGE-M3 (稠密向量 + 学习型稀疏) 双路召回 |
| **检索排序质量** | **MRR@4** | **0.9959** | 互惠名次融合 (RRF) + Cross-Encoder 交叉编码器精排 |
| **检索前向准度** | **Precision@4** | **90.85%** | LlamaIndex 编排 + 三层文档抗噪解析治理 |
| **端侧微调精度** | **Accuracy (4分类)** | **99.75%** | Qwen3-4B MLX 4-bit 双 LoRA 架构（分类 LoRA 准确率 99.75%） |
| **分类综合质量** | **Macro F1-Score** | **0.9975** | 高风险零漏判，真实交替推理与云端故障降级全量通过 |
| **端侧回答对齐** | **4D 综合质量均分** | **3.253 / 5.0** | PERM-GRPO 64 步强化学习（四模型本轮评测均分最高） |
| **盲测偏好得分率** | **Head-to-Head Preference** | **56.25%** | 80 题对决对原始基座 27 胜 36 平 17 负，换序复评一致性 93.75% |
| **系统工程健壮性** | **自动化回归测试** | **590+ Passed** | 592 项全量测试持续通过，覆盖端到端状态机、异步投递与危机熔断 |

---

## 💡 核心架构与技术创新 (Key Technical Highlights)

```mermaid
graph TD
    Client[客户端: Web / 移动端 / 微信生态] -->|SSE 流式交互| API[FastAPI 网关服务]
    
    subgraph ControlPlane["【控制平面】Harness 运行时层"]
        API --> Lock[并发会话文件锁 & 防重幂等]
        Lock --> Mask[敏感隐私实体动态脱敏管道]
        Mask --> DSH[自适应 DSH 上下文压缩调度器]
    end

    subgraph ReasoningPlane["【推理平面】端云协同异构编排引擎"]
        DSH --> EdgeClass[端侧 Qwen3-4B 双 LoRA 引擎<br/>毫秒级私密情感分类 & 危机拦截]
        EdgeClass --> Router{意图分流与状态机驱动}
        
        Router -->|CHAT / 闲聊安抚| CloudLLM["云端深度认知大模型: DeepSeek<br/>(云端异常时自动降级至端侧 Qwen3 回答 LoRA)"]
        Router -->|CONSULT / 循证咨询| RAGEngine[多阶段高精度混合检索 RAG]
        Router -->|RISK / 明确高危| SafetyGate[确定性危机干预 & 安全模板熔断]
        
        RAGEngine --> LangGraph[LangGraph 有界 Agent 工具循环]
        LangGraph --> MCP[Model Context Protocol 工具标准沙箱]
        MCP --> CloudLLM
        CloudLLM --> Review[独立后验安全审核节点]
    end

    subgraph DataPlane["【数据与知识存储平面】"]
        RAGEngine --> Milvus[(Milvus 向量库: BGE-M3)]
        RAGEngine --> BM25[(BM25 稀疏词法索引)]
        LangGraph --> MySQL[(MySQL 业务库 / SQLite)]
        LangGraph --> Redis[(Redis 会话缓存与分布式锁)]
        SafetyGate --> AdminAlert[管理员实时危机个案工单 & Excel 台账导出]
    end
```

### 1. Monorepo 微内核解耦架构 (Microkernel Monorepo)
- **微内核物理隔离** ：系统沉淀核心业务微内核包 `packages/core/app`，将状态机编排、多阶段检索算法、危机评估与通用实体集中维护，彻底解耦具体部署运行环境。
- **多端适配器矩阵** ：
  - `apps/offline`：线下生产完整适配器，搭载 MySQL、Milvus 分布式向量库与 BGE-M3 深度重排，支撑校级高并发私有化交付；
  - `apps/rag_worker`：独立检索计算工作进程，实现重型模型计算与业务 Web 容器的物理进程隔离。
- **静态 AST 架构边界检测** ：配备 [`scripts/check-boundaries.py`](scripts/check-boundaries.py)，通过 Python 抽象语法树分析强制执行单向依赖约束，彻底杜绝下层反向污染。

### 2. 企业级“端云协同”异构计算与 PERM-GRPO 容灾降级体系

系统创新性引入结合前沿论文 **PERM（Findings of ACL 2026）** 与 **GDPO（2026）** 解耦归一化思想的 **PERM-GRPO 原生强化学习体系** ，针对单卡消费级算力完成了 64 步端到端对齐，打造工业级端侧高可用回答降级引擎：

- **端侧极速感知与故障降级** ：采用 **Qwen3-4B-Instruct** 共享基座与独立双 LoRA 适配器（MLX 4-bit / Metal 原生硬件加速）。
  -  **分类 LoRA (`solace-emotion`)** ：以毫秒级延迟完成私密情感识别与危机分级（400 条合成回归测试集准确率  **99.75%** 、Macro F1  **0.9975** 、高风险零漏判）；
  -  **回答 LoRA (`solace-response`)** ：搭载  **PERM-GRPO 第 64 步终态快照** （`solace-response-perm-grpo64`）。在 80 题四模型统一盲测中，四维综合质量均分达到  **3.253** （情绪理解 3.400 / 共情表达 3.263 / 需求满足 3.200 / 整体质感 3.150），本轮均分高于原始基座（3.169）与基础 GRPO（3.212）；在两两盲测对决中对原始基座取得  **56.25%**  偏好得分率（27 胜 36 平 17 负），换序复测一致性达  **93.75%** 。在倾听与支持核心 40 题场景中展现出细腻温情的同理心与扎实交付力。
- **云端深度推理与自适应熔断** ：云端接入大模型（DeepSeek）进行多轮深度共情表达与认知重构；当云端 API 出现超时、5xx 或网络抖动时，系统自动无缝降级至端侧 Qwen3 回答 LoRA，实现“端侧低延迟私密感知 + 云端高智能拟人共情 + 零云依赖离线服务韧性”的完整闭环。
- **独立守护进程看门狗架构** ：训练与运行时配备独立 supervisor watchdog（`gdpo_watchdog.py`），支持超时租约硬管控与自愈式单次 OOM 降级重启，通过运行时限监控和检查点恢复支持训练中断处理。

### 3. 多阶段混合检索（RAG）与三层文档抗噪防线
- **端到端多阶段流水线** ：
  -  **文档切分** ：基于 Docling 统一结构化解析，采用“父段落目标 1200 字符 + 子块目标 512 字符 + 重叠 64 字符”的层级分块，子块用于精准定位，父块用于提供饱满上下文。
  -  **双路召回** ： **BGE-M3** （1024 维稠密语义向量）与学习型稀疏词表（Sparse/BM25）协同召回 Top-20 候选。
  -  **排序融合** ：引入  **RRF（Reciprocal Rank Fusion，互惠名次融合）**  算法，无缝消除稠密分数与稀疏分数的量纲差异。
  -  **深度重排** ：采用  **Cross-Encoder（`bge-reranker-large`）**  进行全交叉自注意力打分，按 `knowledge_top_k` 选择证据注入模型上下文，源码默认最多 5 条；历史评测使用 Top-4。
- **工业级“三层防线”文档治理** ：
  针对非规范 PDF/扫描件，先执行有效字符率预检（默认至少 80%），再进行结构化段落重组和检索重排，降低乱码与无关内容进入上下文的风险。

### 4. 自适应 DSH 上下文自适应压缩机制 (Dynamic Slicing & Harnessing)
- 针对多轮长程会话，创新落地自适应窗口管理策略（默认窗口 `W=8192`，动态触发阈值 3584 tokens，近期原文保留目标 655 tokens）。
- **三阶平滑压缩管道** ：
  1. *第一阶段*：对单次超大工具观察结果进行投影智能裁剪（保留关键首尾与元数据）。
  2. *第二阶段*：触发滚动摘要生成器，对早期轮次进行结构化状态折叠，严格保留近期高频原文窗口。
  3. *第三阶段*：动态证据投影裁剪，确保当前核心问题与高危干预规则获得绝对最高的 Token 预算保障。

### 5. 白盒化 ReAct 可解释性执行轨迹流 (Observable Trace)
- **六阶段执行状态机** ：严格划分为 `Prepare`（脱敏与安全预检）➔ `Reason`（决策摘要）➔ `Act`（工具调用）➔ `Observe`（观察回执）➔ `Answer`（候选生成）➔ `Review / Deliver`（安全审核与分发）。
- **毫秒级 SSE 流式可视化** ：实时打字机事件流与阶段元数据无缝推送，赋予系统极致感知体验与完全白盒化的学术解释性。

### 6. 高可用身份安全与基于分布式租约锁的邮件投递协议
- **安全认证闭环** ：学生邮箱验证码登录、Argon2id 强哈希算法、HttpOnly/Secure 会话 Cookie 降低会话凭据被脚本读取的风险。
- **分布式租约锁投递自愈** ：针对异步 SMTP 投递与本地数据库提交无法跨系统 2PC 的痛点，自研基于原子租约锁（Lease Lock）的幂等自愈协议，彻底杜绝并发重投与告警风暴。

### 7. 临床级双重危机干预闭环 (Human-in-the-Loop)
- **前置规则秒级熔断** ：前置确定性危险模式匹配引擎，一旦捕获极端自伤风险信号，瞬间跳过模型直接判定最高风险级别（HIGH），并强制加载危机干预技能。
- **后验独立安全审核节点** ：生成端配备独立安全审核守卫（Safety Reviewer），对高风险候选回复进行合规审查，必要时触发单次自适应修订；若二次审核未通过，直接降级至临床标准危机求助安全模版。
- **多角色协同与台账追溯** ：自动创建风险追踪工单，联动管理员后台实时预警，并支持一键导出符合防 CSV/Excel 注入规范的中文跟进台账。

---

## 🔄 端到端单轮执行链路 (Execution Pipeline)

```mermaid
flowchart TD
    Req[已认证的聊天请求] --> Service[ChatService: 建立 SSE 通道与取消监听]
    Service --> Harness[Harness: 会话排他锁 & 幂等回执检查]
    Harness --> Pipeline[数据管道: 动态脱敏 / 加载历史 / 授权长期记忆]
    Pipeline --> Classify[端侧 Qwen3 双 LoRA 情感与危机快速识别]
    Classify --> GraphRuntime[LangGraph 核心决策节点: decide]
    
    GraphRuntime --> Branch{模型决策: 工具调用还是最终回答?}
    Branch -->|调用工具| MCPExec[MCP 沙箱校验与执行: 混合检索 / 证据抓取]
    MCPExec --> GraphRuntime
    
    Branch -->|生成回答| ReviewNode{是否命中高风险?}
    ReviewNode -->|是| SafetyReview[独立后验安全审核: 合规校验与自适应修订]
    ReviewNode -->|否| Attrib[规范化证据溯源引用标注]
    SafetyReview -->|审核通过| Attrib
    SafetyReview -->|未通过| SafeTemplate[一键降级临床级安全援助模版]
    SafeTemplate --> Attrib
    
    Attrib --> Receipt[提交 AgentTurnReceipt 最终回执]
    Receipt --> Persist[写入用户消息 / 自动流转心理工单 / 写入审计轨迹]
    Persist --> Stream[SSE 毫秒级打字机流式输出并发送 done]
```

---

## ⚡ 极速上手与部署 (Quick Start)

本次同步包含 Vue 前端、FastAPI 共享核心、在线/线下后端与 RAG Worker 源码，以及前端图片字体、技能提示词、分词器等必要运行资源。配置示例见 `.env.example`。实际 `.env`、数据库、会话记录、知识库原文与索引、模型权重、训练产物、日志及本地中间数据不随本次同步上传，需在部署环境自行配置。

### 本地源码安装

```bash
git clone https://github.com/WenYuecheng/solaceai.git
cd solaceai
python3 -m venv .venv
source .venv/bin/activate
pip install -r apps/offline/requirements.lock
pip install --no-deps -e packages/core -e apps/offline
cd frontend
pnpm install --frozen-lockfile
pnpm build
cd ..
```

配置 MySQL、Redis、Milvus、本地模型服务及共享环境参数后，使用 `python -m apps.offline.main` 启动；默认监听 `127.0.0.1:8082`，接口文档为 `/docs`。前端构建结果由后端直接托管；知识库由管理员在部署后导入。在线轻量后端另见 `apps/online`，其依赖与配置独立。

### 容器编排

```bash
# 先准备 MySQL、环境文件和本地模型，再启动检索基础设施。
docker compose -f deploy/compose/docker-compose.rag.yml up -d
# 该编排包含 API、Redis 和模型 Worker；不包含 MySQL。
docker compose --env-file .env -f apps/offline/compose.yml up -d --build
```

应用 Compose 的 `env_file` 读取 `apps/offline/.env`；根 `.env` 供命令插值及源码启动读取。共享连接参数应按部署方式放入实际使用的环境文件。历史测试数量属于完整项目记录，不表示本次精简仓库包含相同的全部测试和运行资源。

---

## 📂 工程架构与目录矩阵

```
solaceai/
├── packages/core/app/                # 系统核心微内核命名空间 (共享业务与算法实现)
│   ├── agents/                       # LangGraph 状态机、GraphRuntime 与 Harness 编排
│   ├── services/                     # 核心服务: 混合检索、上下文压缩(DSH)、文档治理、危机评估
│   ├── api/                          # FastAPI 路由、SSE 流式端点与管理员管控接口
│   ├── assets/                       # 分词器与城市目录等运行资源
│   ├── skills/                       # 心理支持技能提示词（不含用户数据）
│   └── models/                       # 数据库 ORM 实体与 Pydantic 校验模型
├── apps/                             # 独立应用适配器与独立 Worker
│   ├── online/                       # 在线轻量适配器 (SQLite + Chroma)
│   ├── offline/                      # 线下生产适配器 (MySQL + Milvus + Redis)
│   └── rag_worker/                   # 独立 BGE-M3 / Cross-Encoder 模型计算服务
├── experiments/qwen3_bonus/           # 历史已发布的训练源码，本次不同步本地实验产物
├── docs/                             # 系统权威技术文档矩阵
│   ├── README.md                     # 文档中心全局导航
│   ├── guide.md                      # 🔥 核心白皮书: 系统架构、微内核设计与全栈工程落地权威指南
│   └── evaluation.md                 # 权威性能基准评测报告 (100% Hit@4, RAGAS 深度分析)
├── evaluation/                       # 历史评测说明，本次不同步本地评测数据
├── frontend/                         # Vue 3 + TypeScript + Pinia / Element Plus 前端
├── deploy/                           # Docker Compose 编排、环境预设与模块化依赖
└── tests/                            # 完整项目测试目录，本次精简上传未包含
```

---

## 📜 理论基石与参考文献 (Scientific Foundations & Citations)

SolaceAI 系统架构与核心算法严格遵循并拓展了人工智能、自然语言处理与信息检索领域的经典理论：

```bibtex
@inproceedings{cheng2026perm,
  title={PERM: Psychology-grounded Empathetic Reward Modeling for Large Language Models},
  author={Cheng, Yu and others},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026}
}

@article{zhang2026gdpo,
  title={GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization},
  author={Zhang, Zhaowei and others},
  journal={arXiv preprint arXiv:2601.05242},
  year={2026}
}

@inproceedings{lewis2020retrieval,
  title={Retrieval-augmented generation for knowledge-intensive nlp tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={33},
  pages={9459--9474},
  year={2020}
}

@inproceedings{dettmers2023qlora,
  title={Qlora: Efficient finetuning of quantized llms},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={36},
  year={2023}
}

@inproceedings{cormack2009reciprocal,
  title={Reciprocal rank fusion outperforms condorcet and individual rank learning methods},
  author={Cormack, Gordon V and Clarke, Charles LA and B{\"u}ttcher, Stefan},
  booktitle={Proceedings of the 32nd International ACM SIGIR Conference (SIGIR)},
  pages={758--759},
  year={2009}
}

@inproceedings{yao2022react,
  title={ReAct: Synergizing Reasoning and Acting in Language Models},
  author={Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```

---

## 📄 开源许可与合规致谢

本项目采用 [MIT License](LICENSE) 开源许可。系统仅提供情绪舒缓支持与心理知识科普检索，不替代任何线下面对面的临床医疗诊断与专业精神卫生干预。系统内置临床急救援助热线与快速人工个案转介通道。
