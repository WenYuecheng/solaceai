# SolaceAI 官方技术文档中心 (Documentation Center)

欢迎查阅 **SolaceAI** 官方技术文档中心。本文档矩阵面向 **开源社区、AI 应用开发者、架构师与技术评审团队**，提供完整、权威且与源码严格对齐的系统架构与性能评测参考。

---

## 快速导航矩阵

| 文档分类 | 入口链接 | 核心内容与技术关注点 | 适用受众 |
| :--- | :--- | :--- | :--- |
| **项目主页** | [项目主页 (README)](../README.md) | 实拍运行截图、量化性能看板、端云协同拓扑图、核心技术亮点与快速启动指南 | 全部访客 / 开源社区 |
| **架构与算法** | [系统架构与工程落地权威指南](guide.md) | Monorepo 微内核解耦、Qwen3-4B 双 LoRA 微调与 PERM-GRPO 强化学习、独立看门狗架构、多阶段混合 RAG、三层文档抗噪、DSH 上下文压缩、ReAct 六阶段白盒轨迹、分布式租约锁邮件投递协议与生产容器部署实战 | 研发工程师 / 架构师 / 评审专家 |
| **评测白皮书** | [系统性能基准与多维评测报告](evaluation.md) | 123 题检索与生成基准（Hit@4 100%、MRR@4 0.9959）、四模型 80 题统一盲测（PERM-GRPO 64 步均分 3.253、56.25% 盲审偏好得分率、93.75% 换序一致性）、RAGAS 科学解读、双裁判交叉核验与双 LoRA 消融实验 | 算法专家 / 评测研究员 |

---

## 代码与工程组织规范

- **Python 核心微内核**：集中于 `packages/core/app/`，包含 LangGraph 状态机、Harness、多阶段检索、安全审核以及技能提示词（`skills/`）与分词器/城市目录资源（`assets/`）。
- **Python 应用适配器与 Worker**：包含线下生产版 `apps/offline/`、在线轻量版 `apps/online/` 以及独立模型推理工作进程 `apps/rag_worker/`，各自提供独立的 `Dockerfile`、`compose.yml`、`requirements.lock` 与 `.env.example`。
- **共用前端界面**：位于 `frontend/`（Vue 3 + TypeScript + Pinia / Element Plus），包含运行所需字体与插图资源（`frontend/public/`）。
- **模型微调与对齐源码**：Qwen3 双 LoRA 与 PERM-GRPO 强化学习源码位于 `experiments/qwen3_bonus/`；模型权重与本地训练产物不随开源仓库上传。
- **容器编排与依赖清单**：位于 `deploy/` 及各应用的 `compose.yml`。
- **隐私与数据边界**：真实 `.env`、业务数据库、知识库原文与索引、本地模型权重及评测原始运行产物均不随公开仓库提交，需在部署环境自行配置与导入。
