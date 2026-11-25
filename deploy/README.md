# SolaceAI 部署与基础设施配置 (Deployment & Infrastructure)

本目录集中管理 SolaceAI 系统的容器编排模板与模块化 Python 依赖清单。

---

## 目录组织结构

```text
deploy/
├── compose/
│   └── docker-compose.rag.yml       # Milvus / etcd / MinIO 向量检索基础设施
└── requirements/                    # 按职责拆分的依赖清单
    ├── requirements.txt             # 基础运行依赖
    ├── requirements-dev.txt         # 测试与开发工具依赖
    ├── requirements-documents.txt   # Docling 文档解析独立环境依赖
    ├── requirements-evaluation.txt  # 离线评测套件依赖
    └── requirements-rag-worker.txt  # 独立 BGE-M3 / Cross-Encoder Worker 依赖
```

各应用适配器自身的容器镜像与编排文件位于：
- `apps/offline/Dockerfile`、`apps/offline/compose.yml`、`apps/offline/.env.example`（线下生产版：MySQL + Milvus + Redis + 本地 RAG Worker）
- `apps/online/Dockerfile`、`apps/online/compose.yml`、`apps/online/.env.example`（在线轻量版：SQLite + Chroma + 云端模型 API）
- `apps/rag_worker/Dockerfile`（独立向量化与重排计算服务）

---

## 使用指南

### 1. 准备环境变量与外置资源

复制根配置示例 `.env.example` 为 `.env`，并按实际运行的模式复制对应应用配置：
- 线下生产版：`cp apps/offline/.env.example apps/offline/.env`
- 在线轻量版：`cp apps/online/.env.example apps/online/.env`

应用容器通过各自的 `env_file` 读取 profile 专属参数；根 `.env` 供命令插值及本地源码启动共享读取。知识库原文与模型权重需在部署环境中单独准备，不随开源仓库分发。

### 2. 线下生产版部署（MySQL + Milvus + Redis + RAG Worker）

```bash
# 1) 启动 Milvus / etcd / MinIO 向量检索基础设施
docker compose -f deploy/compose/docker-compose.rag.yml up -d

# 2) 启动线下应用栈（包含 API、Redis 与 RAG Worker；MySQL 需外部提供）
docker compose --env-file .env -f apps/offline/compose.yml up -d --build
```

该编排采用 Linux host 网络模式，本地模型权重由宿主 `models/` 挂载。源码启动默认监听 `127.0.0.1:8082`，容器内服务显式监听 `8080`。

### 3. 在线轻量版部署（SQLite + Chroma）

```bash
docker compose --env-file .env -f apps/online/compose.yml up -d --build
```

源码直接启动：

```bash
pip install -r apps/online/requirements.lock
pip install --no-deps -e packages/core -e apps/online
python -m apps.online.main
```

### 4. 本地源码安装（线下版与独立环境）

```bash
pip install -r apps/offline/requirements.lock
pip install --no-deps -e packages/core -e apps/offline
```

文档解析依赖（`deploy/requirements/requirements-documents.txt`）与 RAG Worker 依赖（`apps/rag_worker/requirements.lock`）建议安装到独立虚拟环境（如 `.venv-rag`），避免与主 API 依赖冲突。
