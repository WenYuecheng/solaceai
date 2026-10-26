# SolaceAI 部署与基础设施配置 (Deployment & Infrastructure)

本目录集中管理 SolaceAI 系统的容器编排、模块化环境依赖与运行环境变量配置。

---

## 目录组织结构

```text
deploy/
├── compose/
│   └── docker-compose.rag.yml       # Milvus / etcd / MinIO
└── requirements/                  # 依赖清单，运行模式以 apps/offline 为准
    ├── requirements.txt
    ├── requirements-dev.txt
    ├── requirements-documents.txt
    ├── requirements-evaluation.txt
    └── requirements-rag-worker.txt
```

---

## 使用指南

本次仓库仅包含线下应用与检索基础设施。完整项目的历史组合编排和轻量模式不属于本次上传范围；请使用 `apps/offline/compose.yml`，不要使用缺失的 Online 入口。

### 1. 准备外置服务与资源

先准备 MySQL、环境文件、本地模型服务，以及本次省略的技能、知识语料、分词器和前端图片字体。应用容器通过 `apps/offline/.env` 加载运行环境；根 `.env` 用于命令插值及源码启动，共享参数需要按实际方式配置。

### 2. 启动检索基础设施

```bash
docker compose -f deploy/compose/docker-compose.rag.yml up -d
```

### 3. 启动线下应用

```bash
# 编排包含 API、Redis 和模型 Worker；MySQL 需要外部提供。
docker compose --env-file .env -f apps/offline/compose.yml up -d --build
```

该编排采用 Linux host 网络模式，模型权重由本地 `models/` 挂载。源码启动默认端口为 8082；容器启动命令显式使用 8080，二者不能混淆。

### 4. 安装线下依赖

```bash
pip install -r apps/offline/requirements.lock
pip install --no-deps -e packages/core -e apps/offline
```

文档解析依赖位于 `deploy/requirements/requirements-documents.txt`，应安装到独立解析环境。模型 Worker 使用自己的锁文件和环境，不要与 API 环境混装。
