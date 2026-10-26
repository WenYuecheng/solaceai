# SolaceAI 部署与基础设施配置 (Deployment & Infrastructure)

本目录集中管理 SolaceAI 系统的容器编排、模块化环境依赖与运行环境变量配置。

---

## 目录组织结构

```
deploy/
├── compose/                      # Docker 容器编排定义
│   ├── docker-compose.yml        # 业务容器栈 (API/前端静态资源 + MySQL + Redis)
│   ├── docker-compose.light.yml  # 轻量级本地容器栈 (适用于低内存/仅测试环境)
│   └── docker-compose.rag.yml    # Milvus/etcd/MinIO 检索基础设施编排
├── requirements/                 # 模块化 Python 依赖清单
│   ├── requirements.txt          # 主应用基础生产依赖
│   ├── requirements-dev.txt      # 开发与单元测试套件依赖 (pytest 等)
│   ├── requirements-documents.txt# 文档解析、Docling 与 OCR 依赖
│   ├── requirements-evaluation.txt # 离线评测与 Ragas 指标评测依赖
│   └── requirements-rag-worker.txt # 独立本地模型服务与 BGE-M3 Worker 依赖
└── env/                          # 环境变量预设与模版
    └── .env.light                # 轻量模式运行环境变量配置
```

---

## 使用指南

### 1. 启动全功能服务
```bash
docker compose -f deploy/compose/docker-compose.yml up -d
```

### 2. 启动轻量化服务
```bash
docker compose -f deploy/compose/docker-compose.light.yml up -d
```

### 3. 安装模块化依赖
```bash
# 基础运行环境
pip install -r deploy/requirements/requirements.txt

# 评测环境
pip install -r deploy/requirements/requirements-evaluation.txt

# 文档解析与版面提取
pip install -r deploy/requirements/requirements-documents.txt
```

所有命令均从项目根目录执行。业务 Compose 的构建上下文指向项目根目录，持久化数据挂载至根目录 `data/`；轻量配置预设从 `deploy/env/.env.light` 读取环境配置模板。生产 RAG 编排提供 Milvus 向量检索集群，独立 Python Worker 可通过应用入口启动。各容器栈均支持按需组合与平滑扩展。
