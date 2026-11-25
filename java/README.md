# SolaceAI · Java 后端

SolaceAI 的 Java 后端实现基于 **Java 17、Spring Boot、Spring AI 与 LangGraph4j** 构建，与 Python 主后端（[`packages/core`](../packages/core/) 与 [`apps`](../apps/)）共享同一套 [Vue 3 前端](../frontend/)，独立管理账号、会话、运行回执、评估报告与危机个案工作流。

## 构建与运行

准备 JDK 17 与 Maven 3.9+，在仓库根目录进入 `java/` 目录，以默认的 `parity` 配置构建并启动：

```sh
cd java
mvn clean package
read -r -s MINDBRIDGE_BOOTSTRAP_ADMIN_PASSWORD
export MINDBRIDGE_BOOTSTRAP_ADMIN_PASSWORD
java -jar target/solaceai-0.1.0.jar --management.health.redis.enabled=false
```

首次在空库启动时，通过环境变量 `MINDBRIDGE_BOOTSTRAP_ADMIN_PASSWORD`（16–128 字符）初始化 `admin` 账号。服务默认监听 `127.0.0.1:8080`，运行数据持久化至 `java/data/parity/`，告警默认采用日志输出模式。启动后可通过 `curl -fsS http://127.0.0.1:8080/actuator/health` 检查服务健康状态；启用 Redis 时移除命令中的 Redis 健康检查覆盖项即可。

- **模型接入**：本地模式默认连接部署于 `127.0.0.1:18181` 的 [双 LoRA 网关](../experiments/qwen3_bonus/)（提供 `solace-emotion` 与 `solace-response`）；接入云端大模型时配置 `MODEL_BACKEND=cloud` 及相应 API 参数即可。
- **混合检索**：默认开箱使用 BM25，知识库支持在管理后台直接导入文档。如需启用 Milvus、BGE-M3 与 Cross-Encoder 精排，配置 `KNOWLEDGE_BACKEND=milvus_bge` 与 `BGE_RERANKER_ENABLED=true`，启动共享 [RAG Worker](../apps/rag_worker/) 后重建向量索引。解析原始 PDF/DOCX 时配置 `DOCLING_PYTHON` 与 `DOCLING_MODULE_PATH`（指向仓库 `packages/core`）。
- **容器编排**：[`docker-compose.parity.yml`](docker-compose.parity.yml) 提供 Java 服务的独立容器定义（采用 Linux host 网络，配合外部模型、向量检索与 Redis 服务使用），环境变量参考 [`.env.example`](.env.example)。

## 接入共享前端

在仓库根目录将 Vue 前端构建至独立目录并由 Java 后端直接托管：

```sh
FRONTEND_DIST="$(mktemp -d)"
(cd frontend && pnpm install --frozen-lockfile && pnpm exec vite build --outDir "$FRONTEND_DIST")
cd java
java -jar target/solaceai-0.1.0.jar --management.health.redis.enabled=false \
  --spring.web.resources.static-locations="file:$FRONTEND_DIST/"
```

访问 `http://127.0.0.1:8080` 即可使用由 Java 后端同源托管的完整学生端与管理员工作台。

## 核心能力对齐

Java 后端已完整打通共用 Vue 前端与本地双 LoRA 推理网关，覆盖「学生流式咨询 → LangGraph4j 状态机编排 → 管理员运行轨迹审计 → 按日期与风险级别导出防注入 Excel 台账」全链路，并与 Python 后端保持核心状态转移与接口契约一致。

开源许可见仓库根目录 [MIT License](../LICENSE)；第三方依赖致谢见 [致谢](THIRD_PARTY_NOTICES.md)。
