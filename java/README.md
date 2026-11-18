# SolaceAI · Java 后端

这是 SolaceAI 的 Java 后端实现，使用 Java 17、Spring Boot、Spring AI 与 LangGraph4j。原 Python 后端位于 [`packages/core`](../packages/core/) 和 [`apps`](../apps/)；两版使用同一套 [Vue 前端](../frontend/)，各自管理账号、会话、运行回执、报告与风险个案。

本目录只发布后端源码、运行时提示词与配置示例。不含前端副本、真实账号或会话、知识文档和索引、模型权重、评测样本、训练与构建产物。现有业务数据不会在两版之间自动迁移。

## 构建与运行

准备 JDK 17 和 Maven 3.9+。从仓库根目录进入 `java/`，使用默认的 `parity` 配置：

```sh
cd java
mvn clean package
read -r -s MINDBRIDGE_BOOTSTRAP_ADMIN_PASSWORD
export MINDBRIDGE_BOOTSTRAP_ADMIN_PASSWORD
java -jar target/solaceai-0.1.0.jar --management.health.redis.enabled=false
```

首次在空业务库启动时，输入 16–128 个字符的初始密码以创建 `admin`；不填写则不会创建管理员。默认仅监听 `127.0.0.1:8080`，关闭演示账号，以 `java/data/parity/` 存储运行数据，告警仅记日志。可用 `curl -fsS http://127.0.0.1:8080/actuator/health` 检查应用启动；该检查不验证模型。使用 Redis 时去掉命令中的健康检查覆盖。

- **模型**：本地模式需要可自行部署的 [双 LoRA 网关](../experiments/qwen3_bonus/)，默认为 `127.0.0.1:18181`，提供 `solace-emotion` 和 `solace-response`。权重和网关令牌自行准备；接入云端需要显式配置 `MODEL_BACKEND=cloud`、模型服务和凭据。
- **检索**：默认 BM25，空知识库需由管理员导入文档。要使用 Milvus、BGE-M3 与 Cross-Encoder，配置 `KNOWLEDGE_BACKEND=milvus_bge` 和 `BGE_RERANKER_ENABLED=true`，启动共享 [RAG Worker](../apps/rag_worker/) 后重建向量索引。原始 PDF/DOCX 还需设置 `DOCLING_PYTHON` 与 `DOCLING_MODULE_PATH`（指向本仓库的 `packages/core`）。
- **容器**：[`docker-compose.parity.yml`](docker-compose.parity.yml) 仅运行 Java 服务，使用 Linux 宿主网络；模型、向量服务和 Redis 须另行准备。用 [`.env.example`](.env.example) 作为配置模板，真实凭据不要提交。

## 接入共享前端

从仓库根目录将前端构建到临时目录，不会覆盖 Python 版的静态资源：

```sh
FRONTEND_DIST="$(mktemp -d)"
(cd frontend && pnpm install --frozen-lockfile && pnpm exec vite build --outDir "$FRONTEND_DIST")
cd java
java -jar target/solaceai-0.1.0.jar --management.health.redis.enabled=false \
  --spring.web.resources.static-locations="file:$FRONTEND_DIST/"
```

打开 `http://127.0.0.1:8080`。由 Java 同源提供页面与 API，不需要关闭请求来源校验。先停止上一条启动命令开的进程再启动这一个；建议两种后端各用自己的数据目录。

## 验证范围

Java 版已用原前端和真实本地模型验证「学生咨询 → 管理员运行审计 → 按日期和风险下载台账」。三组固定输入的 Python/Java 状态图对照一致；这不代表完整生产环境或模型质量指标等同。根 README 的 Python 历史分数不归于 Java，真实 Milvus、Redis、SMTP 的联合验收和完整 Java 评测仍待补齐。

许可证见仓库根目录的 [MIT License](../LICENSE)；第三方代码来源见 [致谢](THIRD_PARTY_NOTICES.md)。
