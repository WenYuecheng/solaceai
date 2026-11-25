# 本机双 LoRA 接入（2026-10-22）

本次用户已明确授权部署与替换原分类器。沿用已有方案，代码仅在 experiments/qwen3_bonus 内变更，不修改原业务文件。

- [x] 安装锁定的 macOS vLLM/Metal 0.31.0；下载相同固定 revision 基座，MLX 4bit 转换，不合并 LoRA。
- [x] 校验训练来源与最终适配器，真实交替推理分类/回答；记录格式、内存、速度和模型名。
- [x] 在独立离线入口中向已有风险评估注入分类客户端，替换该入口的 Ollama 分类调用，保留隐私处理/高危规则/标签映射。
- [x] 回答仅云端失败时替代最终文本；保持正常云端零回答 LoRA 调用、工具只读、独立审核、取消和流式前缀约束。
- [x] 测试先行，修复旧实验在项目目录重组后失效的导入和应用入口；运行接入/SSE 回归。
- [x] 提供单命令启动/停止与完整部署验收报告，并保持旧入口可随时回滚。

网关回环监听 18181，引擎回环监听 18182，共享单一 4-bit MLX 基座与双 rank=8 LoRA 适配器（按 `model` 字段路由），单并发 2K 上下文，空闲 5 分钟自动卸载引擎释放显存；初次冷启动等待上限 10 秒，超时后平滑回退规则兜底并继续后台预热。

## 2026-10-23 本地部署实施记录

训练与离线对照结果详见 [REPORT.md](REPORT.md)，本节记录本机 Apple Silicon Metal 推理栈部署与验收过程。

独立回归测试全部通过：主工程 450 passed / 3 skipped，实验及分类模块 108 passed / 1 skipped。

大型模型下载由后台接续脚本 `artifacts/remote/resume_local_deployment.py` 编排执行，按序完成固定版本哈希校验、Metal 导入检测、MLX 4-bit 转换、双适配器联调验收与候选应用启动，输出分别记录于 `artifacts/local-deployment/continuation.json` 与 `acceptance.json`。

启动、查看状态与停止命令（在项目根目录执行）：

```sh
.venv/bin/python -m experiments.qwen3_bonus.local_stack start
.venv/bin/python -m experiments.qwen3_bonus.local_stack status
.venv/bin/python -m experiments.qwen3_bonus.local_stack stop
```

候选应用地址为 `http://127.0.0.1:8082`（网关 `18181`、引擎 `18182`），复用线下数据库与混合检索栈，仅在分类与云端失败分支接入本地双 LoRA 网关，停止后即可无缝切回原有入口。

部署确认稳定运行后，通过 `cleanup --apply` 清理临时下载缓存、重复训练压缩包与冗余发布包，保留 BGE、Docling、部署基座/适配器、数据库及唯一实验记录（清理明细写入 `artifacts/local-deployment/cleanup.json`）。

## 实际部署验收（2026-10-23）

已完成固定基座完整校验与 MLX 4-bit 转换；四类情绪分类、六次双适配器交替调用、云端故障降级、SSE 流式输出、空闲卸载及重载均全部通过，原始验收记录保存于 `artifacts/local-deployment/acceptance.json`。

本机单并发 SSE 实测采样：首字延迟约 **0.56 秒**，解码速度约 **36.7 tokens/s**。

修复两处真实部署问题：vLLM0.31使用 `--no-enable-log-requests`；端口探测匹配HTTP服务器SO_REUSEADDR，允许TIME_WAIT中的空闲卸载后立即重载，仍拒绝已占用监听端口。

原共享MySQL容器引用了已失效的Java项目初始化脚本挂载，无法启动。已用同一原有数据库卷、同一镜像和原配置启动 `solaceai-local-mysql`，不重建数据库；旧容器保留。Docker原有Redis和Milvus组件已恢复。候选应用8082已返回UP。

最终验收：清理后完整回归546通过、4跳过；两个LoRA再次真实推理通过；网页与健康端点均HTTP200。清理23组/项文件，观察到磁盘空闲增加19.01GB。删除明细见 `artifacts/local-deployment/cleanup.json`；保留BGE、Docling、部署基座/适配器、数据库和唯一实验证据。
