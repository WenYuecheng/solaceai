# 可选的 Qwen3 双 LoRA 本地兜底

本目录提供独立网关和显式选择的 SolaceAI 入口。原 `app.main:app`、已有 DeepSeek 配置、原 Qwen 情绪分类器和原应用文件均不需要修改。默认不开启新本地调用；仅启动网关也不会自动接管应用。

固定基座为 `Qwen/Qwen3-4B-Instruct-2507`，revision 为 `cdbee75f17c01a7cc42f958dc650907174af0554`。部署使用原基座转换得到的 **MLX 4-bit safetensors** ，与两个独立 PEFT LoRA 组合：`solace-emotion` 输出“正常 / 焦虑 / 低落 / 高风险”，`solace-response` 生成短回复。此路径不使用 GGUF，也不通过合并 adapter 生成两个完整基座副本。

## 准备与检查

先按本目录训练与导出流程生成两个 adapter，在 Apple Silicon 独立环境安装经确认兼容的 vLLM Metal / MLX 运行时，并执行部署转换。示例配置 `experiments/qwen3_bonus/configs/serving.example.json` 使用绝对路径；应复制为本目录下的可信本地配置，再改为实际安装和产物路径。配置只由启动操作指定，HTTP 请求不能选择配置、文件路径或第三个模型。

启动前会核对 `config.json`、`bonus_source.json`、各 safetensors 的 SHA-256，以及两个 `bonus_adapter.json` 的基座 revision、任务、rank、alpha、target modules、trained 标志与权重 SHA-256。基座来源清单覆盖配置与全部模型权重文件，确保本地加载来源完整可追溯。

仅检查本地条件，不加载模型：

```bash
.venv/bin/python -m experiments.qwen3_bonus.probe \
  --config experiments/qwen3_bonus/configs/serving.example.json \
  --output experiments/qwen3_bonus/artifacts/prerequisites.json
```

基座、训练产物和 Metal 环境全部就绪后，可执行双 adapter 联调冒烟检查：

```bash
.venv/bin/python -m experiments.qwen3_bonus.probe \
  --config experiments/qwen3_bonus/configs/serving.json \
  --output experiments/qwen3_bonus/artifacts/metal-smoke.json --live
```

`--live` 仅访问本机回环，拉起受控引擎进程，分别请求 `solace-emotion` 与 `solace-response` 后优雅关闭，并输出包含 `genuine_inference` 与格式校验结果的诊断报告。

## 独立网关

```bash
.venv/bin/python -m experiments.qwen3_bonus.serving \
  --config experiments/qwen3_bonus/configs/serving.json
```

网关只绑定 `127.0.0.1:18181`，提供 `GET /health` 与 `POST /v1/chat/completions`。健康端点只返回状态、活动请求数和启动失败次数。`/health` 不会加载模型。聊天请求模型只接受两个固定名称；消息只接受文本的 system/user/assistant 角色，不接受 tools、adapter_path、多媒体或任意扩展参数。请求体上限 64 KiB；输出上限为情绪 16、回复 256 tokens；引擎总上下文固定 2048 tokens，超限由引擎拒绝并向调用方返回 503，应用继续走原来的降级流程。

```bash
curl http://127.0.0.1:18181/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"solace-emotion","messages":[{"role":"user","content":"明天考试，有点紧张"}],"max_tokens":16}'
```

首次有效请求启动一个 `127.0.0.1:18182` 子进程。启动参数包含：

```text
vllm serve <MLX基座绝对路径>
  --host 127.0.0.1 --port 18182
  --enable-lora
  --lora-modules solace-emotion=<绝对路径> solace-response=<绝对路径>
  --max-loras 2 --max-lora-rank 8 --max-model-len 2048 --max-num-seqs 1
```

请求通过 model 名称路由，不在请求间修改全局 adapter。网关在生成期间持有串行锁，SSE 结束或取消时关闭上游连接并释放锁。SSE 总时长和输出字节数都有上限；已经开始的 SSE 发生故障时发送错误事件并结束。

并发首次请求共用一次后台预热。单个冷请求最多等待 10 秒就绪，未完成则返回 503，后台预热继续；后续请求可在就绪后使用。启动失败有冷却期与连续失败上限，达到上限后需要重启网关。活动请求清零并空闲 300 秒后，网关终止整个引擎进程组，从而释放模型权重占用。没有使用仅释放 KV cache 的 sleep 代替进程退出。网关自身保留，下一次有效请求可重新启动引擎。

已有监听者不会被直接复用。引擎端口启动前必须空闲，就绪时需同时确认监听者属于本次启动的进程组、随机子进程 token 校验成功，且 `/v1/models` 包含两个预注册名称。子进程仅继承少量系统环境，不继承云端 key、HTTP 代理或任意父环境；动态 LoRA 更新被关闭。两段 HTTP 传输都禁用代理环境和重定向。

`local_token` 可选；设置后调用网关必须携带对应 Bearer token。此 token 不会透传成引擎 token。应用使用自定义 token 或端口时，可显式构造 `LocalGatewayClient(url=..., token=...)` 并传给 `create_app(local_client=...)`；该注入接口用于集成测试或受控研究运行，调用者负责先完成同一发布门禁。默认生产工厂使用 `127.0.0.1:18181`、无 token。

## 应用接入与发布门禁

原入口保持不变。先按评测流程生成 `passed: true` 的发布报告，才能通过默认真实客户端启用新入口：

```bash
BONUS_LOCAL_FALLBACK_ENABLED=true \
BONUS_RELEASE_REPORT=experiments/qwen3_bonus/artifacts/release-report.json \
.venv/bin/uvicorn qwen3_bonus.asgi:create_app --factory \
  --host 127.0.0.1 --port 8080 --workers 1
```

`BONUS_LOCAL_FALLBACK_ENABLED` 默认 `false`，设为 `true` 且发布报告校验通过后启用；独立网关也可单独用于离线评测与压测。

该工厂重用原应用、原鉴权和原聊天端点，在这个应用实例内连接新的 ChatService / worker / harness / runtime 工厂。已有端点、构造器和运行函数通过独立命名空间副本替换注入点；没有修改原模块的全局类、函数或 Settings，也没有导入即生效的 monkeypatch。SolaceAI 若修改这些装配函数，必须重新运行接入测试。

只有两类调用可进入新本地模型：

- 云端生成/原生决定请求失败后，用原始消息、真实历史及检索证据生成纯文本回答。决定接口返回 `{"answer": ...}`，不声明本地模型具有云端工具能力。本地结构化工具调用或协议封装会被拒绝；上下文超限错误继续上抛给原上下文处理流程。
- 云端心理评估已在原评估服务中失败，来源为 `local_heuristic`，且当前不属于原离线情绪路径时，才调用本地情绪 adapter。标签不合法、网关冷启动 503 或本地推理失败时，保留原启发式结果。

DeepSeek 正常时优先走云端链路；意图、摘要、工具决策与独立安全复核由既有主链承担，端侧双 LoRA 聚焦于私密情绪分类与云端异常时的共情降级回复。高风险规则始终优先，若一轮内触发本地回复降级，`used_fallback` 标志将持续保持至本轮结束并强制工具只读，下一轮自动重置；已开始流式输出或已取消的请求保持原状态不二次切换。

## 验证范围与上游依据

本地测试命令：

```bash
.venv/bin/python -m pytest qwen3_bonus/tests/test_serving.py \
  qwen3_bonus/tests/test_integration.py -q
```

测试覆盖请求限制、禁用时零调用、超时兜底、冷启动 503、恢复、同轮标志、历史证据保留、错误标签、SSE 后缀故障/取消、真实轻量 HTTP 子进程启动与空闲回收、端口拥有者核验、实际新应用端点接线，以及原模块与安全客户端不被替换。测试不访问付费 API。

参数形式依据 [vLLM LoRA 官方文档](https://github.com/vllm-project/vllm/blob/main/docs/features/lora.md)；Apple Silicon/MLX 后端说明见 [vLLM Metal 官方文档](https://docs.vllm.ai/projects/vllm-metal/en/latest/)。最终兼容性以实际安装版本和 `probe --live` 的结果为准。

发布报告现在绑定 MLX 基座清单、两个适配器的配置/权重/训练来源清单 SHA-256。回答与分类的候选输出必须在 capture 时记录同一组指纹，release 时再次核对；换权重后不能沿用旧捕获或旧报告。`BONUS_SERVING_CONFIG` 指定与网关相同的配置文件，应用读取其中的网关端口、token 和超时。未设置时使用 `configs/serving.example.json`。评测期间保持产物不变，重新导出后先停止旧网关再重启，避免已有进程持有旧权重。

租卡前的加载预检使用 `compatibility.py` 创建显式 `trained=false` 的零适配器，验证量化基座与 PEFT 路由兼容性；训练完成后的双 LoRA 验收通过 `probe --live` 与离线评测套件执行。

排队与冷启动共用单次 10 秒截止；原模型调用耗时也计入后续本地剩余预算。Graph 决策阶段用局部 ContextVar 传递截止，网关的 `X-Solace-Timeout` 只能缩短配置超时，不能扩大它。本地明确的上下文超限以标准 `context_length_exceeded` 返回给原压缩处理，普通加载/推理故障才返回 503。


## 学生端全本地模式（2026-10-23）

学生页面现显示实际模型并提供云端/本地选择。`modelBackend=local`按请求投影设置，分类和回答均调用此网关，辅助路由和高危判断使用明确规则，不调用付费云端API。回复适配器不负责生成工具协议；工具执行仍由既有受控图完成。正常云端模式仍可在云端回复失败时使用本地回答作为备用。云服务器拒绝本地模式请求。

全本地端到端验收记录与数据库状态校验已归档留存。


## 2026-10-25 当前回答降级选择

本机 `serving.local.json` 的 `response_adapter` 指向 `artifacts/adapters/solace-response-perm-grpo64/`，服务别名为 `solace-response`。该模型为完整 64 步 PERM-GRPO 终态快照，在 80 题四模型统一评测中四维质量均分达到 **3.253**、对基座偏好得分率达 **56.25%**，在倾听与陪伴核心场景表现突出。分类适配器为 `solace-emotion`，云端主调用和业务 API 契约保持不变；原 GRPO32+SFT 适配器保留于 `artifacts/adapters/solace-response/` 供对照或快速回滚。

部署选择依据、权重指纹与运行绑定见 [完整评测](PERM_EVALUATION.md) 与 `artifacts/local-deployment/selected-model.json`。双模型交替推理、四类分类、云端故障降级、SSE 流式输出、闲置卸载与重载均已验收通过，可通过 `python -m experiments.qwen3_bonus.local_stack start|stop` 启停（入口 `http://127.0.0.1:8082`）。


### 2026-10-28 线下运行时升级

本机 `serving.local.json` 已升级至独立 `.venv-metal-032` 环境（vLLM / vLLM-Metal 0.32.0，MLX 0.32.2），继续复用原 MLX 4-bit 基座与双 LoRA 适配器。四类分类、12 次交替调用、双模型 SSE、空闲卸载与重载全量通过，12 次确定性输出与上一版本完全一致（首次冷启 91.403 秒，重载 15.868 秒），线下服务健康检查返回 `UP`，相关 51 项回归测试全部通过。依赖锁定文件已记录精确版本、官方 wheel 地址及固定 Git revision。
