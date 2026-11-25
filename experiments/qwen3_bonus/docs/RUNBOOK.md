# 从本机准备到租卡、评测和回滚

所有命令均从 SolaceAI 项目根目录执行。macOS 与 Linux GPU 环境各自使用独立的虚拟环境（`.venv-tools` / `.venv-metal` / `.venv-gpu`），与主业务 `.venv` 相互隔离。

## 1. 免费本机准备

数据已经下载、清洗并冻结在 `experiments/qwen3_bonus/artifacts/data`，包括 JSONL、Parquet、清单和指纹。复现细节见 DATA.md；重复生成须指定新目录，原冻结目录不要手工修改。

```sh
uv venv --python 3.11 experiments/qwen3_bonus/.venv-tools
uv pip install --python experiments/qwen3_bonus/.venv-tools/bin/python --require-hashes -r experiments/qwen3_bonus/requirements-tools.lock
uv pip install --python experiments/qwen3_bonus/.venv-tools/bin/python --require-hashes -r experiments/qwen3_bonus/requirements-calibration.lock
experiments/qwen3_bonus/.venv-tools/bin/python -m experiments.qwen3_bonus.models download --kind base
experiments/qwen3_bonus/.venv-tools/bin/python -m experiments.qwen3_bonus.models download --kind reward
```

若环境已存在就跳过创建。基座约数 GB，奖励模型也需完整权重；目前目录中的零散下载文件不代表下载完成。下载命令可以重试，不会调用模型 API，不需要 token；只有生成并通过 `bonus_source.json` 校验才算完成。保持足够磁盘空间，整个模型、环境和缓存建议留 25 GB 以上。不要使用未验证的镜像权重替换固定 revision。

打开 `artifacts/calibration/pairs.json`，逐对审阅 `prompt`、`a`、`b`，填写 `expected`（`"a"` 或 `"b"`）、`reviewed: true` 及 `reviewer` 标识；校准过程与冻结评测集 `response_eval` 保持严格隔离。

```sh
experiments/qwen3_bonus/.venv-tools/bin/python -m experiments.qwen3_bonus.calibration score
experiments/qwen3_bonus/.venv-tools/bin/python -m experiments.qwen3_bonus.calibration check
```

只有 20 对全部经人审、至少 15 对排名相符才继续。未通过则停止租卡准备，保留失败结果，重新考虑奖励模型/任务，而不是降低通过线。评分输出不可覆盖；重评指定新的 `--output`，训练时相应传 `--calibration`。

## 2. 租卡前验证本机 Metal 兼容性

Metal 和训练使用独立依赖组。这里锁定官方 0.31.0 匹配的 macOS arm64 Python 3.12 wheels，mlx-lm 为官方依赖指定的固定 Git commit。不是把 PyPI 的旧版 `vllm-metal` 当作当前 GitHub release 安装。

```sh
uv venv --python 3.12 experiments/qwen3_bonus/.venv-metal
uv pip install --python experiments/qwen3_bonus/.venv-metal/bin/python -r experiments/qwen3_bonus/requirements-metal.lock
experiments/qwen3_bonus/.venv-metal/bin/python -m experiments.qwen3_bonus.deployment convert
cp experiments/qwen3_bonus/configs/serving.example.json experiments/qwen3_bonus/configs/serving.json
```

检查配置中的绝对路径（`engine_executable`、`model_path` 与适配器路径）。工具、校准与 Linux GPU 锁文件支持 `--require-hashes` 校验，`requirements-metal.lock` 锁定官方 vLLM-Metal 依赖与固定 Git commit。

在进入 GPU 训练前，可先通过零初始化 LoRA 验证量化基座加载与双适配器路由接口：

```sh
experiments/qwen3_bonus/.venv-metal/bin/python -m experiments.qwen3_bonus.compatibility \
  --config experiments/qwen3_bonus/configs/serving.json
```

该步骤输出 `artifacts/compatibility/temporary-adapters` 与 `report.json`，验证通过（`compatibility_passed: true` 且 `genuine_inference: true`）后即可进入训练阶段；生产网关会自动拒绝 `trained=false` 的预检产物。

原始基座、MLX 量化基座、tokenizer 和适配器会占磁盘多份；运行时仍是一个 MLX 基座加两份 LoRA。不要同时启动基座对照服务与双 LoRA 服务，以免 M4 16 GB 内存不足。

## 3. GPU 训练前准备

将整个 `qwen3_bonus` 的代码、锁文件、`artifacts/data`、完整 `artifacts/models/base` 和 `reward`、人工已通过的 calibration、compatibility 报告传到 Linux。复制完整模型清单，保持指纹不变；不要上传 `.venv-*`、`artifacts/python`、`uv-cache`、临时适配器。数据准备也需要的原始项目输入可留本机；训练直接读已冻结的 JSONL/Parquet，不依赖原业务运行环境。

选择单张 24 GB 或满足配方要求的 NVIDIA GPU，核对 CUDA、驱动、主机内存、共享内存及磁盘空间。训练监控只负责进程与检查点，实例启停由部署者管理。

## 4. Linux 环境和首轮 pilot

新建隔离环境，不在业务环境安装 GPU 包。锁文件针对 Linux x86_64 / Python 3.11，PyTorch 2.6 / vLLM 0.8.5 / verl 0.5.0，GPU 驱动必须与实际 wheel 的 CUDA 运行时兼容。确认 `nvidia-smi`、`torch.cuda.is_available()`、BF16 支持和唯一一张可见 GPU。首阶段有导入或显存问题时先停止，修复环境后再训练。

```sh
uv venv --python 3.11 experiments/qwen3_bonus/.venv-gpu
uv pip install --python experiments/qwen3_bonus/.venv-gpu/bin/python --require-hashes -r experiments/qwen3_bonus/requirements-gpu.lock
experiments/qwen3_bonus/.venv-gpu/bin/python -c 'import torch,verl,vllm; print(torch.__version__,verl.__version__,vllm.__version__,torch.cuda.is_available(),torch.cuda.is_bf16_supported())'
experiments/qwen3_bonus/.venv-gpu/bin/python -m experiments.qwen3_bonus.models verify --kind base
experiments/qwen3_bonus/.venv-gpu/bin/python -m experiments.qwen3_bonus.models verify --kind reward
experiments/qwen3_bonus/.venv-gpu/bin/python -m experiments.qwen3_bonus.training recipe
```

Parquet 已提供；只有缺少时才执行 `python -m experiments.qwen3_bonus.training parquet`，避免覆盖冻结文件。recipe 只打印官方配置，不训练。需要 `/dev/shm` 和足够 CPU/RAM 供 Ray、offload、奖励模型使用；0.6B 奖励模型固定 CPU，降低显存峰值但会拖慢采样循环。

先执行入口帮助检查当前参数，再按已冻结配方运行两步 pilot，验证模型、奖励、日志与导出链路：

```sh
python -m experiments.qwen3_bonus.training grpo --help
```

检查 launch.json、steps.jsonl、rollouts.jsonl 和 training_result.json，记录加载耗时、每步耗时、峰值显存、组内奖励差异及适配器完整性。缺少模型、审核或兼容报告时，入口会拒绝训练。

## 5. 回答 GRPO 与分类 SFT

正式 GRPO 从原指令基座开始，不把两步 pilot 当作已经完成的质量训练，也不把分类 LoRA 混入回答。新 run 目录不可覆盖；断点用于保全当前训练产物，首轮不提供未经测试的跨 run 自动恢复。

```sh
python -m experiments.qwen3_bonus.training grpo --help
python -m experiments.qwen3_bonus.training sft --help
```

历史首轮回答 GRPO 为 32 步，分类 SFT 为 150 个优化步；后续 PERM 训练的 64 步配置见 PERM_GRPO.md。每轮使用独立输出目录，并保留冻结参数与来源指纹。

按配置间隔及正常停止时保存检查点。遇到连续无组内奖励差异或非有限指标时停止；watchdog 用于终止异常挂起进程，非有限训练产物不进入导出流程。结束后校验产物，并由部署者管理实例生命周期。

## 6. 取回、导出和真实双 LoRA 冒烟

将 emotion/response run 完整复制到本机 `artifacts/runs`，包含 training_result、adapter/或 checkpoints、日志和配置。不要只下载配置 JSON。导出验证真实 safetensors、结构和训练基座指纹，再把导出的基座身份规范化为固定 Qwen revision；原训练文件保留不变。

```sh
experiments/qwen3_bonus/.venv-tools/bin/python -m experiments.qwen3_bonus.deployment export \
  --emotion-run experiments/qwen3_bonus/artifacts/runs/emotion \
  --response-run experiments/qwen3_bonus/artifacts/runs/response
experiments/qwen3_bonus/.venv-metal/bin/python -m experiments.qwen3_bonus.probe \
  --config experiments/qwen3_bonus/configs/serving.json --output experiments/qwen3_bonus/artifacts/metal-smoke.json --live
```

两个已训练适配器各自从同一 MLX 基座加载。对照 BF16 训练环境与量化本机相同样例的身份、输出、无效标签及安全问题；BF16 可以在停止 GPU 前只做少量固定输出检查，完整评测放本机。未通过本机真实加载/格式检查就不进入发布评测。

## 7. 本机基座与适配器评测

先独立启动 **不挂适配器** 的同一 MLX 基座，固定身份 solace-base；前台运行，完成捕获后按 Ctrl-C 停止，再启动双 LoRA 网关。

```sh
experiments/qwen3_bonus/.venv-metal/bin/vllm serve "$PWD/experiments/qwen3_bonus/artifacts/models/base-mlx4" \
  --served-model-name solace-base --host 127.0.0.1 --port 18183 \
  --max-model-len 2048 --max-num-seqs 1
```

另开本机终端，从项目根目录捕获全部题目（正式门禁不能用 --limit）：

```sh
experiments/qwen3_bonus/.venv-tools/bin/python -m experiments.qwen3_bonus.evaluation capture --task classification \
  --data experiments/qwen3_bonus/artifacts/data/classification_eval.jsonl --endpoint http://127.0.0.1:18183/v1 \
  --model solace-base --base-manifest experiments/qwen3_bonus/artifacts/models/base-mlx4/bonus_source.json \
  --output experiments/qwen3_bonus/artifacts/eval/class-base.json
experiments/qwen3_bonus/.venv-tools/bin/python -m experiments.qwen3_bonus.evaluation capture --task response \
  --data experiments/qwen3_bonus/artifacts/data/response_eval.jsonl --endpoint http://127.0.0.1:18183/v1 \
  --model solace-base --base-manifest experiments/qwen3_bonus/artifacts/models/base-mlx4/bonus_source.json \
  --output experiments/qwen3_bonus/artifacts/eval/response-base.json
```

停止基座服务，启动训练后网关：

```sh
.venv/bin/python -m experiments.qwen3_bonus.serving --config experiments/qwen3_bonus/configs/serving.json
```

冷启动第一题可能 503，后台继续预热；等待健康就绪后重新用新输出路径捕获，失败不会生成半份合格结果。冻结权重，重导出后必须先重启网关。再捕获：

```sh
.venv/bin/python -m experiments.qwen3_bonus.evaluation capture --task classification \
  --data experiments/qwen3_bonus/artifacts/data/classification_eval.jsonl --endpoint http://127.0.0.1:18181/v1 \
  --model solace-emotion --serving-config experiments/qwen3_bonus/configs/serving.json \
  --output experiments/qwen3_bonus/artifacts/eval/class-adapter.json
.venv/bin/python -m experiments.qwen3_bonus.evaluation capture --task response \
  --data experiments/qwen3_bonus/artifacts/data/response_eval.jsonl --endpoint http://127.0.0.1:18181/v1 \
  --model solace-response --serving-config experiments/qwen3_bonus/configs/serving.json \
  --output experiments/qwen3_bonus/artifacts/eval/response-adapter.json
.venv/bin/python -m experiments.qwen3_bonus.evaluation blind \
  --base experiments/qwen3_bonus/artifacts/eval/response-base.json --adapter experiments/qwen3_bonus/artifacts/eval/response-adapter.json \
  --output experiments/qwen3_bonus/artifacts/eval/blind
```

只给评审看 blind/sheet.json，不看 key.private.json。五个维度分别给 a/b 打 1–5 分，choice 填 a/b/tie，reviewed=true、reviewer 填本人标识、safety_flags 的 a/b 分别填 true/false 标记各答案是否有严重安全问题，保留理由。severe_safety_regression 留给解盲程序计算；只要候选有严重安全问题就禁止发布。不得修改 prompt/a/b。40 题必须全部完成；并列不是胜。dimensions 的每项按 {"a": 评分, "b": 评分} 填写，完整评分是人审完成的条件，门禁再按总体胜负和安全性决定，不伪装成心理学量表。

```sh
.venv/bin/python -m experiments.qwen3_bonus.evaluation release \
  --classification-base experiments/qwen3_bonus/artifacts/eval/class-base.json \
  --classification-adapter experiments/qwen3_bonus/artifacts/eval/class-adapter.json \
  --sheet experiments/qwen3_bonus/artifacts/eval/blind/sheet.json --key experiments/qwen3_bonus/artifacts/eval/blind/key.private.json \
  --serving-config experiments/qwen3_bonus/configs/serving.json --output experiments/qwen3_bonus/artifacts/release-report.json
```

候选 capture 时和 release 时的权重指纹必须相同；未训练基座必须与部署量化基座一致。分类 F1 不降且高风险漏判不增、40 题胜数大于负数且没有新增严重安全问题，才 passed=true。报告不自动开启功能。

40 条 PsyDT 测试集覆盖真实多轮上下文，与项目现有的上下文证据、危机只读和流式中断专项测试互补，共同组成发布前的完整回归验证。

## 8. 性能记录、开启与回滚

采集首字延迟、生成速度与进程 RSS 内存的方法见 `benchmark.py --help`，建议分开记录冷启动（cold）与热请求（warm）基线。

按照 SERVING.md 启用新入口：

```sh
BONUS_LOCAL_FALLBACK_ENABLED=true BONUS_RELEASE_REPORT=experiments/qwen3_bonus/artifacts/release-report.json \
BONUS_SERVING_CONFIG=experiments/qwen3_bonus/configs/serving.json \
.venv/bin/uvicorn qwen3_bonus.asgi:create_app --factory --host 127.0.0.1 --port 8080 --workers 1
```

保持开关关闭或继续使用原入口即可维持默认行为；回滚时关闭开关、重启应用并停止独立网关即可完成切换。

完成测试后将训练步数、模型指纹、400 条分类指标、盲测维度得分及冷/热延迟汇总更新至 `REPORT.md`。

性能采集示例（网关运行时，第一次可能是 10 秒返回 503 的冷请求；保留该报告，模型就绪后换输出文件重新测 warm）：

```sh
experiments/qwen3_bonus/.venv-metal/bin/python -m experiments.qwen3_bonus.benchmark --config experiments/qwen3_bonus/configs/serving.json \
  --output experiments/qwen3_bonus/artifacts/benchmark-cold.json
experiments/qwen3_bonus/.venv-metal/bin/python -m experiments.qwen3_bonus.benchmark --config experiments/qwen3_bonus/configs/serving.json \
  --output experiments/qwen3_bonus/artifacts/benchmark-warm.json
```

报告的速度是对最终文本重新分词的近似值，首字时间从发起本地请求算起；未得到正文时不产生虚构的首字/速度。每 0.5 秒采样引擎进程组 RSS，交换量是 sysctl 返回的系统总量。更精确的峰值/设备分配可另外用 macOS Instruments 测量，并明确采集方法。

## 本次远端自动任务（2026-10-22）

代码已迁至 `experiments/qwen3_bonus/`；旧章节中的 `qwen3_bonus/` 路径与模块名，应分别改为 `experiments/qwen3_bonus/` 和 `experiments.qwen3_bonus`。

用户本次授权了模型评审替代20对人工核对，并要求先在已租GPU执行训练。原始未审核表保留，另存 model-reviewed.json，review_type=model、附每项理由；模型评审不会被统计成人工核对。对应显式参数为 --allow-model-review / --allow-model-calibration。本机 Metal 兼容验证本次延后到发布前（--defer-local-compatibility），功能仍保持关闭，不会自动发布。

历史实例记录为 RTX 4080 SUPER / 32760 MiB、BF16=True、62 GB 主机内存。远端目录 `/root/autodl-tmp/solace-qwen3/experiments/qwen3_bonus`。

训练改为 GRPO 每4步保存小型适配器快照，保留每个快照，完整FSDP检查点只保留最近一个以避免占满50GB数据盘。分类每16步存档，保留全部小型适配器快照及最近两个Trainer检查点。每次存档和最终退出都从真实日志画图，输出PNG/SVG；TensorBoard标量逐步写入。没有日志的指标显示Not logged，不补造曲线。

`remote_pipeline.py` 串联奖励评分/门禁、2步pilot、最多32步GRPO、分类SFT。各阶段命令、起止、退出原因记录在 artifacts/remote；不替换失败的算法为另一套方法。环境/模型准备超过运行时限、奖励排序未过线或训练报错，任务保存失败记录并结束。

TensorBoard 只监听远端 127.0.0.1:6006，通过 SSH 隧道在本机 http://127.0.0.1:16006 查看。日志与模型产物校验后由部署者管理实例启停。
