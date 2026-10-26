# SolaceAI：Qwen3 双 LoRA 算法实验

这是独立于原业务的“借鉴心理学评价维度的小规模 GRPO 双 LoRA 实验”。所有新增代码、数据、虚拟环境和产物都在本目录，原入口与原配置无需修改。默认关闭新兜底。

分类 LoRA 和回答 LoRA 已完成真实训练、回传与 Metal 验收。本机回答降级采用 **PERM 启发式 GRPO 第64步最终快照**，对应 `artifacts/adapters/solace-response-perm-grpo64/`；完整单次训练共64步。四模型80题统一评测后用于最终部署验收。本机入口为 http://127.0.0.1:8082 。

最新结果见[四模型完整评测与部署选择](docs/PERM_EVALUATION.md)：第64步四维质量均分达到 **3.253**，全面超越原始基座（3.169）、GRPO32（3.212）和原GRPO32＋SFT（3.203）；对原始基座偏好胜率达 **56.25%**（27胜36平17负）；在核心倾听与心理陪伴40题场景中，语义完成率达50.0%（20/40）、四维均分达3.125，在高共情峰值回答上优势突出。这是匿名模型评审，不声称显著优于或临床验证。80题用于四个固定候选的最终服务选择。

| 内容 | 当前状态 |
|---|---|
| 分类 | 1200条训练，150步NF4 QLoRA-SFT；400条合成回归准确率98.75%→99.75% |
| 回答 | 本轮原生GRPO 64步完整终态快照，四维均分3.253（参评模型最高），对基座偏好胜率56.25%；原GRPO32与GRPO32＋SFT保留对照 |
| 奖励审核 | 1.7B CPU FP32校准19/20、任务交付8/8；0.6B为20/20、8/8；均为模型审核 |
| 服务 | 双静态LoRA、单请求串行、10秒冷启动等待、5分钟空闲卸载；真实Metal验收已通过 |
| 接入 | 独立本机候选入口替换分类，仅云端失败时调用回答LoRA；原入口保留 |
| 费用 | 本轮估算约6.45元，含前期与旧尝试预留4元；平台已确认关机，账单未核销；完整对比在本机完成，无付费API调用 |

当前接续状态见 [实施记录](WORKLOG.md)，本机操作见 [部署计划](docs/LOCAL_DEPLOY_PLAN.md)。

2026-10-23补充：[实际能力快速评测](artifacts/quick-eval-20261023/REPORT.md)已完成168份初评、34份换序复评及128次本机补测；当前回答LoRA在24题中对基座15胜9负，倾听类4/4胜，多轮交付和引用仍有弱项。这是模型评审，人工与临床验证状态不变。

阅读顺序：

1. [完整操作手册](docs/RUNBOOK.md)：本机准备、租卡、停止计费、导出、盲评、开启及回滚。
2. [算法与论文讲解](docs/LEARNING.md)：LoRA、GRPO、奖励、论文借鉴边界及面试表述。
3. [数据来源与去重](docs/DATA.md)：版本、许可、合成性质、切分和泄漏检查。
4. [服务与项目接入](docs/SERVING.md)：冷启动、SSE、工具边界和显式入口。
5. [本轮完整评测](docs/PERM_EVALUATION.md)：四模型80题、逐题评分、开发选模、资源与部署选择；[历史实验报告](docs/REPORT.md)保留先前NLL及快速评测。

从项目根目录执行免费验证：

```sh
.venv/bin/python -m pytest experiments/qwen3_bonus/tests -q
.venv/bin/python -m experiments.qwen3_bonus.training recipe
.venv/bin/python -m experiments.qwen3_bonus.probe --config experiments/qwen3_bonus/configs/serving.example.json --output experiments/qwen3_bonus/artifacts/prerequisites.json
```

模型依赖隔离在 `.venv-tools` / `.venv-metal` / `.venv-gpu`，不要安装到原 `.venv`。`requirements-gpu.lock` 锁定 Linux Python 3.11 的依赖和哈希；训练实测与本机部署验收分别记录。当前只完成本机接续，不新增租卡或充值。

文件对应：`data.py` 数据，`calibration.py` 奖励核对，`training.py` 预算和启动，`verl_entry.py` / `sft_entry.py` 训练，`deployment.py` MLX 转换与适配器导出，`compatibility.py` 租卡前零适配器兼容试验，`evaluation.py` 捕获/盲评/门禁，`serving.py` 独立网关，`integration.py` / `asgi.py` 显式接入。
