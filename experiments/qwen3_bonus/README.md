# SolaceAI：Qwen3 双 LoRA 算法实验

这是独立于原业务的“借鉴心理学评价维度的小规模 GRPO 双 LoRA 实验”。所有新增代码、数据、虚拟环境和产物都在本目录，原入口与原配置无需修改。默认关闭新兜底。

分类 LoRA 和回答 LoRA 已完成真实训练、回传与 Metal 验收。本机回答降级采用 **PERM 启发式 GRPO 第64步最终快照** ，对应 `artifacts/adapters/solace-response-perm-grpo64/`；完整单次训练共64步。四模型80题统一评测后用于最终部署验收。本机入口为 http://127.0.0.1:8082 。

最新结果见[四模型完整评测与部署选择](docs/PERM_EVALUATION.md)：第64步四维质量均分达到 **3.253** ，在本轮四模型评测中位居第一（高于原始基座 3.169、GRPO32 3.212 与 GRPO32+SFT 3.203）；对原始基座偏好得分率达 **56.25%** （27胜36平17负）；在核心倾听与心理陪伴 40 题场景中，语义完成率达 50.0%（20/40）、四维均分达 3.125，在高共情峰值回答上优势突出。

| 内容 | 当前状态 |
|---|---|
| 分类 | 1200条训练，150步NF4 QLoRA-SFT；400条分类回归准确率 98.75%→99.75% |
| 回答 | 本轮原生GRPO 64步完整终态快照，四维均分3.253（参评模型最高），对基座偏好得分率56.25%；原GRPO32与GRPO32＋SFT保留对照 |
| 奖励审核 | 1.7B CPU FP32校准19/20、任务交付8/8；0.6B为20/20、8/8 |
| 服务 | 双静态LoRA、单请求串行、10秒冷启动等待、5分钟空闲卸载；真实Metal验收已通过 |
| 接入 | 独立本机候选入口替换分类，仅云端失败时调用回答LoRA；原入口保留 |

本机部署方案与接续记录见 [部署计划](docs/LOCAL_DEPLOY_PLAN.md)。

2026-10-23 阶段补充：快速能力对比已完成 168 份初评、34 份换序复评及 128 次本机补测，本机回答 LoRA 在 24 题中对基座取得 15 胜 9 负，倾听类 4/4 全胜。

阅读顺序：

1. [完整操作手册](docs/RUNBOOK.md)：本机准备、训练环境、导出、盲评、开启及回滚。
2. [算法与论文讲解](docs/LEARNING.md)：LoRA、GRPO、奖励、论文借鉴边界及实现边界。
3. [数据来源与去重](docs/DATA.md)：版本、许可、合成性质、切分和泄漏检查。
4. [服务与项目接入](docs/SERVING.md)：冷启动、SSE、工具边界和显式入口。
5. [本轮完整评测](docs/PERM_EVALUATION.md)：四模型80题、逐题评分、开发选模、资源与部署选择；[历史实验报告](docs/REPORT.md)保留先前NLL及快速评测。

从项目根目录执行本地验证：

```sh
.venv/bin/python -m pytest experiments/qwen3_bonus/tests -q
.venv/bin/python -m experiments.qwen3_bonus.training recipe
.venv/bin/python -m experiments.qwen3_bonus.probe --config experiments/qwen3_bonus/configs/serving.example.json --output experiments/qwen3_bonus/artifacts/prerequisites.json
```

模型依赖隔离在 `.venv-tools` / `.venv-metal` / `.venv-gpu`，不要安装到原 `.venv`。`requirements-gpu.lock` 锁定 Linux Python 3.11 的依赖和哈希；训练实测与本机部署验收分别记录。当前部署与训练使用隔离环境。

文件对应：`data.py` 数据，`calibration.py` 奖励核对，`training.py` 训练监控和启动，`verl_entry.py` / `sft_entry.py` 训练，`deployment.py` MLX 转换与适配器导出，`compatibility.py` 训练前零适配器兼容试验，`evaluation.py` 捕获/盲评/门禁，`serving.py` 独立网关，`integration.py` / `asgi.py` 显式接入。
