# SolaceAI 评测数据集与评测基准资产库 (Evaluation Datasets & Benchmarks)

本目录集中管理 SolaceAI 系统的所有离线评测数据集、人工专家标注资产与基准打分产物。

> [!NOTE]
> 评测入口统一收敛于 `evaluation/suites/`，与业务运行时完全解耦。每次评测快照自动记录源码、业务实现与技能指纹，确保多轮实验结果可严格追溯对比。

---

## 目录组织结构

```
evaluation/
├── suites/                       # 评测体系 Python 执行源码套件 (RAGAS、检索回归、打分裁判)
├── configs/                      # 评测框架与工具配置文件 (Promptfoo 等)
│   └── promptfoo.yaml
├── data/                         # 离线评测基准数据集
│   ├── development-cases.json    # 核心开发集用例
│   ├── development-qrels.json    # 冻结标注相关度真值
│   ├── cpsycoune-subset.json     # 心理咨询真实子集测试例
│   ├── glm_quality_cases.json    # 深度语义质量评估测试集
│   ├── quality_rag_cases.json    # RAG 增强检索测试用例
│   └── drafts/                   # 历史与草稿用例集
├── annotations/                  # [统一收拢] 人工专家双盲标注资产
│   ├── 标注指南.md               # 心理危机与干预质量人工评估标准
│   ├── check_annotations.py      # 标注完整性与一致性校验工具
│   ├── 标注模板/                 # 语料切片清单与问答模板
│   ├── 标注结果/                 # 123 题专家人工复核打分结果
│   └── 参考文件/                 # 心理评估参考规范
└── results/                      # 评测运行产物与对比报表
    ├── final/                    # 终态本地与云端评测结果快照
    ├── live/                     # 在线实时联调评测输出
    └── development-report.json   # 开发期全量回归评测报表
```
