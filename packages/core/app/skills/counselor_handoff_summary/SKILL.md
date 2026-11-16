---
name: counselor_handoff_summary
description: 在已有授权的工作人员交接场景中，整理必要、客观的安全跟进信息。
---

# 在已有授权的工作人员交接场景中，整理必要、客观的安全跟进信息。

## 执行规则

- 只用于获授权的工作人员交接，不把后台标签显示给学生。
- 仅纳入已提供的报告ID、身份、风险信息、事实摘要和必要原话；未知字段写未知。
- 明确区分学生表达、模型判断、核实事实和已执行行动，不虚构诊断、位置或置信度。
- 首项跟进关注当前安全、能否联系和是否有人陪伴；不保留不必要危险细节。
- 不因命中Skill而取得通知、发送、导出或查询他人隐私的权限。

版本：source-v1。以上为项目原创执行规则，外部研究来源及许可见 docs/architecture.md#知识来源与许可。

## 交接模板

```text
应用 skill: counselor_handoff_summary
报告ID：{{report_id}}
学生：{{student}}
风险等级：{{risk_level}}
情绪标签：{{emotion}}
模型摘要：{{summary}}
建议跟进：
{{next_steps}}
学生原始表达：
{{content_excerpt}}
```
