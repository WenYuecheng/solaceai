"""SolaceAI 关系型数据库持久化数据模型包 (SQLAlchemy ORM Entities)

本数据模型包提供遵循 SQLAlchemy 2.0 声明式规范的全部实体定义：
- 身份与安全：`UserAccount` (用户与 RBAC 权限), `SessionVisibility` (会话归属权限)；
- 会话与消息：`ChatSession` (会话上下文), `ChatMessage` (消息流), `ChatMessageRun` (执行绑定)；
- 心理测评与危机：`PsychologicalReport` (评估报告), `RiskCase` (危机个案), `CaseNote` (人工跟进), `AlertRecord` (预警告警)；
- 知识多版本存储：`KnowledgeChunk` (文本切片), `KnowledgeChunkMetadata` (切片元数据), `KnowledgeDocumentVersion` (文档版本)；
- 任务与可靠性：`ToolJob` (异步作业), `DeadLetterRecord` (死信隔离), `ExcelRecord` (线下台账)；
- 审计与度量：`AgentRunTrace` (链路追踪), `ToolAuditRecord` (工具策略审计), `ToolExecution` (幂等执行), `AnswerFeedback` (反馈度量), `RunMetrics` (延迟度量), `AgentStoreItem` (持久 KV 记忆)。
"""
