"""SolaceAI 数据传输对象与模式定义包 (Schemas & Data Transfer Objects)

本模式包集中定义前后端与内部服务间交互的 Pydantic V2 契约对象：
- 对话交互：`ChatRequest` (对话请求), `ChatStreamEvent` (SSE 流式响应事件)；
- 知识库运维：`KnowledgeIngestRequest` (文档摄入请求), `KnowledgeIngestResponse` (摄入结果)；
- 报告与个案：`ReportResponse` (评估报告详情), `RiskCaseResponse` (危机个案追踪), `CaseNoteResponse` (人工备注)；
- 作业与死信：`ToolJobResponse` (异步作业状态), `DeadLetterResponse` (死信排查记录)；
- 审计与追踪：`AgentRunTraceResponse` (智能体决策全景), `ToolAuditResponse` (工具安全治理)；
- 评价与度量：`AnswerFeedbackRequest` (问答满意度反馈), `AiMessage` (统一大模型消息格式)。
"""
