/**
 * 回答链路追踪阶段与执行主体映射模块 (Answer Trace Phases & Actors).
 *
 * 将后端多智能体及工具执行步骤映射为标准可视化阶段 (Prepare/Reason/Act/Observe/Answer/Review/Deliver)
 * 以及对应负责组件 (MainAgent, RetrievalAgent, SafetyAgent, MailService 等)。
 */
import type { AnswerTracePhase, AnswerTraceStep } from '@/types/api'

/**
 * 链路阶段标准元数据配置表（展示标题与中文说明）
 */
export const tracePhases: Record<AnswerTracePhase, { label: string; description: string }> = {
  prepare: { label: 'Prepare', description: '输入与上下文准备' },
  reason: { label: 'Reason', description: '决策摘要' },
  act: { label: 'Act', description: '执行动作' },
  observe: { label: 'Observe', description: '观察结果' },
  answer: { label: 'Answer', description: '生成与接收回答' },
  review: { label: 'Review', description: '回复检查' },
  deliver: { label: 'Deliver', description: '记录与通知' },
}

/** 内部 stage 到大阶段 phase 的兜底映射表 */
const stagePhases: Record<string, AnswerTracePhase> = {
  agent: 'reason', retrieval_policy: 'reason', observation: 'observe',
  recall: 'act', fusion: 'act', evidence_selection: 'act', retrieval: 'act', rerank: 'act', memory_search: 'act', memory_save: 'act', memory_delete: 'act', evidence_read: 'act', skill: 'act', tool: 'act',
  answer: 'answer', answer_ready: 'answer', review: 'review', report: 'deliver', case: 'deliver', email: 'deliver',
}

/** 内部 stage 到执行主体组件 actor 的默认映射表 */
const stageActors: Record<string, string> = {
  agent: 'MainAgent', observation: 'MainAgent', answer: 'MainAgent', answer_ready: 'MainAgent',
  recall: 'RetrievalAgent', fusion: 'RetrievalAgent', evidence_selection: 'RetrievalAgent', retrieval: 'RetrievalAgent', rerank: 'RetrievalAgent', review: 'SafetyAgent', risk: 'SafetyAgent',
  memory_search: 'MainAgent', memory_save: 'MainAgent', memory_delete: 'MainAgent', evidence_read: 'MainAgent', skill: 'MainAgent',
  report: 'ReportService', case: 'CaseService', email: 'MailService',
}

/**
 * 计算或提取追踪步骤对应的大阶段类型。
 *
 * @param step - 步骤对象。
 * @returns 阶段类型枚举值（如 reason, act, answer 等）。
 */
export function phaseForStep(step: AnswerTraceStep): AnswerTracePhase {
  return step.phase && Object.hasOwn(tracePhases, step.phase) ? step.phase : stagePhases[step.stage] ?? 'prepare'
}

/**
 * 计算或提取追踪步骤对应的执行主体名称。
 *
 * @param step - 步骤对象。
 * @returns 主体名称（如 MainAgent, SafetyAgent, Runtime 等）。
 */
export function actorForStep(step: AnswerTraceStep): string {
  return step.actor || stageActors[step.stage] || 'Runtime'
}
