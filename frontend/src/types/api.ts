export interface Role { authority: string }

export interface Profile {
  id: number
  username: string
  email?: string | null
  requiresSetup?: boolean
  displayName: string
  roles: Role[]
}

export type ChatBackend = 'cloud' | 'local'
export interface ChatModel { provider: string; model: string; label?: string }

export interface AgentStatus {
  provider: string
  model: string
  realModelEnabled: boolean
  runtimeProfile?: string
  availableChatBackends?: ChatBackend[]
  defaultChatBackend?: ChatBackend
  chatModels?: Partial<Record<ChatBackend, ChatModel>>
}

export interface Report {
  id: number
  sessionId: string
  username: string
  displayName: string
  content: string
  intent: string
  emotion: string
  emotionScore: number
  riskLevel: string
  summary: string
  createdAt: string
}

export interface RiskCase {
  id: number
  reportId: number
  riskLevel: string
  status: string
  owner: string
  summary: string
  handoffSummary: string
  acknowledgedBy?: string | null
  acknowledgedAt?: string | null
  createdAt: string
  updatedAt: string
}

export interface ToolRecord {
  id: number
  reportId: number
  status: string
  message: string
  createdAt: string
  channel?: string | null
  recipient?: string | null
  filePath?: string | null
}

export interface ConversationMessage { role: string; content: string; createdAt: string }
export interface Conversation { sessionId: string; title: string; messages: ConversationMessage[] }

export interface KnowledgeStatus {
  backend?: 'chroma_bm25' | 'milvus_bge'
  databaseChunks: number
  vectorAvailable: boolean
  vectorChunks?: number
  [key: string]: unknown
}

export interface EvidenceCitation { source: string; chunkId?: string | number; preview?: string; title?: string; sectionPath?: string[]; cited?: boolean }

export type AnswerTraceStatus = 'running' | 'completed' | 'skipped' | 'failed' | 'degraded' | 'cancelled'
export type AnswerTracePhase = 'prepare' | 'reason' | 'act' | 'observe' | 'answer' | 'review' | 'deliver'
export interface RetrievalDetail {
  channel: string
  queryIndex: number
  method: string
  total: number
  candidates: { rank: number; chunkId: number | string; source: string; content: string; score: number }[]
}
export interface AnswerTraceStep {
  retrievalDetails?: RetrievalDetail[]
  phase?: AnswerTracePhase
  actor?: string
  round?: number
  tool?: string
  nextAction?: string
  id: string
  sequence: number
  stage: string
  title: string
  why: string
  result: string
  status: AnswerTraceStatus
  startedAt: string
  durationMs: number | null
  count?: number
}
export interface AnswerTrace {
  runId: string
  messageId?: number
  status: AnswerTraceStatus
  steps: AnswerTraceStep[]
}

export type ChatStreamEvent =
  | { type: 'step'; runId: string; sessionId: string; step: AnswerTraceStep }
  | { type: 'meta'; sessionId?: string; runId?: string }
  | { type: 'token'; content?: string }
  | { type: 'replace'; content: string }
  | { type: 'error'; message?: string }
  | { type: 'citation'; citations: EvidenceCitation[] }
  | { type: 'status'; message?: string }
  | { type: 'done'; messageId?: number; runId?: string; traceStatus?: AnswerTraceStatus }

export interface AuthSession {
  token: string
  username: string
  displayName: string
  roles: Role[]
}
