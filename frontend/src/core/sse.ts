/**
 * Server-Sent Events (SSE) 流式解析器模块 (SSE Stream Parser).
 *
 * 负责解析长连接返回的 text/event-stream 数据块，按标准事件边界拆分，
 * 严格校验事件数据形态并分发至客户端流式事件回调。
 */
import type { ChatStreamEvent } from '@/types/api'

/**
 * 校验解析出的 JSON 对象是否符合合法的对话流事件 (ChatStreamEvent) 契约。
 *
 * @param value - 待校验的原始反序列化对象。
 * @returns 是否为有效的 ChatStreamEvent 类型守卫。
 */
function isChatStreamEvent(value: unknown): value is ChatStreamEvent {
  if (!value || typeof value !== 'object') return false
  const type = (value as { type?: unknown }).type
  if (type === 'step') {
    const event = value as Record<string, unknown>
    if (typeof event.runId !== 'string' || typeof event.sessionId !== 'string' || !event.step || typeof event.step !== 'object') return false
    const step = event.step as Record<string, unknown>
    return ['id', 'stage', 'title', 'why', 'result', 'startedAt'].every(key => typeof step[key] === 'string')
      && typeof step.sequence === 'number' && Number.isFinite(step.sequence)
      && (step.durationMs === null || (typeof step.durationMs === 'number' && Number.isFinite(step.durationMs)))
      && ['running', 'completed', 'skipped', 'failed', 'degraded', 'cancelled'].includes(String(step.status))
      && (step.phase === undefined || ['prepare', 'reason', 'act', 'observe', 'answer', 'review', 'deliver'].includes(String(step.phase)))
      && (step.round === undefined || (typeof step.round === 'number' && Number.isInteger(step.round) && step.round >= 1 && step.round <= 10000))
      && ['actor', 'tool', 'nextAction'].every(key => step[key] === undefined || (typeof step[key] === 'string' && step[key].length <= 64))
  }
  if (type === 'replace') return typeof (value as { content?: unknown }).content === 'string'
  if (type === 'citation') return Array.isArray((value as { citations?: unknown }).citations)
  return type === 'meta' || type === 'token' || type === 'error' || type === 'status' || type === 'done'
}

/**
 * 构建带分块缓冲区的流式事件解析器。
 *
 * @param onEvent - 单个合法事件到达时的处理回调函数。
 * @returns 包含 push（写入数据块）与 flush（冲刷缓冲区）方法的解析器实例。
 */
export function createSseParser(onEvent: (event: ChatStreamEvent) => void) {
  let buffer = ''

  /**
   * 解析单个由空行分隔的 SSE 数据块。
   *
   * @param block - 双换行符分隔的原始文本块。
   */
  function parseBlock(block: string) {
    const data = block
      .split(/\r?\n/)
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart())
      .join('\n')
    if (!data) return
    let event: unknown
    try {
      event = JSON.parse(data)
    } catch {
      // 单条格式异常的服务端事件不得中断整个剩余数据流的处理
      return
    }
    if (isChatStreamEvent(event)) onEvent(event)
  }

  return {
    /**
     * 将网络收到的文本块追加至内部缓冲区并触发完整块解析。
     *
     * @param chunk - 本次收到的流式文本片段。
     */
    push(chunk: string) {
      buffer += chunk
      const blocks = buffer.split(/\r?\n\r?\n/)
      buffer = blocks.pop() ?? ''
      blocks.forEach(parseBlock)
    },
    /**
     * 冲刷缓冲区中剩余的尾部数据（流结束时调用）。
     */
    flush() {
      if (buffer.trim()) parseBlock(buffer)
      buffer = ''
    },
  }
}
