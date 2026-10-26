import { describe, expect, it } from 'vitest'
import { createSseParser } from '@/core/sse'

describe('SSE parser', () => {
  it('accepts citations, progress and completion events', () => {
    const events: unknown[] = []
    const parser = createSseParser(event => events.push(event))
    parser.push('data: {"type":"status","message":"正在整理依据"}\n\ndata: {"type":"citation","citations":[{"source":"sleep.md","chunkId":"c1","preview":"固定起床时间"}]}\n\ndata: {"type":"done"}\n\n')
    expect(events).toHaveLength(3)
  })
  it('reassembles fragmented events and ignores malformed payloads', () => {
    const events: unknown[] = []
    const parser = createSseParser((event) => events.push(event))

    parser.push('data: {"type":"meta","sessionId":"s-1"}\n')
    parser.push('\ndata: not-json\n\ndata: {"type":"token","content":"你好"}\n\n')

    expect(events).toEqual([
      { type: 'meta', sessionId: 's-1' },
      { type: 'token', content: '你好' },
    ])
  })

  it('flushes a final event without a trailing separator', () => {
    const events: unknown[] = []
    const parser = createSseParser((event) => events.push(event))
    parser.push('data: {"type":"error","message":"断开"}')
    parser.flush()
    expect(events).toEqual([{ type: 'error', message: '断开' }])
  })
})

it('keeps ReAct metadata on streamed steps and rejects malformed rounds', () => {
  const events: unknown[] = []
  const parser = createSseParser(event => events.push(event))
  const step = { id: 's', sequence: 1, stage: 'agent', title: '决策摘要', why: '选择动作', result: '', startedAt: '2026-10-21T00:00:00Z', durationMs: null, status: 'running', phase: 'reason', actor: 'MainAgent', round: 1, nextAction: 'search_knowledge' }
  parser.push(`data: ${JSON.stringify({ type: 'step', runId: 'r', sessionId: 's', step })}\n\n`)
  parser.push(`data: ${JSON.stringify({ type: 'step', runId: 'r', sessionId: 's', step: { ...step, round: 'invalid' } })}\n\n`)
  expect(events).toHaveLength(1)
  expect(events[0]).toMatchObject({ step: { phase: 'reason', round: 1, actor: 'MainAgent' } })
})
