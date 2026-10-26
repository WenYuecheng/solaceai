import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useAuthStore } from '@/stores/auth'
import { useChatStore } from '@/stores/chat'

function streamResponse(chunks: string[]): Response {
  const encoder = new TextEncoder()
  const stream = new ReadableStream({
    start(controller) {
      chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)))
      controller.close()
    },
  })
  return new Response(stream, { status: 200 })
}

describe('chat store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    useAuthStore().acceptSession('token', {
      id: 1, username: 'student', displayName: '学生', roles: [],
    })
  })

  it('streams a response into a single assistant message', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(streamResponse([
      'data: {"type":"meta","sessionId":"s-1"}\n\n',
      'data: {"type":"token","content":"你好"}\n\n',
      'data: {"type":"token","content":"，我在。"}\n\n',
    ])))
    const chat = useChatStore()

    await chat.send('最近很累')

    expect(chat.sessionId).toBe('s-1')
    expect(chat.phase).toBe('done')
    expect(chat.messages.map((item) => [item.role, item.content])).toEqual([
      ['user', '最近很累'],
      ['assistant', '你好，我在。'],
    ])
  })

  it('replaces provisional streamed text with the final persisted answer', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(streamResponse([
      'data: {"type":"meta","sessionId":"s","runId":"run-final"}\n\n',
      'data: {"type":"token","content":"暂定回答[99]"}\n\n',
      'data: {"type":"replace","content":"最终回答[1]"}\n\n',
      'data: {"type":"done","messageId":42,"runId":"run-final"}\n\n',
    ])))
    const chat = useChatStore()
    await chat.send('问题')
    expect(chat.messages.at(-1)?.content).toBe('最终回答[1]')
    expect(chat.messages.at(-1)?.messageId).toBe(42)
    expect(chat.messages.at(-1)?.runId).toBe('run-final')
    expect(chat.messages).toHaveLength(2)
    expect(chat.phase).toBe('done')
  })

  it('preserves source evidence on the assistant message', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(streamResponse([
      'data: {"type":"citation","citations":[{"source":"sleep.md","chunkId":"c1","preview":"固定起床时间"}]}\n\n',
      'data: {"type":"token","content":"先固定起床时间"}\n\n',
      'data: {"type":"done"}\n\n',
    ])))
    const chat = useChatStore()
    await chat.send('睡眠不好')
    expect(chat.messages.at(-1)?.citations?.[0].source).toBe('sleep.md')
  })

  it('cancels the active request without adding an error message', async () => {
    vi.stubGlobal('fetch', vi.fn((_url, options: RequestInit) => new Promise((_resolve, reject) => {
      options.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
    })))
    const chat = useChatStore()
    const pending = chat.send('请停一下')
    chat.stop()
    await pending
    expect(chat.phase).toBe('idle')
    expect(chat.error).toBe('')
  })

  it('turns a streamed error event into a retryable error state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(streamResponse([
      'data: {"type":"error","message":"模型暂时不可用"}\n\n',
    ])))
    const chat = useChatStore()
    await chat.send('你好')
    expect(chat.phase).toBe('error')
    expect(chat.error).toBe('模型暂时不可用')
  })
})

describe('conversation switching', () => {
  beforeEach(() => { setActivePinia(createPinia()) })
  it('loads history and continues the selected public session', async () => {
    const fetch = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({ sessionId: 'saved', messages: [{ id: '1', role: 'user', content: '之前' }, { id: '2', role: 'assistant', content: '回答' }] }))).mockResolvedValueOnce(streamResponse(['data: {"type":"token","content":"继续"}\n\n']))
    vi.stubGlobal('fetch', fetch)
    const chat = useChatStore()
    await chat.loadSession('saved')
    expect(chat.messages.map(m => m.role)).toEqual(['user', 'assistant'])
    await chat.send('下一步')
    expect(JSON.parse(fetch.mock.calls[1][1].body).sessionId).toBe('saved')
  })
  it('ignores old stream metadata after reset even if transport ignores abort', async () => {
    let resolve!: (response: Response) => void
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(r => { resolve = r })))
    const chat = useChatStore()
    const pending = chat.send('old')
    chat.reset()
    resolve(streamResponse(['data: {"type":"meta","sessionId":"old-session"}\n\n']))
    await pending
    expect(chat.sessionId).toBeNull()
    expect(chat.messages).toEqual([])
    expect(chat.phase).toBe('idle')
  })
  it('ignores a history response superseded by a new conversation', async () => {
    let resolve!: (response: Response) => void
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(r => { resolve = r })))
    const chat = useChatStore()
    const pending = chat.loadSession('old')
    chat.reset()
    resolve(new Response(JSON.stringify({ sessionId: 'old', messages: [] })))
    await pending
    expect(chat.sessionId).toBeNull()
  })
})

describe('public answer trace', () => {
  beforeEach(() => { setActivePinia(createPinia()) })
  const step = { id: 'step-1', sequence: 1, stage: 'retrieval', title: '检索参考资料', why: '寻找依据', result: '', status: 'running', startedAt: '2026-10-21T00:00:00Z', durationMs: null }
  it('updates a step in place and binds its trace to the final answer', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(streamResponse([
      `data: ${JSON.stringify({ type: 'step', runId: 'r', sessionId: 's', step })}\n\n`,
      `data: ${JSON.stringify({ type: 'step', runId: 'r', sessionId: 's', step: { ...step, status: 'completed', result: '找到资料', count: 2, durationMs: 45 } })}\n\n`,
      'data: {"type":"done","messageId":42,"runId":"r","traceStatus":"completed"}\n\n',
    ])))
    const chat = useChatStore()
    await chat.send('问题')
    expect(chat.messages.at(-1)?.answerTrace).toMatchObject({ runId: 'r', messageId: 42, status: 'completed' })
    expect(chat.messages.at(-1)?.answerTrace?.steps).toHaveLength(1)
    expect(chat.messages.at(-1)?.answerTrace?.steps[0]).toMatchObject({ status: 'completed', durationMs: 45 })
  })
  it('keeps cancelled trace visible and terminates running steps', async () => {
    let streamController!: ReadableStreamDefaultController<Uint8Array>
    const stream = new ReadableStream<Uint8Array>({ start(controller) { streamController = controller } })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(stream)))
    const chat = useChatStore()
    const pending = chat.send('问题')
    await vi.waitFor(() => expect(chat.phase).toBe('thinking'))
    streamController.enqueue(new TextEncoder().encode(`data: ${JSON.stringify({ type: 'step', runId: 'r', sessionId: 's', step })}\n\n`))
    await vi.waitFor(() => expect(chat.messages.at(-1)?.answerTrace?.steps).toHaveLength(1))
    chat.stop()
    streamController.close()
    await pending
    expect(chat.messages.at(-1)?.answerTrace?.status).toBe('cancelled')
    expect(chat.messages.at(-1)?.answerTrace?.steps[0].status).toBe('cancelled')
  })
  it('marks unfinished steps failed when the transport ends without a terminal event', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(streamResponse([
      `data: ${JSON.stringify({ type: 'step', runId: 'r', sessionId: 's', step })}\n\n`,
    ])))
    const chat = useChatStore()
    await chat.send('问题')
    expect(chat.phase).toBe('error')
    expect(chat.messages.at(-1)?.answerTrace?.steps[0].status).toBe('failed')
  })
})

describe('per-turn model backend', () => {
  beforeEach(() => { setActivePinia(createPinia()) })
  it('sends the selected backend and retains it when a failed turn is retried', async () => {
    const { useSystemStore } = await import('@/stores/system')
    const system = useSystemStore()
    system.agent = {
      provider: 'deepseek', model: 'deepseek-chat', realModelEnabled: true,
      availableChatBackends: ['cloud', 'local'], defaultChatBackend: 'cloud',
    }
    system.selectChatBackend('local')
    const fetch = vi.fn()
      .mockResolvedValueOnce(streamResponse(['data: {"type":"error","message":"暂时不可用"}\n\n']))
      .mockResolvedValueOnce(streamResponse(['data: {"type":"done"}\n\n']))
    vi.stubGlobal('fetch', fetch)
    const chat = useChatStore()
    await chat.send('你好')
    system.selectChatBackend('cloud')
    await chat.retry()
    const requests = fetch.mock.calls.map(call => JSON.parse(call[1].body))
    expect(requests.map(request => request.modelBackend)).toEqual(['local', 'local'])
    expect(requests[0].runId).toBe(requests[1].runId)
    expect(system.chatBackend).toBe('local')
  })
  it('defaults a new browser to cloud', async () => {
    const fetch = vi.fn().mockResolvedValue(streamResponse(['data: {"type":"done"}\n\n']))
    vi.stubGlobal('fetch', fetch)
    await useChatStore().send('你好')
    expect(JSON.parse(fetch.mock.calls[0][1].body).modelBackend).toBe('cloud')
  })
})
