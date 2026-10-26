import { reactive, ref, shallowRef } from 'vue'
import { defineStore } from 'pinia'
import { request } from '@/core/api'
import { createSseParser } from '@/core/sse'
import { useSystemStore } from '@/stores/system'
import type { AnswerTrace, AnswerTraceStatus, ChatBackend, EvidenceCitation } from '@/types/api'

export type ChatRole = 'user' | 'assistant'
export type ChatPhase = 'idle' | 'thinking' | 'streaming' | 'done' | 'error'
export interface ChatMessage { id: string; role: ChatRole; content: string; citations?: EvidenceCitation[]; status?: string; runId?: string; messageId?: number; answerTrace?: AnswerTrace | null }

export const useChatStore = defineStore('chat', () => {
  const sessionId = shallowRef<string | null>(null)
  const pendingSessionId = shallowRef<string | null>(null)
  const messages = ref<ChatMessage[]>([])
  const phase = shallowRef<ChatPhase>('idle')
  const error = shallowRef('')
  const lastUserMessage = shallowRef('')
  const lastRunId = shallowRef('')
  let lastModelBackend: ChatBackend = 'cloud'
  let controller: AbortController | null = null
  let generation = 0

  async function send(input: string, reuseRunId?: string, reuseBackend?: ChatBackend) {
    const content = input.trim()
    if (!content || controller) return
    lastUserMessage.value = content
    lastRunId.value = reuseRunId || crypto.randomUUID().replaceAll('-', '')
    lastModelBackend = reuseBackend ?? useSystemStore().chatBackend
    error.value = ''
    phase.value = 'thinking'
    messages.value.push({ id: crypto.randomUUID(), role: 'user', content })
    const assistant = reactive<ChatMessage>({ id: crypto.randomUUID(), role: 'assistant', content: '' })
    messages.value.push(assistant)
    let receivedDone = false
    const currentGeneration = ++generation
    const activeController = new AbortController()
    controller = activeController

    try {
      const response = await request('/api/chat/stream', {
        method: 'POST',
        signal: activeController.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId: sessionId.value, message: content, runId: lastRunId.value, modelBackend: lastModelBackend }),
      })
      if (currentGeneration !== generation) return
      if (!response.body) throw new Error('浏览器无法读取流式回复')
      const parser = createSseParser((event) => {
        if (currentGeneration !== generation) return
        if (event.type === 'step') {
          if (assistant.runId && assistant.runId !== event.runId) return
          assistant.runId = event.runId
          sessionId.value = event.sessionId
          assistant.answerTrace ??= { runId: event.runId, status: 'running', steps: [] }
          const steps = assistant.answerTrace.steps
          const index = steps.findIndex(step => step.id === event.step.id)
          if (index < 0) steps.push(event.step)
          else steps[index] = event.step
          steps.sort((a, b) => a.sequence - b.sequence)
        }
        if (event.type === 'meta') { if (event.sessionId) sessionId.value = event.sessionId; assistant.runId = event.runId }
        if (event.type === 'token') {
          phase.value = 'streaming'
          assistant.content += event.content ?? ''
        }
        if (event.type === 'replace') assistant.content = event.content
        if (event.type === 'error') throw new Error(event.message || '回复生成失败')
        if (event.type === 'citation') assistant.citations = event.citations
        if (event.type === 'status') assistant.status = event.message
        if (event.type === 'done') {
          receivedDone = true
          phase.value = 'done'
          assistant.messageId = event.messageId
          assistant.runId = event.runId || assistant.runId
          if (assistant.answerTrace) {
            assistant.answerTrace.messageId = event.messageId
            terminalTrace(assistant, event.traceStatus || 'completed')
          }
        }
      })
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      while (true) {
        const { done, value } = await reader.read()
        if (currentGeneration !== generation) { await reader.cancel(); return }
        if (done) break
        parser.push(decoder.decode(value, { stream: true }))
      }
      parser.push(decoder.decode())
      parser.flush()
      if (assistant.answerTrace && !receivedDone) throw new Error('回复连接已中断，请重新发送')
      phase.value = 'done'
    } catch (reason) {
      if (currentGeneration !== generation) return
      if (reason instanceof DOMException && reason.name === 'AbortError') {
        phase.value = 'idle'
        terminalTrace(assistant, 'cancelled')
        if (!assistant.content && !assistant.answerTrace) messages.value = messages.value.filter((item) => item.id !== assistant.id)
      } else {
        error.value = reason instanceof Error ? reason.message : '连接中断，请重新发送'
        phase.value = 'error'
        terminalTrace(assistant, 'failed')
        if (!assistant.content) assistant.content = '这次连接没有完成。你可以重新发送刚才的内容。'
      }
    } finally {
      if (controller === activeController) controller = null
    }
  }

  function terminalTrace(message: ChatMessage, status: AnswerTraceStatus) {
    if (!message.answerTrace) return
    message.answerTrace.status = status
    for (const step of message.answerTrace.steps) {
      if (step.status === 'running') {
        step.status = status === 'completed' ? 'completed' : status
        step.result = status === 'cancelled' ? '此步骤已停止。' : status === 'failed' ? '连接已中断，此步骤未确认完成。' : step.result
        if (step.durationMs == null) step.durationMs = Math.max(0, Date.now() - Date.parse(step.startedAt))
      }
    }
  }

  function stop() {
    generation++
    controller?.abort()
    controller = null
    pendingSessionId.value = null
    phase.value = 'idle'
    const last = messages.value.at(-1)
    if (last?.role === 'assistant') {
      if (last.answerTrace?.status === 'running') terminalTrace(last, 'cancelled')
      if (!last.content && !last.answerTrace) messages.value.pop()
    }
  }

  async function loadSession(publicId: string) {
    reset()
    pendingSessionId.value = publicId
    const currentGeneration = ++generation
    const activeController = new AbortController()
    controller = activeController
    phase.value = 'thinking'
    try {
      const response = await request(`/api/sessions/${encodeURIComponent(publicId)}`, { signal: activeController.signal })
      const data = await response.json() as { sessionId: string; messages: ChatMessage[] }
      if (currentGeneration !== generation) return
      sessionId.value = data.sessionId
      messages.value = data.messages.filter(item => item.role === 'user' || item.role === 'assistant')
      phase.value = 'idle'
    } catch (reason) {
      if (currentGeneration !== generation) return
      error.value = reason instanceof Error ? reason.message : '无法加载会话'
      phase.value = 'error'
    } finally {
      if (currentGeneration === generation) pendingSessionId.value = null
      if (controller === activeController) controller = null
    }
  }

  function invalidateSession(publicId: string) {
    if (sessionId.value === publicId || pendingSessionId.value === publicId) reset()
  }

  async function retry() {
    if (lastUserMessage.value && !controller) {
      const last = messages.value.at(-1)
      if (last?.role === 'assistant' && phase.value === 'error') messages.value.pop()
      const previousUser = messages.value.at(-1)
      if (previousUser?.role === 'user') messages.value.pop()
      useSystemStore().selectChatBackend(lastModelBackend)
      await send(lastUserMessage.value, lastRunId.value, lastModelBackend)
    }
  }

  function reset() {
    stop()
    sessionId.value = null
    messages.value = []
    phase.value = 'idle'
    error.value = ''
    lastUserMessage.value = ''
    lastRunId.value = ''
  }

  return { sessionId, pendingSessionId, invalidateSession, messages, phase, error, lastUserMessage, send, stop, retry, reset, loadSession }
})
