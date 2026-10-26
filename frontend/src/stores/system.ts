import { computed, shallowRef } from 'vue'
import { defineStore } from 'pinia'
import { requestJson } from '@/core/api'
import type { AgentStatus, ChatBackend } from '@/types/api'

export const useSystemStore = defineStore('system', () => {
  const healthy = shallowRef<boolean | null>(null)
  const agent = shallowRef<AgentStatus | null>(null)
  const loading = shallowRef(false)
  const selectedChatBackend = shallowRef<ChatBackend | null>(null)
  const availableChatBackends = computed<ChatBackend[]>(() => {
    const backends = agent.value?.availableChatBackends?.filter(backend => backend === 'cloud' || backend === 'local')
    return backends?.length ? backends : ['cloud']
  })
  const chatBackend = computed<ChatBackend>(() => {
    const preferred = selectedChatBackend.value ?? agent.value?.defaultChatBackend ?? 'cloud'
    return availableChatBackends.value.includes(preferred) ? preferred : availableChatBackends.value[0]!
  })
  const chatModelLabel = computed(() => {
    if (!agent.value) return '模型读取中'
    const model = agent.value.chatModels?.[chatBackend.value]
    if (model) return `${model.provider} · ${model.label ? `${model.label} · ` : ''}${model.model}`
    return modelLabel.value
  })
  function selectChatBackend(backend: ChatBackend) {
    if (availableChatBackends.value.includes(backend)) selectedChatBackend.value = backend
  }
  const modelLabel = computed(() => {
    if (!agent.value) return '模型读取中'
    if (!agent.value.realModelEnabled) return 'Mock 演示'
    return `${agent.value.provider} · ${agent.value.model}`
  })

  async function load(includeAgent = true) {
    loading.value = true
    try {
      const health = await fetch('/actuator/health')
      const body = await health.json() as { status?: string }
      healthy.value = health.ok && body.status === 'UP'
    } catch { healthy.value = false }
    if (includeAgent) {
      try { agent.value = await requestJson<AgentStatus>('/api/agent/status') } catch { agent.value = null }
    }
    loading.value = false
  }
  return { healthy, agent, loading, modelLabel, load, availableChatBackends, chatBackend, chatModelLabel, selectChatBackend }
})
