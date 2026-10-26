import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import StudentHeader from '@/components/student/StudentHeader.vue'
import { useSystemStore } from '@/stores/system'
import { useChatStore } from '@/stores/chat'

const cloud = { provider: 'openai', model: 'deepseek-flash' }
const local = { provider: 'vllm-metal', model: 'solace-response', label: 'Qwen3-4B · 回答 LoRA' }

describe('student model choice', () => {
  beforeEach(() => { setActivePinia(createPinia()) })
  function configure(backends: ('cloud' | 'local')[] = ['cloud', 'local']) {
    useSystemStore().agent = {
      ...cloud, realModelEnabled: true, runtimeProfile: 'offline',
      availableChatBackends: backends, defaultChatBackend: 'cloud', chatModels: { cloud, local },
    }
    return mount(StudentHeader, { props: { displayName: '学生' } })
  }
  it('shows concise stable model names and switches only this browser', async () => {
    const wrapper = configure()
    const group = wrapper.get('[role="group"][aria-label="对话模型"]')
    expect(group.findAll('button').map(button => button.text())).toEqual(['DeepSeek Flash', 'Qwen3'])
    expect(group.get('[data-backend="cloud"]').attributes('aria-pressed')).toBe('true')
    await group.get('[data-backend="local"]').trigger('click')
    expect(useSystemStore().chatBackend).toBe('local')
    expect(group.get('[data-backend="local"]').attributes('aria-pressed')).toBe('true')
    expect(group.text()).not.toMatch(/openai|vllm|LoRA|solace-response|4B/)
    expect(wrapper.find('select').exists()).toBe(false)
    setActivePinia(createPinia())
    expect(useSystemStore().chatBackend).toBe('cloud')
  })
  it('shows one quiet cloud choice when local is unavailable', () => {
    const wrapper = configure(['cloud'])
    const group = wrapper.get('[role="group"][aria-label="对话模型"]')
    expect(group.findAll('button').map(button => button.text())).toEqual(['DeepSeek Flash'])
    expect(group.get('button').attributes('disabled')).toBeDefined()
  })
  it.each(['thinking', 'streaming'] as const)('prevents switching during %s', async (phase) => {
    const wrapper = configure()
    useChatStore().phase = phase
    await wrapper.vm.$nextTick()
    const button = wrapper.get('[data-backend="local"]')
    expect(button.attributes('disabled')).toBeDefined()
    await button.trigger('click')
    expect(useSystemStore().chatBackend).toBe('cloud')
  })
})
