import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import ChatComposer from '@/components/student/ChatComposer.vue'

describe('ChatComposer', () => {
  it('emits trimmed text and clears the field', async () => {
    const wrapper = mount(ChatComposer, { props: { phase: 'idle' } })
    const textarea = wrapper.get('textarea')
    await textarea.setValue('  我想聊聊考试  ')
    expect(wrapper.text()).toContain('10 / 1000')
    await wrapper.get('form').trigger('submit')
    expect(wrapper.emitted('send')).toEqual([['我想聊聊考试']])
    expect((textarea.element as HTMLTextAreaElement).value).toBe('')
  })

  it('shows a stop action while a reply is active', () => {
    const wrapper = mount(ChatComposer, { props: { phase: 'streaming' } })
    expect(wrapper.findAll('button[type="button"]').some((button) => button.text().includes('停止回复'))).toBe(true)
  })
})
