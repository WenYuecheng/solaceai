import { mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ConfirmDialog from '@/components/shared/ConfirmDialog.vue'

describe('garden confirmation dialog', () => {
  beforeEach(() => {
    Object.defineProperty(HTMLDialogElement.prototype, 'showModal', { configurable: true, writable: true, value: vi.fn(function (this: HTMLDialogElement) { this.setAttribute('open', '') }) })
    Object.defineProperty(HTMLDialogElement.prototype, 'close', { configurable: true, writable: true, value: vi.fn(function (this: HTMLDialogElement) { this.removeAttribute('open') }) })
  })
  afterEach(() => vi.restoreAllMocks())
  it('keeps the explanation escaped and only confirms on the explicit action', async () => {
    const wrapper = mount(ConfirmDialog, { props: { open: true, title: '移除会话？', message: '<script>unsafe</script>', confirmLabel: '移除会话' } })
    expect(wrapper.find('script').exists()).toBe(false)
    expect(wrapper.text()).toContain('<script>unsafe</script>')
    await wrapper.get('[data-confirm]').trigger('click')
    expect(wrapper.emitted('confirm')).toHaveLength(1)
    expect(wrapper.emitted('cancel')).toBeUndefined()
    wrapper.unmount()
  })
  it('cancels on Escape without confirming', async () => {
    const wrapper = mount(ConfirmDialog, { props: { open: true, title: '开始新会话？', message: '当前对话会清空', confirmLabel: '开始新会话' } })
    await wrapper.get('dialog').trigger('cancel')
    expect(wrapper.emitted('cancel')).toHaveLength(1)
    expect(wrapper.emitted('confirm')).toBeUndefined()
    wrapper.unmount()
  })
})
