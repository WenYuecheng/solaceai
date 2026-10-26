import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import ResponsiveSidebar from '@/components/student/ResponsiveSidebar.vue'

describe('responsive student navigation', () => {
  afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })
  it('keeps history in the desktop sidebar', () => {
    vi.stubGlobal('innerWidth', 1440)
    const wrapper = mount(ResponsiveSidebar, { props: { open: false }, slots: { default: '<button>历史会话</button>' } })
    expect(wrapper.find('aside').exists()).toBe(true)
    expect(wrapper.text()).toContain('历史会话')
    wrapper.unmount()
  })
  it('opens mobile navigation as a modal and exposes a close action', async () => {
    vi.stubGlobal('innerWidth', 390)
    Object.defineProperty(HTMLDialogElement.prototype, 'showModal', { configurable: true, writable: true, value: () => {} })
    Object.defineProperty(HTMLDialogElement.prototype, 'close', { configurable: true, writable: true, value: () => {} })
    const show = vi.spyOn(HTMLDialogElement.prototype, 'showModal').mockImplementation(function (this: HTMLDialogElement) { this.setAttribute('open', '') })
    vi.spyOn(HTMLDialogElement.prototype, 'close').mockImplementation(function (this: HTMLDialogElement) { this.removeAttribute('open') })
    const wrapper = mount(ResponsiveSidebar, { props: { open: false }, slots: { default: '<button>历史会话</button>' } })
    await wrapper.setProps({ open: true })
    await nextTick()
    expect(show).toHaveBeenCalled()
    expect(wrapper.get('dialog').attributes('aria-label')).toBe('会话与支持')
    await wrapper.get('button[aria-label="关闭会话侧栏"]').trigger('click')
    expect(wrapper.emitted('update:open')).toEqual([[false]])
    wrapper.unmount()
  })
})
