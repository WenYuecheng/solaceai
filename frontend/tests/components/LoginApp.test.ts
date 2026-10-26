import { mount } from '@vue/test-utils'
import { createTestingPinia } from '@pinia/testing'
import { describe, expect, it, vi } from 'vitest'
import LoginApp from '@/apps/LoginApp.vue'

describe('LoginApp', () => {
  it('presents one clear, accessible login task', () => {
    const wrapper = mount(LoginApp, {
      global: { plugins: [createTestingPinia({ createSpy: vi.fn })] },
    })
    expect(wrapper.get('h1').text()).toContain('暂时放一放')
    expect(wrapper.get('label[for="email"]').text()).toBe('邮箱')
    expect(wrapper.get('label[for="code"]').text()).toBe('邮箱验证码')
    expect(wrapper.findAll('form')).toHaveLength(1)
    expect(wrapper.text()).not.toContain('管理员端只提供')
  })
})
