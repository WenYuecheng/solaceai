import { mount, flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, expect, it, vi } from 'vitest'
import AccessForm from '@/components/auth/AccessForm.vue'
import { useAuthStore } from '@/stores/auth'

beforeEach(() => { setActivePinia(createPinia()) })
it('combines student registration and email login in one verified-email form', async () => {
  const wrapper = mount(AccessForm)
  await wrapper.get('[data-mode="email"]').trigger('click')
  expect(wrapper.find('input[type="email"]').exists()).toBe(true)
  expect(wrapper.find('input[autocomplete="one-time-code"]').exists()).toBe(true)
  expect(wrapper.find('[data-mode="register"]').exists()).toBe(false)
  expect(wrapper.find('input[name="confirmPassword"]').exists()).toBe(false)
  expect(wrapper.text()).toContain('自动创建学生账号')
})
it('submits new and existing email users through the same endpoint', async () => {
  const auth = useAuthStore()
  const profile = { id: 1, username: 'student_test', displayName: '新同学', roles: [{ authority: 'ROLE_USER' }] }
  const login = vi.spyOn(auth, 'emailLogin').mockResolvedValue(profile)
  const wrapper = mount(AccessForm)
  await wrapper.get('[data-mode="email"]').trigger('click')
  await wrapper.get('input[name="email"]').setValue('student@example.com')
  await wrapper.get('input[name="code"]').setValue('123456')
  await wrapper.get('form').trigger('submit')
  await flushPromises()
  expect(login).toHaveBeenCalledWith('student@example.com', '123456')
  expect(wrapper.emitted('authenticated')?.[0]).toEqual([profile])
})
it('keeps short legacy usernames usable in password mode', async () => {
  const wrapper = mount(AccessForm)
  await wrapper.get('[data-mode="password"]').trigger('click')
  expect(wrapper.get('input[name="username"]').attributes('minlength')).toBeUndefined()
})
