import { mount, flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, expect, it, vi } from 'vitest'
import AccessForm from '@/components/auth/AccessForm.vue'
import { useAuthStore } from '@/stores/auth'

beforeEach(() => setActivePinia(createPinia()))
it('keeps verified new users on credential setup until completion', async () => {
  const auth = useAuthStore()
  const pending = { id: 4, username: 'student_auto', displayName: '新同学', roles: [{ authority: 'ROLE_USER' }], requiresSetup: true }
  vi.spyOn(auth, 'emailLogin').mockImplementation(async () => { auth.profile = pending; return pending })
  const wrapper = mount(AccessForm)
  await wrapper.get('input[name="email"]').setValue('new@example.com')
  await wrapper.get('input[name="code"]').setValue('123456')
  await wrapper.get('form').trigger('submit')
  await flushPromises()
  expect(wrapper.emitted('authenticated')).toBeUndefined()
  expect(wrapper.find('input[name="confirmPassword"]').exists()).toBe(true)
})
it('labels password login as accepting either username or email', async () => {
  const wrapper = mount(AccessForm)
  await wrapper.get('[data-mode="password"]').trigger('click')
  expect(wrapper.get('label[for="username"]').text()).toContain('邮箱')
})
