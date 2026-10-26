import { mount, flushPromises } from '@vue/test-utils'
import { it, expect, vi } from 'vitest'
import MemoryPanel from '@/components/student/MemoryPanel.vue'
import { createPinia, setActivePinia } from 'pinia'
import { useAuthStore } from '@/stores/auth'

it('shows explicitly saved memories and deletes the selected entry', async () => {
  setActivePinia(createPinia())
  useAuthStore().acceptSession('token', { id: 1, username: 'x', displayName: 'x', roles: [] })
  const fetch = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify([{id:1,content:'我喜欢简短回答'}]), {status:200}))
    .mockResolvedValueOnce(new Response('{"deleted":true}', {status:200}))
  vi.stubGlobal('fetch', fetch)
  const wrapper = mount(MemoryPanel)
  await flushPromises()
  expect(wrapper.text()).toContain('我喜欢简短回答')
  await wrapper.get('button[data-delete-memory]').trigger('click')
  await flushPromises()
  expect(wrapper.text()).not.toContain('我喜欢简短回答')
  expect(fetch.mock.calls[1][1].method).toBe('DELETE')
})
