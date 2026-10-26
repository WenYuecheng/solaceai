import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useAuthStore } from '@/stores/auth'

describe('login errors', () => {
  beforeEach(() => { setActivePinia(createPinia()); localStorage.clear() })
  it.each([401, 500])('does not expose a response body for status %s', async (status) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"detail":"Bad credentials /private/config API_KEY=secret"}', { status })))
    const store = useAuthStore()
    await expect(store.login('student', 'wrong')).rejects.toThrow()
    expect(store.error).not.toMatch(/detail|credentials|private|secret/)
    expect(store.error).toBe(status === 401 ? '用户名或密码不正确，请重新输入' : '服务暂时不可用，请稍后重试')
  })
  it('does not expose a network exception', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('internal proxy host')))
    const store = useAuthStore()
    await expect(store.login('student', 'wrong')).rejects.toThrow()
    expect(store.error).toBe('网络连接异常，请稍后重试')
  })
})

describe('cookie authentication', () => {
  beforeEach(() => { setActivePinia(createPinia()); sessionStorage.clear() })
  const profile = { id: 2, username: 'newstudent', displayName: '同学', roles: [{ authority: 'ROLE_USER' }] }
  it('posts credentials without storing a reusable password token', async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify(profile)))
    vi.stubGlobal('fetch', fetcher)
    await useAuthStore().login('newstudent', 'password123')
    expect(fetcher.mock.calls[0]?.[0]).toBe('/api/auth/login')
    expect(JSON.parse(fetcher.mock.calls[0]?.[1].body)).toEqual({ username: 'newstudent', password: 'password123' })
    expect(sessionStorage.getItem('solaceai.auth')).not.toContain(btoa('newstudent:password123'))
  })
  it('resumes a cookie session without browser storage', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(profile))))
    expect(await useAuthStore().resume()).toEqual(profile)
  })
  it('logs in through an email code', async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify(profile)))
    vi.stubGlobal('fetch', fetcher)
    expect(await useAuthStore().emailLogin('student@qq.com', '123456')).toEqual(profile)
    expect(fetcher.mock.calls[0]?.[0]).toBe('/api/auth/email-login')
  })
  it('revokes the server session on logout', async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response('{}'))
    vi.stubGlobal('fetch', fetcher)
    const store = useAuthStore()
    store.acceptSession('legacy', profile)
    await store.logout()
    expect(fetcher.mock.calls[0]?.[0]).toBe('/api/auth/logout')
    expect(store.profile).toBeNull()
  })
})
