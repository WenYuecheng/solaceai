import { beforeEach, describe, expect, it, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { requestJson } from '@/core/api'
import { useAuthStore } from '@/stores/auth'

describe('API client', () => {
  it('maps an error status without exposing the response body', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"detail":"请先确认接手再结案"}', { status: 409 })))
    await expect(requestJson('/api/example')).rejects.toThrow(/^当前操作存在冲突，请刷新后重试$/)
  })
  beforeEach(() => setActivePinia(createPinia()))

  it('adds the current Basic authorization header', async () => {
    const auth = useAuthStore()
    auth.acceptSession('encoded-token', {
      id: 1, username: 'student', displayName: '学生', roles: [],
    })
    const fetchMock = vi.fn().mockResolvedValue(new Response('{"ok":true}', {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }))
    vi.stubGlobal('fetch', fetchMock)

    await expect(requestJson<{ ok: boolean }>('/api/example')).resolves.toEqual({ ok: true })
    expect(fetchMock).toHaveBeenCalledWith('/api/example', expect.objectContaining({
      headers: expect.any(Headers),
    }))
    const headers = fetchMock.mock.calls[0][1].headers as Headers
    expect(headers.get('Authorization')).toBe('Basic encoded-token')
  })

  it('surfaces a readable server error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('权限不足', { status: 403 })))
    await expect(requestJson('/api/example')).rejects.toThrow('没有权限执行此操作')
  })

  it('surfaces validation and quality error detail for status 422', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"detail":"该文档质量不达标，请检查后重新上传"}', { status: 422 })))
    await expect(requestJson('/api/example')).rejects.toThrow('该文档质量不达标，请检查后重新上传')
  })
})
