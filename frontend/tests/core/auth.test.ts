import { describe, expect, it } from 'vitest'
import { AUTH_KEY, encodeCredentials, readAuth, saveAuth } from '@/core/auth'

describe('authentication compatibility', () => {
  it('uses the legacy session key and Basic token shape', () => {
    const token = encodeCredentials('student', '学生123')
    saveAuth(token, { id: 1, username: 'student', displayName: '学生', roles: [] })

    expect(AUTH_KEY).toBe('solaceai.auth')
    expect(readAuth()).toMatchObject({ token, username: 'student', displayName: '学生' })
    const decoded = Uint8Array.from(atob(token), (character) => character.charCodeAt(0))
    expect(new TextDecoder().decode(decoded)).toBe('student:学生123')
  })

  it('returns null for malformed session data', () => {
    sessionStorage.setItem(AUTH_KEY, '{broken')
    expect(readAuth()).toBeNull()
  })
})
