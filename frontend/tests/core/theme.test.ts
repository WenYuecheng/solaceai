import { describe, expect, it, vi } from 'vitest'
import { applyTheme } from '@/core/theme'

describe('unified light appearance', () => {
  it('ignores a previous dark preference and the OS theme', () => {
    localStorage.setItem('solaceai.theme', 'dark')
    vi.stubGlobal('matchMedia', vi.fn().mockReturnValue({ matches: true }))
    applyTheme()
    expect(document.documentElement.dataset.theme).toBe('light')
    expect(document.documentElement.style.colorScheme).toBe('light')
    expect(localStorage.getItem('solaceai.theme')).toBeNull()
    vi.unstubAllGlobals()
  })
})
