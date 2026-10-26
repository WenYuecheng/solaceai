import { defineStore } from 'pinia'
import { applyTheme } from '@/core/theme'

export const useThemeStore = defineStore('theme', () => {
  const resolved = 'light' as const
  function initialize() { applyTheme() }
  return { resolved, initialize }
})
