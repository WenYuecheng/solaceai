/**
 * 全局用户认证与状态管理 Pinia Store (Authentication Store).
 *
 * 管理当前登录用户 Profile、权限角色计算、账号密码登录、邮箱验证码登录、会话恢复与登出。
 */
import { computed, shallowRef } from 'vue'
import { defineStore } from 'pinia'
import { clearAuth, isAdmin, readAuth, saveAuth } from '@/core/auth'
import { ApiError, requestJson } from '@/core/api'
import type { Profile } from '@/types/api'

export const useAuthStore = defineStore('auth', () => {
  /** 当前登录用户信息 */
  const profile = shallowRef<Profile | null>(null)
  /** 认证请求进行中状态 */
  const loading = shallowRef(false)
  /** 认证错误提示信息 */
  const error = shallowRef('')
  /** 是否具备管理员/教师权限 */
  const admin = computed(() => profile.value ? isAdmin(profile.value) : false)

  /**
   * 接受并持久化新会话。
   *
   * @param token - 认证凭证（兼容旧版调用方仍可能传入 Basic token；新版基于 HttpOnly Cookie 流程不再向客户端明文持久化）
   * @param nextProfile - 用户个人资料对象
   */
  function acceptSession(token: string, nextProfile: Profile) {
    saveAuth(token, nextProfile)
    profile.value = nextProfile
    error.value = ''
  }

  /**
   * 发起底层认证请求并处理异常。
   *
   * @param path - 认证接口路径
   * @param body - 请求请求体参数
   */
  async function authenticate(path: string, body: unknown) {
    loading.value = true
    error.value = ''
    clearAuth()
    try {
      const nextProfile = await requestJson<Profile>(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      acceptSession('', nextProfile)
      return nextProfile
    } catch (reason) {
      error.value = reason instanceof ApiError ? reason.message : '网络连接异常，请稍后重试'
      throw reason
    } finally {
      loading.value = false
    }
  }

  /**
   * 账号密码登录。
   *
   * @param username - 用户名
   * @param password - 密码
   */
  function login(username: string, password: string) {
    return authenticate('/api/auth/login', { username: username.trim(), password })
  }

  /**
   * 邮箱验证码快捷登录。
   *
   * @param email - 邮箱地址
   * @param code - 6 位数字验证码
   */
  function emailLogin(email: string, code: string) {
    return authenticate('/api/auth/email-login', { email: email.trim(), code })
  }

  /**
   * 页面刷新时尝试恢复已存在的会话。
   */
  async function resume() {
    const legacy = readAuth()
    try {
      const nextProfile = await requestJson<Profile>('/api/profile')
      acceptSession(legacy?.token || '', nextProfile)
      return nextProfile
    } catch {
      clearAuth()
      profile.value = null
      return null
    }
  }

  /**
   * 登出当前账号并清除服务端与本地会话。
   */
  async function logout() {
    try {
      await requestJson('/api/auth/logout', { method: 'POST' })
      clearAuth()
      profile.value = null
      error.value = ''
    } catch (reason) {
      error.value = reason instanceof ApiError ? reason.message : '退出失败，请检查网络后重试'
      throw reason
    }
  }

  return { profile, loading, error, admin, acceptSession, login, emailLogin, resume, logout }
})
