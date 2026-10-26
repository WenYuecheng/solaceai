/**
 * 前端鉴权与会话凭证持久化模块 (Authentication & Session Storage).
 * 
 * 负责客户端身份凭证的 Base64 编码、会话状态持久化 (sessionStorage) 与管理员角色判定。
 */
import type { AuthSession, Profile } from '@/types/api'

/** 会话状态在 sessionStorage 中的存储键名 */
export const AUTH_KEY = 'solaceai.auth'

/**
 * 将用户名与密码安全编码为标准 HTTP Basic 认证的 Base64 凭证串。
 *
 * @param username - 用户登录名。
 * @param password - 用户密码。
 * @returns UTF-8 兼容的 Base64 编码字符串。
 */
export function encodeCredentials(username: string, password: string): string {
  const bytes = new TextEncoder().encode(`${username}:${password}`)
  let binary = ''
  for (const byte of bytes) binary += String.fromCharCode(byte)
  return btoa(binary)
}

/**
 * 保存当前认证会话至浏览器 sessionStorage。
 *
 * @param token - 认证凭证 Token（如 Base64 凭证）。
 * @param profile - 登录成功的用户信息对象。
 */
export function saveAuth(token: string, profile: Profile): void {
  const session: AuthSession = {
    token,
    username: profile.username,
    displayName: profile.displayName,
    roles: profile.roles ?? [],
  }
  sessionStorage.setItem(AUTH_KEY, JSON.stringify(session))
}

/**
 * 从浏览器 sessionStorage 中读取并反序列化当前认证会话。
 *
 * @returns 有效的会话对象，若不存在或数据已损坏则返回 null。
 */
export function readAuth(): AuthSession | null {
  try {
    const parsed = JSON.parse(sessionStorage.getItem(AUTH_KEY) ?? 'null') as Partial<AuthSession> | null
    if (typeof parsed?.token !== 'string' || !parsed.username) return null
    return {
      token: parsed.token,
      username: parsed.username,
      displayName: parsed.displayName ?? parsed.username,
      roles: Array.isArray(parsed.roles) ? parsed.roles : [],
    }
  } catch {
    return null
  }
}

/**
 * 清除本地会话存储，实现安全登出。
 */
export function clearAuth(): void {
  sessionStorage.removeItem(AUTH_KEY)
}

/**
 * 校验当前用户角色中是否具备管理员权限 (ROLE_ADMIN)。
 *
 * @param profile - 包含 roles 字段的用户信息或会话对象。
 * @returns 具备管理员权限返回 true，否则返回 false。
 */
export function isAdmin(profile: Pick<Profile, 'roles'> | AuthSession): boolean {
  return profile.roles.some((role) => role.authority === 'ROLE_ADMIN')
}
