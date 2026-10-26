/**
 * 前端 HTTP 请求与统一错误处理网关 (API Client & Error Handling Gateway).
 *
 * 封装原生 fetch 请求，自动携带鉴权头与 CSRF 防护标识，
 * 统一拦截 401 自动登出与将服务端错误映射为友好的中文提示信息。
 */
import { clearAuth, readAuth } from '@/core/auth'

/**
 * 前端 API 请求异常封装类。
 */
export class ApiError extends Error {
  /**
   * 构造 API 异常实例。
   *
   * @param message - 用户可见的错误描述文案。
   * @param status - HTTP 响应状态码。
   */
  constructor(message: string, readonly status: number) {
    super(message)
    this.name = 'ApiError'
  }
}

/**
 * 根据 HTTP 状态码获取标准化、防信息泄露的用户端中文提示。
 *
 * @param status - HTTP 响应状态码。
 * @returns 友好的用户端提示文案。
 */
export function publicErrorMessage(status: number): string {
  if (status === 401) return '用户名或密码不正确，请重新输入'
  if (status === 403) return '没有权限执行此操作'
  if (status === 404) return '请求的内容不存在或已移除'
  if (status === 409) return '当前操作存在冲突，请刷新后重试'
  if (status === 422 || status === 400) return '请检查输入内容和长度'
  if (status === 429) return '请求较多，请稍后重试'
  return '服务暂时不可用，请稍后重试'
}

/**
 * 执行底层 HTTP 请求并进行统一认证注头与状态拦截。
 *
 * @param path - API 请求相对或绝对路径。
 * @param options - 原生 Fetch 请求配置。
 * @returns 成功的 Response 响应对象。
 * @throws {ApiError} 当响应状态码非 2xx 时抛出统一封装的异常。
 */
export async function request(path: string, options: RequestInit = {}): Promise<Response> {
  const auth = readAuth()
  const headers = new Headers(options.headers)
  if (auth?.token) headers.set('Authorization', `Basic ${auth.token}`)
  headers.set('X-Requested-With', 'SolaceAI')
  const response = await fetch(path, { ...options, credentials: 'same-origin', headers })
  if (!response.ok) {
    if (response.status === 401) clearAuth()
    let message = publicErrorMessage(response.status)
    if (path.startsWith('/api/auth/') || response.status === 422) {
      try {
        const payload = await response.clone().json()
        const safeAuthMessages = new Set([
          '请检查输入信息及密码确认',
          '用户名或密码错误',
          '验证码无效或已过期',
          '登录已过期，请重新登录',
          '用户名或邮箱已被使用',
          '请求过于频繁，请稍后再试',
          '请等待60秒后重新发送验证码',
          '邮件服务暂不可用，请稍后重试',
          '请求来源无效',
          '原密码不正确',
          '请先设置用户名和密码',
          '账号已完成设置',
          '账号已绑定邮箱',
          '账号凭据已更新，请重新登录',
        ])
        if (
          typeof payload?.detail === 'string' &&
          payload.detail.trim() &&
          (!path.startsWith('/api/auth/') || safeAuthMessages.has(payload.detail))
        ) {
          message = payload.detail
        }
      } catch {
        // 解析失败时降级使用状态码映射的通用提示信息
      }
    }
    throw new ApiError(message, response.status)
  }
  return response
}

/**
 * 发送 HTTP 请求并将响应体自动解析为 JSON 格式。
 *
 * @template T - 预期的返回数据类型。
 * @param path - API 请求路径。
 * @param options - 原生 Fetch 配置。
 * @returns 解析后的 JSON 数据 Promise。
 */
export async function requestJson<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await request(path, options)
  return response.json() as Promise<T>
}
