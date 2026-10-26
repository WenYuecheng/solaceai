/**
 * 前端文本与数值展示格式化工具模块 (Display Formatting Utilities).
 *
 * 提供日期时间本地化（中文简短格式）、百分比转换以及业务状态与风险枚举的中文字典翻译。
 */

/**
 * 将 ISO 格式时间字符串格式化为中文友好的日期时间文案。
 *
 * @param value - ISO 时间字符串或 null/undefined。
 * @returns 格式化后的日期时间字符串，若为空或非法则返回 '—'。
 */
export function formatDateTime(value?: string | null): string {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'
  return new Intl.DateTimeFormat('zh-CN', { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}

/**
 * 计算数值占总数的百分比并格式化为不含小数位的中文百分比字符串。
 *
 * @param value - 当前分子计数值。
 * @param total - 总量分母计数值。
 * @returns 中文百分比字符串（如 '85%'）。
 */
export function percentage(value: number, total: number): string {
  return new Intl.NumberFormat('zh-CN', { style: 'percent', maximumFractionDigits: 0 }).format(total ? value / total : 0)
}

/**
 * 将系统内部状态枚举码（如 SUCCESS, HIGH, OPEN 等）转换为对应的中文描述标签。
 *
 * @param value - 枚举状态代码。
 * @returns 对应的中文语义标签，未知值则兜底返回 '未知' 或原值。
 */
export function statusLabel(value?: string | null): string {
  const labels: Record<string, string> = {
    SUCCESS: '成功', FAILED: '失败', OPEN: '待跟进', ALERT_SENT: '已预警', ACKNOWLEDGED: '已确认', CLOSED: '已结案',
    LOW: '低风险', MEDIUM: '中风险', HIGH: '高风险', CHAT: '日常交流', CONSULT: '心理咨询', RISK: '风险干预',
  }
  const normalized = String(value ?? '').toUpperCase()
  return labels[normalized] ?? value ?? '未知'
}
