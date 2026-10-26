/**
 * 后台统计图表数据构建工具模块 (Chart Data Builders).
 *
 * 负责聚合近 7 日趋势曲线数据点与维度（风险、意图、个案状态）占比分布数据。
 */
import { countBy } from '@/core/analytics'

/**
 * 将日期对象或日期字符串格式化为 YYYY-MM-DD 本地日期键名。
 *
 * @param value - Date 对象或时间字符串。
 * @returns YYYY-MM-DD 格式字符串。
 */
function localDateKey(value: Date | string): string {
  const date = value instanceof Date ? value : new Date(value)
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

/**
 * 根据记录的创建时间统计生成连续 7 天的每日咨询/报告趋势折线数据。
 *
 * @template T - 包含 createdAt 字段的实体类型。
 * @param records - 历史记录列表。
 * @param now - 基准当前时间，默认为当前客户端时间。
 * @returns 包含 7 个日期点（日期标签 label 与数量 value）的数组。
 */
export function buildSevenDayTrend<T extends { createdAt: string }>(records: T[], now = new Date()) {
  const base = new Date(now)
  base.setHours(0, 0, 0, 0)
  const counts = new Map<string, number>()
  records.forEach((record) => counts.set(localDateKey(record.createdAt), (counts.get(localDateKey(record.createdAt)) ?? 0) + 1))
  return Array.from({ length: 7 }, (_, index) => {
    const date = new Date(base)
    date.setDate(base.getDate() - (6 - index))
    return {
      label: new Intl.DateTimeFormat('zh-CN', { month: 'numeric', day: 'numeric' }).format(date),
      value: counts.get(localDateKey(date)) ?? 0,
    }
  })
}

/**
 * 统计指定属性的枚举值在集合中的频次并映射为图表饼图/环形图所需的数据结构。
 *
 * @template T - 记录对象类型。
 * @template K - 待统计的属性键名。
 * @param records - 记录对象列表。
 * @param key - 分组属性名。
 * @param definitions - 枚举值与展示名称的元组数组（如 [['HIGH', '高风险'], ...]）。
 * @returns 包含名称 name 与计数值 value 的图表数据项列表。
 */
export function buildDistributionData<T extends object, K extends keyof T>(
  records: T[],
  key: K,
  definitions: Array<[string, string]>,
) {
  const counts = countBy(records, key)
  return definitions.map(([value, name]) => ({ name, value: counts.get(value) ?? 0 }))
}
