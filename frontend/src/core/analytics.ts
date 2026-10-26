/**
 * 后台统计分析与指标汇总核心模块 (Analytics & Metric Calculations).
 *
 * 提供通用的数组属性聚合计数及仪表盘核心指标（总报告数、高危数、跟进个案、导出成功率等）的计算逻辑。
 */

type RecordLike = object

/**
 * 按照指定属性键的值对列表项进行分组计数。
 *
 * @template T - 记录对象类型。
 * @template K - 分组属性名。
 * @param items - 数据项数组。
 * @param key - 用于分组的键名。
 * @returns 统计 Map（键为大写值字符串，值为计数）。
 */
export function countBy<T extends RecordLike, K extends keyof T>(items: T[], key: K): Map<string, number> {
  const counts = new Map<string, number>()
  for (const item of items) {
    const value = String(item[key] ?? 'UNKNOWN').toUpperCase()
    counts.set(value, (counts.get(value) ?? 0) + 1)
  }
  return counts
}

/**
 * 汇总计算管理控制台顶部仪表盘的核心量化指标。
 *
 * @param reports - 咨询评估报告列表。
 * @param cases - 风险个案列表。
 * @param excel - Excel 导出任务记录列表。
 * @param alerts - 风险邮件预警记录列表。
 * @returns 仪表盘各关键指标汇总对象。
 */
export function buildDashboardMetrics(
  reports: Array<{ riskLevel?: unknown }>,
  cases: Array<{ status?: unknown }>,
  excel: Array<{ status?: unknown }>,
  alerts: Array<{ status?: unknown }>,
) {
  return {
    reports: reports.length,
    highRisk: reports.filter((item) => item.riskLevel === 'HIGH').length,
    cases: cases.length,
    openCases: cases.filter((item) => item.status === 'OPEN').length,
    excel: excel.length,
    excelSuccess: excel.filter((item) => item.status === 'SUCCESS').length,
    alerts: alerts.length,
    alertSuccess: alerts.filter((item) => item.status === 'SUCCESS').length,
  }
}
