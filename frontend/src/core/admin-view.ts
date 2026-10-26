/**
 * 教师/管理员后台视图路由与报告筛选工具模块 (Admin Views & Filtering).
 *
 * 提供后台多面板（全览、个案、咨询报告、邮件交付、知识库）URL 同步与多条件搜索过滤能力。
 */

/** 后台可用子视图类型联合 */
export type AdminSection = 'overview' | 'cases' | 'reports' | 'delivery' | 'knowledge'

/** 后台合法视图名称集合 */
const sections = new Set<AdminSection>(['overview', 'cases', 'reports', 'delivery', 'knowledge'])

/**
 * 从浏览器 URL 查询参数中解析当前激活的后台视图。
 *
 * @param search - window.location.search 字符串。
 * @returns 合法的后台视图枚举，默认兜底为 'overview'。
 */
export function parseAdminSection(search: string): AdminSection {
  const value = new URLSearchParams(search).get('view') as AdminSection | null
  return value && sections.has(value) ? value : 'overview'
}

/**
 * 根据选中的新视图更新当前 URL 的查询参数。
 *
 * @param current - 当前 URL 路径及参数。
 * @param section - 目标视图类型。
 * @returns 组装好的新 pathname + search 字符串。
 */
export function updateAdminSection(current: string, section: AdminSection): string {
  const url = new URL(current, 'https://solaceai.local')
  url.searchParams.set('view', section)
  return `${url.pathname}${url.search}`
}

/**
 * 根据搜索关键字、风险等级及咨询意向组合过滤报告列表。
 *
 * @template T - 包含展示姓名、用户名、摘要等字段的报告对象。
 * @param reports - 原始报告数组。
 * @param query - 文本搜索关键字（匹配姓名/学号/摘要）。
 * @param risk - 风险等级筛选值（'ALL' 或具体等级）。
 * @param intent - 意图筛选值（'ALL' 或具体意图）。
 * @returns 过滤后的报告子集。
 */
export function filterReports<T extends {
  displayName?: string
  username?: string
  summary?: string
  riskLevel?: string
  intent?: string
}>(reports: T[], query: string, risk: string, intent: string): T[] {
  const needle = query.trim().toLocaleLowerCase('zh-CN')
  return reports.filter((report) => {
    const haystack = `${report.displayName ?? ''} ${report.username ?? ''} ${report.summary ?? ''}`.toLocaleLowerCase('zh-CN')
    return (!needle || haystack.includes(needle))
      && (risk === 'ALL' || report.riskLevel === risk)
      && (intent === 'ALL' || report.intent === intent)
  })
}
