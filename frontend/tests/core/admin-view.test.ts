import { describe, expect, it } from 'vitest'
import { filterReports, parseAdminSection, updateAdminSection } from '@/core/admin-view'

describe('admin view state', () => {
  it('keeps the selected section deep-linkable', () => {
    expect(parseAdminSection('?view=reports')).toBe('reports')
    expect(parseAdminSection('?view=unknown')).toBe('overview')
    expect(updateAdminSection('/admin.html?view=cases&debug=1', 'knowledge')).toBe('/admin.html?view=knowledge&debug=1')
  })

  it('filters reports across identity, summary, risk and intent', () => {
    const reports = [
      { id: 1, displayName: '林同学', username: 'lin', summary: '考试焦虑', riskLevel: 'MEDIUM', intent: 'CONSULT' },
      { id: 2, displayName: '周同学', username: 'zhou', summary: '日常问候', riskLevel: 'LOW', intent: 'CHAT' },
    ]
    expect(filterReports(reports, '焦虑', 'ALL', 'ALL').map((item) => item.id)).toEqual([1])
    expect(filterReports(reports, '', 'LOW', 'CHAT').map((item) => item.id)).toEqual([2])
  })
})
