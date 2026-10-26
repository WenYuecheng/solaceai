import { describe, expect, it } from 'vitest'
import { buildDashboardMetrics, countBy } from '@/core/analytics'

describe('admin analytics', () => {
  it('derives dashboard counts without mutating API records', () => {
    const reports = [
      { riskLevel: 'HIGH', emotion: 'ANXIETY', intent: 'RISK' },
      { riskLevel: 'LOW', emotion: 'NORMAL', intent: 'CHAT' },
    ]
    const before = structuredClone(reports)
    expect(buildDashboardMetrics(reports, [{ status: 'OPEN' }], [{ status: 'SUCCESS' }], [])).toEqual({
      reports: 2,
      highRisk: 1,
      cases: 1,
      openCases: 1,
      excel: 1,
      excelSuccess: 1,
      alerts: 0,
      alertSuccess: 0,
    })
    expect(reports).toEqual(before)
    expect(countBy(reports, 'riskLevel')).toEqual(new Map([['HIGH', 1], ['LOW', 1]]))
  })
})
