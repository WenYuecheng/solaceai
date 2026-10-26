import { describe, expect, it } from 'vitest'
import { buildDistributionData, buildSevenDayTrend } from '@/core/chart-options'

describe('chart option data', () => {
  it('creates a stable seven-day series including empty days', () => {
    const today = new Date('2026-08-26T12:00:00+08:00')
    const data = buildSevenDayTrend([
      { createdAt: '2026-08-26T08:00:00+08:00' },
      { createdAt: '2026-08-24T08:00:00+08:00' },
    ], today)
    expect(data.map((item) => item.value)).toEqual([0, 0, 0, 0, 1, 0, 1])
    expect(data.at(-1)?.label).toBe('8/26')
  })

  it('keeps zero-value categories visible for accessible summaries', () => {
    expect(buildDistributionData([{ riskLevel: 'HIGH' }], 'riskLevel', [
      ['LOW', '低风险'], ['MEDIUM', '中风险'], ['HIGH', '高风险'],
    ])).toEqual([
      { name: '低风险', value: 0 }, { name: '中风险', value: 0 }, { name: '高风险', value: 1 },
    ])
  })
})
