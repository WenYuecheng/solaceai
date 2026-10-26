import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import CaseFollowup from '@/components/admin/CaseFollowup.vue'
import { requestJson } from '@/core/api'
vi.mock('@/core/api', () => ({ requestJson: vi.fn() }))
const caseItem = { id: 1, reportId: 1, riskLevel: 'HIGH', status: 'OPEN', owner: '', summary: '', handoffSummary: '', createdAt: '', updatedAt: '' }

describe('case followup', () => {
  beforeEach(() => vi.mocked(requestJson).mockReset())
  it('loads history, acknowledges through authenticated API and emits the refreshed case', async () => {
    vi.mocked(requestJson).mockResolvedValueOnce([{ id: 1, actor: 'teacher', note: '访谈记录', createdAt: '' }])
    const wrapper = mount(CaseFollowup, { props: { caseItem } })
    await flushPromises()
    expect(wrapper.text()).toContain('访谈记录')
    const next = { ...caseItem, status: 'ACKNOWLEDGED', acknowledgedBy: 'teacher' }
    vi.mocked(requestJson).mockResolvedValueOnce(next).mockResolvedValueOnce([])
    await wrapper.get('[data-action="acknowledge"]').trigger('click')
    await flushPromises()
    expect(requestJson).toHaveBeenCalledWith('/api/admin/cases/1/acknowledge', expect.objectContaining({ method: 'POST' }))
    expect(wrapper.emitted('updated')?.[0]).toEqual([next])
  })
  it('requires note text and acknowledgement before close, and prevents closed mutations', async () => {
    vi.mocked(requestJson).mockResolvedValue([])
    const wrapper = mount(CaseFollowup, { props: { caseItem } })
    await flushPromises()
    expect(wrapper.get('[data-action="notes"]').attributes('disabled')).toBeDefined()
    expect(wrapper.get('[data-action="close"]').attributes('disabled')).toBeDefined()
    await wrapper.setProps({ caseItem: { ...caseItem, status: 'CLOSED' } })
    expect(wrapper.find('[data-action="acknowledge"]').exists()).toBe(false)
    expect(wrapper.find('textarea').exists()).toBe(false)
  })
  it('shows errors and re-enables actions after a failed request', async () => {
    vi.mocked(requestJson).mockResolvedValueOnce([]).mockRejectedValueOnce(new Error('请重新加载'))
    const wrapper = mount(CaseFollowup, { props: { caseItem } })
    await flushPromises()
    await wrapper.get('[data-action="acknowledge"]').trigger('click')
    await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('请重新加载')
    expect(wrapper.get('[data-action="acknowledge"]').attributes('disabled')).toBeUndefined()
  })
})
