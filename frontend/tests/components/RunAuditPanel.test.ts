import { mount, flushPromises } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import RunAuditPanel from '@/components/admin/RunAuditPanel.vue'
import { requestJson } from '@/core/api'
vi.mock('@/core/api', () => ({ requestJson: vi.fn() }))
const trace = { id: 7, sessionId: 'session', intent: 'CONSULT', riskLevel: 'LOW', createdAt: '', sanitizedInput: '失眠', agentSteps: [{ agent: 'KnowledgeAgent', action: 'retrieve', observation: '查阅资料' }, { kind: 'emotion_observation', payload: { label: '焦虑', status: 'ok', model: 'local-ft' } }], retrievedKnowledge: [{ source: 'campus.txt', content: '校园心理咨询预约说明' }] }
describe('run audit panel', () => {
  beforeEach(() => { vi.mocked(requestJson).mockReset() })
  it('loads traces and queue only when mounted and presents local observation and evidence', async () => {
    vi.mocked(requestJson).mockImplementation(async (path) => path.endsWith('agent-traces') ? [trace] : path.endsWith('tool-jobs') ? [{ id: 1, kind: 'CASE_CREATE', status: 'PENDING', attempts: 0, maxAttempts: 3 }] : [{ id: 2, kind: 'ALERT_SEND', reason: '投递失败' }])
    const wrapper = mount(RunAuditPanel)
    await flushPromises()
    expect(requestJson).toHaveBeenCalledTimes(4)
    expect(wrapper.text()).toContain('焦虑')
    expect(wrapper.text()).toContain('local-ft')
    expect(wrapper.text()).toContain('校园心理咨询预约说明')
    expect(wrapper.text()).toContain('确定性检索')
    expect(wrapper.text()).toContain('投递失败')
  })
  it('handles malformed older emotion trace and retries failed loads', async () => {
    vi.mocked(requestJson).mockRejectedValue(new Error('读取失败'))
    const wrapper = mount(RunAuditPanel)
    await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('读取失败')
    vi.mocked(requestJson).mockImplementation(async (path) => path.endsWith('agent-traces') ? [{ ...trace, agentSteps: [{ agent: 'EmotionClassifier', observation: 'bad json' }] }] : [])
    await wrapper.get('button').trigger('click')
    await flushPromises()
    expect(wrapper.text()).toContain('暂无分类观察')
    expect(wrapper.find('[role="alert"]').exists()).toBe(false)
  })
})
