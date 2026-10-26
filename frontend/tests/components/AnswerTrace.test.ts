import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import AnswerTrace from '@/components/student/AnswerTrace.vue'

describe('AnswerTrace', () => {
  it('expands each answer trace by default with why, outcome and timing', () => {
    const wrapper = mount(AnswerTrace, { props: { trace: { runId: 'r', status: 'completed', steps: [
      { id: 's', sequence: 1, stage: 'retrieval', title: '检索参考资料', why: '寻找依据', result: '找到资料', status: 'completed', durationMs: 125, startedAt: '2026-10-21T00:00:00Z', count: 2 },
    ] } } })
    expect(wrapper.get('details').attributes('open')).toBeDefined()
    expect(wrapper.text()).toContain('寻找依据')
    expect(wrapper.text()).toContain('找到资料')
    expect(wrapper.text()).toContain('125 ms')
    expect(wrapper.text()).toContain('2')
  })
  it('labels old replies without inventing historical execution', () => {
    expect(mount(AnswerTrace).text()).toContain('此历史回复未记录执行过程')
  })
})

it('shows ReAct phases, agent, round and actual tool on timeline steps', () => {
  const wrapper = mount(AnswerTrace, { props: { trace: { runId: 'react', status: 'completed', steps: [
    { id: 'reason', sequence: 1, stage: 'agent', phase: 'reason', actor: 'MainAgent', round: 2, nextAction: 'search_knowledge', title: '决策摘要', why: '选择后续动作', result: '检索参考资料', status: 'completed', durationMs: 250, startedAt: '2026-10-21T00:00:00Z' },
    { id: 'act', sequence: 2, stage: 'retrieval', phase: 'act', actor: 'RetrievalAgent', round: 2, tool: 'search_knowledge', title: '检索参考资料', why: '寻找依据', result: '执行完成', status: 'completed', durationMs: 100, startedAt: '2026-10-21T00:00:00Z' },
    { id: 'observe', sequence: 3, stage: 'observation', phase: 'observe', actor: 'MainAgent', round: 2, tool: 'search_knowledge', title: '观察工具结果', why: '整理结果', result: '已接收结果', count: 3, status: 'completed', durationMs: 1, startedAt: '2026-10-21T00:00:00Z' },
  ] } } })
  expect(wrapper.get('[data-phase="reason"]').text()).toContain('Reason')
  expect(wrapper.get('[data-phase="act"]').text()).toContain('Act')
  expect(wrapper.get('[data-phase="observe"]').text()).toContain('Observe')
  expect(wrapper.text()).toContain('第 2 轮')
  expect(wrapper.text()).toContain('MainAgent')
  expect(wrapper.text()).toContain('search_knowledge')
})


it('shows actual per-stage ranked snippets and escapes document markup', () => {
  const wrapper = mount(AnswerTrace, { props: { trace: { runId: 'details', status: 'completed', steps: [
    { id: 'dense', sequence: 1, stage: 'recall', title: '多路召回', why: '检索', result: '完成', status: 'completed', durationMs: 100, startedAt: '2026-10-22T00:00:00Z', retrievalDetails: [
      { channel: 'dense', queryIndex: 1, method: 'cosine', total: 1, candidates: [{ rank: 1, chunkId: 9, source: '压力管理', content: '<img src=x onerror=alert(1)>呼吸练习', score: .875 }] },
      { channel: 'sparse', queryIndex: 1, method: 'inner_product', total: 0, candidates: [] },
    ] },
  ] } } })
  expect(wrapper.text()).toContain('密集向量召回（Dense） · 1 条')
  expect(wrapper.text()).toContain('0.875000')
  expect(wrapper.text()).toContain('压力管理')
  expect(wrapper.text()).toContain('本阶段无命中资料')
  expect(wrapper.find('img').exists()).toBe(false)
})
