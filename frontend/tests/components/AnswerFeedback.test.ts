import { mount, flushPromises } from '@vue/test-utils'
import { describe, it, expect, vi } from 'vitest'
import MessageList from '@/components/student/MessageList.vue'
import { requestJson } from '@/core/api'
vi.mock('@/core/api', () => ({requestJson: vi.fn().mockResolvedValue({saved:true})}))
vi.mock('element-plus', async (importOriginal) => ({...await importOriginal<typeof import('element-plus')>(), ElMessage:{success:vi.fn(),error:vi.fn()}}))

describe('answer feedback', () => {
  it('does not expose feedback controls for completed messages', () => {
    const wrapper = mount(MessageList, {props:{messages:[{id:'a',messageId:12,runId:'run-a',role:'assistant',content:'回答'}],phase:'done',error:''},global:{stubs:{'el-dialog':true}}})
    expect(wrapper.find('.feedback-actions').exists()).toBe(false)
    expect(wrapper.find('select').exists()).toBe(false)
    expect(wrapper.text()).toContain('本次回复未提供可核对的来源')
  })
  it('does not present feedback for an unfinished message', () => {
    const wrapper = mount(MessageList, {props:{messages:[{id:'a',role:'assistant',content:'正在生成'}],phase:'streaming',error:''},global:{stubs:{'el-dialog':true}}})
    expect(wrapper.find('.feedback-actions').exists()).toBe(false)
  })
})

 it('opens the actual evidence dialog after a source click', async () => {
  vi.mocked(requestJson).mockResolvedValueOnce({source:'who.md',title:'压力自助',content:'实际原文',sourceUrl:'https://www.who.int/',license:'CC BY-NC-SA'})
  const wrapper = mount(MessageList, {attachTo:document.body,props:{messages:[{id:'a',runId:'run-a',role:'assistant',content:'建议[chunk:1]',citations:[{source:'who.md',chunkId:1,title:'压力自助',preview:'原文摘要',cited:true}]}],phase:'done',error:''}})
  await wrapper.find('.source-link').trigger('click')
  await flushPromises()
  expect(document.body.querySelector('[role="dialog"]')?.textContent).toContain('实际原文')
  expect(document.body.querySelector('[role="dialog"] a')?.getAttribute('href')).toBe('https://www.who.int/')
  wrapper.unmount()
 })
