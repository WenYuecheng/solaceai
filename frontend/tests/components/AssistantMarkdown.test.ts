import { mount } from '@vue/test-utils'
import { describe, it, expect } from 'vitest'
import AssistantMarkdown from '@/components/student/AssistantMarkdown'
import MessageSources from '@/components/student/MessageSources.vue'
import MessageList from '@/components/student/MessageList.vue'
const citations = [{source:'书籍',title:'章节标题',sectionPath:['第一章','情绪'],chunkId:'abc',preview:'预览',cited:true},{source:'参考资料',chunkId:'def',cited:false}]
describe('safe assistant Markdown', () => {
  it('renders Markdown and incomplete fenced streaming content', () => {
    const wrapper = mount(AssistantMarkdown, {props:{content:'# 标题\n\n- **重点**\n- `code`\n\n```js\nunclosed <script>'}})
    expect(wrapper.find('h1').text()).toBe('标题')
    expect(wrapper.find('strong').text()).toBe('重点')
    expect(wrapper.findAll('li')).toHaveLength(2)
    expect(wrapper.find('pre').text()).toContain('unclosed <script>')
  })
  it('escapes HTML and rejects dangerous links and image fetches', () => {
    const wrapper = mount(AssistantMarkdown, {props:{content:'<img src=x onerror=alert(1)>\n<script>alert(1)</script>\n[x](javascript:alert(1)) [x](data:text/html,evil) ![image](https://evil.test/a) [safe](https://example.com)'}})
    expect(wrapper.find('script').exists()).toBe(false)
    expect(wrapper.find('img').exists()).toBe(false)
    expect(wrapper.findAll('a')).toHaveLength(1)
    expect(wrapper.find('a').attributes('rel')).toBe('noopener noreferrer')
    expect(wrapper.text()).toContain('<script>')
  })
  it('links known citations only, leaving code intact', async () => {
    const wrapper = mount(AssistantMarkdown, {props:{content:'依据[chunk:abc] 未知[chunk:other] `不点击[chunk:abc]`',citations,interactive:true}})
    expect(wrapper.findAll('button')).toHaveLength(1)
    await wrapper.find('button').trigger('click')
    expect(wrapper.emitted('evidence')?.[0]).toEqual([citations[0]])
    expect(wrapper.text()).toContain('[chunk:other]')
  })
  it('shows expanded metadata and renders historical user text literally', () => {
    const wrapper = mount(MessageList, {props:{messages:[{id:'u',role:'user',content:'**普通文本** <script>x</script>'},{id:'a',role:'assistant',content:'回答',citations,runId:'run-a'}],phase:'done',error:''},global:{stubs:{'el-dialog':true}}})
    expect(wrapper.find('details').attributes()).toHaveProperty('open')
    expect(wrapper.text()).toContain('章节标题')
    expect(wrapper.text()).toContain('第一章 / 情绪')
    expect(wrapper.text()).toContain('正文已引用')
    expect(wrapper.text()).toContain('检索参考')
    expect(wrapper.find('.message.user strong').exists()).toBe(false)
    expect(wrapper.find('script').exists()).toBe(false)
  })
  it('deduplicates source groups and keeps incomplete citation text inert', () => {
    const sources = mount(MessageSources, {props:{citations:[...citations, citations[0]!],interactive:true}})
    expect(sources.findAll('button')).toHaveLength(2)
    expect(sources.find('summary').text()).toBe('参考来源 · 2 条')
    const partial = mount(AssistantMarkdown, {props:{content:'**还没完成 [chunk:abc',citations,interactive:true}})
    expect(partial.find('button').exists()).toBe(false)
    expect(partial.text()).toContain('[chunk:abc')
  })

})
