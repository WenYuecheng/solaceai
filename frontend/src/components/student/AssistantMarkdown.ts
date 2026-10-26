import { defineComponent, h, type PropType, type VNodeChild } from 'vue'
import MarkdownIt from 'markdown-it'
import type { Token } from 'markdown-it'
import type { EvidenceCitation } from '@/types/api'

const markdown = new MarkdownIt({ html: false, breaks: true, linkify: false })
const safeLink = (url: string) => /^(https?:\/\/|mailto:)/i.test(url) && !/[\u0000-\u0020]/.test(url)

export default defineComponent({
  name: 'AssistantMarkdown',
  props: {
    content: { type: String, required: true },
    citations: { type: Array as PropType<EvidenceCitation[]>, default: () => [] },
    interactive: Boolean,
  },
  emits: { evidence: (_citation: EvidenceCitation) => true },
  setup(props, { emit }) {
    function textNodes(content: string): VNodeChild[] {
      return content.split(/(\[chunk:[^\]\r\n]+\])/g).map(part => {
        const id = /^\[chunk:([^\]]+)\]$/.exec(part)?.[1]
        const citation = id ? props.citations.find(item => String(item.chunkId) === id) : undefined
        return citation && props.interactive
          ? h('button', { type: 'button', class: 'inline-citation', onClick: () => emit('evidence', citation), 'aria-label': `查看依据：${citation.title || citation.source}` }, part)
          : part
      })
    }
    function renderTokens(tokens: Token[]): VNodeChild[] {
      const root: VNodeChild[] = []
      const stack: VNodeChild[][] = [root]
      for (const token of tokens) {
        const current = stack[stack.length - 1]!
        if (token.nesting === -1) { if (stack.length > 1) stack.pop(); continue }
        if (token.type === 'inline') { current.push(...renderTokens(token.children || [])); continue }
        if (token.type === 'text') { current.push(...textNodes(token.content)); continue }
        if (token.type === 'code_inline') { current.push(h('code', token.content)); continue }
        if (token.type === 'fence' || token.type === 'code_block') { current.push(h('pre', [h('code', token.content)])); continue }
        if (token.type === 'image') { current.push(token.content); continue }
        if (token.type === 'softbreak' || token.type === 'hardbreak') { current.push(h('br')); continue }
        if (token.type === 'hr') { current.push(h('hr')); continue }
        if (token.nesting === 1) {
          const children: VNodeChild[] = []
          const href = String(token.attrGet('href') || '')
          const tag = token.type === 'link_open' && !safeLink(href) ? 'span' : token.tag
          const attrs = tag === 'a' ? { href, target: '_blank', rel: 'noopener noreferrer' } : token.type === 'ordered_list_open' ? { start: token.attrGet('start') || undefined } : {}
          current.push(h(tag, attrs, children)); stack.push(children)
        }
      }
      return root
    }
    return () => h('div', { class: 'assistant-markdown' }, renderTokens(markdown.parse(props.content, {})))
  },
})
