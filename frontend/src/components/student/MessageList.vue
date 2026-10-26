<script setup lang="ts">
import { nextTick, useTemplateRef, watch, ref, computed } from 'vue'
import { requestJson } from '@/core/api'
import { ElDialog, ElMessage } from 'element-plus'
import AssistantMarkdown from './AssistantMarkdown'
import MessageSources from './MessageSources.vue'
import AnswerTrace from './AnswerTrace.vue'
import type { EvidenceCitation } from '@/types/api'
import type { ChatMessage, ChatPhase } from '@/stores/chat'

const props = defineProps<{ messages: ChatMessage[]; phase: ChatPhase; error: string }>()
const emit = defineEmits<{ quick: [message: string]; retry: [] }>()
const scroller = useTemplateRef<HTMLDivElement>('scroller')
const welcomePrompts = ['最近有件事放不下', '说不上来，但很累', '帮我整理一下情绪']
const promptMessages = ['最近发生了一件事，让我一直放不下。', '我说不上来原因，但最近总觉得很累。', '我想先整理一下现在的情绪。']

const evidence = ref<{source: string; title?: string; sourceUrl?: string; license?: string; content: string; sectionPath?: string[]; documentVersion?: string} | null>(null)
const evidenceOpen = ref(false)
const sourceUrl = computed(() => {
  const url = evidence.value?.sourceUrl || ''
  return /^https?:\/\/[^\s\x00-\x1f]+$/i.test(url) ? url : ''
})
async function openEvidence(message: ChatMessage, citation: EvidenceCitation) {
  if (!message.runId || citation.chunkId == null) return
  try { evidence.value = await requestJson(`/api/evidence/${encodeURIComponent(message.runId)}/${encodeURIComponent(String(citation.chunkId))}`); evidenceOpen.value = true }
  catch (error) { ElMessage.error(error instanceof Error ? error.message : '无法读取依据') }
}
watch(() => props.messages.map((item) => item.content).join(''), async () => {
  await nextTick()
  if (scroller.value) scroller.value.scrollTop = scroller.value.scrollHeight
})
</script>

<template>
  <div ref="scroller" class="message-list" aria-live="polite" aria-label="对话消息">
    <div v-if="!messages.length" class="welcome-state">
      <h2>今天的心情，想从哪里说起？</h2>
      <p>一件小事，一种心情，都可以。<br>不必整理好语言，想到什么就写下来。</p>
      <div class="welcome-actions">
        <button v-for="(label, index) in welcomePrompts" :key="label" type="button" @click="emit('quick', promptMessages[index])">{{ label }}</button>
      </div>
    </div>
    <article v-for="message in messages" :key="message.id" class="message" :class="message.role">
      <div class="message-label">{{ message.role === 'user' ? '你' : 'SolaceAI' }}</div>
      <AnswerTrace v-if="message.role === 'assistant'" :trace="message.answerTrace" :active="message === messages.at(-1) && (phase === 'thinking' || phase === 'streaming')" />
      <div v-if="message.role === 'assistant' && message.content" class="reply-heading"><span aria-hidden="true">❀</span> 给你的回复</div>
      <div class="bubble"><AssistantMarkdown v-if="message.role === 'assistant'" :content="message.content" :citations="message.citations" :interactive="!!message.runId" @evidence="openEvidence(message, $event)" /><template v-else>{{ message.content }}</template><span v-if="message.role === 'assistant' && phase === 'streaming' && message === messages.at(-1)" class="stream-caret" aria-label="正在回复"></span></div>
      <p v-if="message.status && !message.content" class="message-status" role="status">{{ message.status }}</p>
      <MessageSources v-if="message.citations?.length" :citations="message.citations" :interactive="!!message.runId" @evidence="openEvidence(message, $event)" />
      <p v-if="message.role === 'assistant' && !message.citations?.length && message.content && (message !== messages.at(-1) || phase === 'done' || phase === 'idle')" class="message-status">本次回复未提供可核对的来源</p>
    </article>
    <el-dialog v-model="evidenceOpen" title="参考原文" width="min(700px, 92vw)"><template v-if="evidence"><h3>{{ evidence.title || evidence.source }}</h3><p v-if="sourceUrl"><a :href="sourceUrl" target="_blank" rel="noopener noreferrer">打开资料原始来源</a></p><small v-if="evidence.license">{{ evidence.license }}</small><p>{{ evidence.sectionPath?.join(' / ') }}</p><p class="evidence-content">{{ evidence.content }}</p><small v-if="evidence.documentVersion">文档版本：{{ evidence.documentVersion.slice(0, 12) }}</small></template></el-dialog>
    <div v-if="error" class="chat-error" role="alert"><span>{{ error }}</span><button type="button" @click="emit('retry')">重新发送</button></div>
  </div>
</template>

<style scoped>
.source-link { color: var(--text-soft); background: transparent; border: 0; cursor: pointer; text-decoration: underline; } .evidence-content { white-space: pre-wrap; line-height: 1.8; }
.bubble :deep(.assistant-markdown) { white-space: normal; }
.bubble :deep(.assistant-markdown > :first-child) { margin-top: 0; }
.bubble :deep(.assistant-markdown > :last-child) { margin-bottom: 0; }
.bubble :deep(pre) { white-space: pre; overflow-x: auto; padding: 12px; background: var(--surface-solid); border-radius: 8px; }
.bubble :deep(code) { font-family: var(--font-mono); }
.bubble :deep(a), .bubble :deep(.inline-citation) { color: var(--lagoon); text-decoration: underline; }
.bubble :deep(.inline-citation) { background: transparent; border: 0; font: inherit; cursor: pointer; }
.reply-heading { margin: 0 0 8px; color: #4e6750; font-size: 12px; font-weight: 600; }.reply-heading span { color: #91a578; font-size: 18px; margin-right: 5px; }
.message-status, .message-sources { color: var(--text-soft); font-size: 12px; line-height: 1.6; margin: 8px 4px; }
.message-sources summary { cursor: pointer; }
.message-list { min-height: 0; overflow-y: auto; overscroll-behavior: contain; padding: 25px clamp(20px, 3vw, 44px); scroll-behavior: smooth; background: transparent; }
.welcome-state { min-height: 100%; display: grid; place-content: center; justify-items: center; padding: 16px 12px; text-align: center; }

.welcome-state h2 { max-width: 700px; margin: 6px 0 0; font-family: var(--font-playful, var(--font-body)); font-size: clamp(25px, 2.4vw, 34px); font-weight: 400; line-height: 1.3; letter-spacing: -.03em; text-wrap: balance; }.welcome-state > p:not(.eyebrow) { max-width: 590px; margin: 15px 0 0; color: #5f7053; font-size: 13px; line-height: 1.75; }.welcome-actions { display: flex; justify-content: center; gap: 9px; margin-top: 25px; flex-wrap: wrap; }.welcome-actions button { min-height: 40px; padding: 0 15px; border: 1px solid var(--line); border-radius: 14px; color: #60764f; background: #f1f5e4; font-size: 12px; transition: transform var(--duration-fast), border-color var(--duration-fast); }.welcome-actions button:hover {  border-color: var(--line-strong); }
.message { max-width: min(760px, 100%); margin: 0 0 22px;  }.message.user { margin-left: auto; max-width: 85%; }.message-label { margin: 0 4px 7px; color: var(--text-faint); font-family: var(--font-body); font-size: 11px; }.message.user .message-label { text-align: right; }.bubble { padding: 10px 0; border: 0; border-radius: 0; background: #fffef9e8; box-shadow: none; color: #354537; font-size: 15px; line-height: 1.95; white-space: pre-wrap; overflow-wrap: anywhere; }.message.user .bubble { border: 0; padding: 14px 18px; border-radius: 19px 19px 4px 19px; color: #43573d; background: #eaf0dd; font-size: 14px; }.stream-caret { width: 7px; height: 1.1em; display: inline-block; margin-left: 4px; vertical-align: -.18em; border-radius: 4px; background: var(--aqua); animation: caret 750ms steps(2) infinite; }
.chat-error { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin: 0 auto 18px; padding: 13px 15px; border: 1px solid color-mix(in srgb, var(--danger) 35%, transparent); border-radius: 13px; color: var(--danger); background: color-mix(in srgb, var(--danger) 8%, var(--surface-raised)); }.chat-error button { border: 0; color: inherit; background: transparent; font-weight: 700; text-decoration: underline; }
@keyframes caret { 50% { opacity: .2; } }
@media (max-width: 720px) { .message-list { min-height: 0; padding: 20px 18px; }.welcome-state { min-height: 0; padding: 22px 10px; }.welcome-state h2 { font-size: 32px; }.welcome-actions { margin-top: 18px; }.message { max-width: 92%; } }
@media (prefers-reduced-motion: reduce) { .stream-caret { animation: none; } }
@media (max-width: 720px) { .welcome-state h2 { font-size: 27px; }.welcome-state > p:not(.eyebrow) { font-size: 13px; margin-top: 12px; }.welcome-actions { gap: 7px; }.welcome-actions button { min-height: 40px; font-size: 12px; }.message { max-width: 96%; } }
@media (max-height: 740px) { .welcome-state { padding-block: 8px; }.welcome-state h2 { font-size: 28px; }.welcome-actions { margin-top: 16px; } }

.welcome-state { min-height: 100%; padding: 40px 12px; }
.welcome-state h2 { max-width: 580px; padding: 12px 18px; border-radius: 20px; background: #fffef9d9; }
.welcome-state > p:not(.eyebrow) { padding: 10px 18px; border-radius: 16px; background: #fffef9dc; color: #526347; }
.welcome-actions button { background: #fffef9ed; color: #43583c; border-color: #c7d4b6; }
.bubble { padding: 14px 18px; border-radius: 16px; }
@media (max-width: 720px) { .welcome-state { min-height: 100%; padding: 24px 4px; }.welcome-state h2 { font-size: 26px; }.welcome-state > p:not(.eyebrow) { font-size: 12px; } }

.message.assistant .bubble { background: transparent; border: 0; padding: 10px 2px; border-radius: 0; font-family: var(--font-playful); font-weight: 400; font-size: 18px; line-height: 1.9; letter-spacing: .015em; color: #344a34; }
.message.assistant .bubble :deep(strong) { font-weight: 400; color: #2c492f; text-decoration: underline; text-decoration-color: #92a678; text-underline-offset: 4px; }
.message.assistant .bubble :deep(code), .message.assistant .bubble :deep(pre) { font-family: var(--font-mono); font-size: 13px; letter-spacing: 0; }
@media (max-width: 720px) { .message.assistant .bubble { font-size: 17px; line-height: 1.9; } }
</style>
