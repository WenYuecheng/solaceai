<script setup lang="ts">
import { computed, nextTick, useTemplateRef } from 'vue'
import type { ChatPhase } from '@/stores/chat'

const props = defineProps<{ phase: ChatPhase }>()
const emit = defineEmits<{ send: [message: string]; stop: []; openQuick: [] }>()
const draft = defineModel<string>({ default: '' })
const input = useTemplateRef<HTMLTextAreaElement>('input')
const active = computed(() => props.phase === 'thinking' || props.phase === 'streaming')

async function submit() {
  const message = draft.value.trim()
  if (!message || active.value) return
  emit('send', message)
  draft.value = ''
  await nextTick()
  input.value?.focus()
}

function focus() { input.value?.focus() }
defineExpose({ focus })
</script>

<template>
  <form class="composer" @submit.prevent="submit">
    <div class="composer-box">
      <label class="sr-only" for="student-message">写下你想说的话</label>
      <textarea id="student-message" ref="input" v-model="draft" name="message" maxlength="1000" rows="2" placeholder="慢慢写，我在这里听着…" :disabled="active"></textarea>
      <div class="composer-meta">
        <span>对话仅用于心理支持</span><span class="mono">{{ draft.length }} / 1000</span>
      </div>
    </div>
    <div class="composer-actions">
      <button class="quick-trigger secondary-button" type="button" @click="emit('openQuick')">情绪快捷入口</button>
      <span class="send-hint">Enter 换行 · 点击按钮发送</span>
      <button v-if="active" class="stop-button secondary-button" type="button" @click="emit('stop')"><i aria-hidden="true"></i>停止回复</button>
      <button v-else class="primary-button send-button" type="submit" :disabled="!draft.trim()">
        发送消息
      </button>
    </div>
  </form>
</template>

<style scoped>
.composer { position: relative; z-index: 3; display: grid; gap: 15px; padding: 15px 22px 18px; border-top: 1px solid var(--line); background: #fffef9c9;  }
.composer-box { overflow: hidden; border: 1px solid var(--line); border-radius: 18px; background: #f9faef; transition: border-color var(--duration-fast), box-shadow var(--duration-fast); }
.composer-box:focus-within { border-color: var(--aqua); box-shadow: 0 0 0 4px color-mix(in srgb, var(--aqua) 12%, transparent); }
.composer textarea { width: 100%; min-height: 64px; display: block; padding: 15px 16px 7px; resize: none; max-height: 140px; border: 0; outline: 0; color: var(--text); background: transparent; line-height: 1.65; }
.composer textarea::placeholder { color: var(--text-faint); }
.composer-meta { display: flex; justify-content: space-between; gap: 14px; padding: 0 15px 11px; color: var(--text-faint); font-size: 11px; }
.composer-actions { display: flex; align-items: center; justify-content: flex-end; gap: 10px; }
.send-hint { margin-right: auto; color: var(--text-faint); font-size: 11px; }.send-button { display: inline-flex; align-items: center; gap: 11px; border-radius: 13px; background: #4e6750; }.quick-trigger { display: inline-flex; align-items: center; justify-content: center; min-height: 44px; padding: 0 14px; line-height: 1.3; border-radius: 12px; font-size: 12px; }.stop-button { display: inline-flex; align-items: center; gap: 8px; }.stop-button i { width: 9px; height: 9px; border-radius: 2px; background: var(--danger); }
@media (max-width: 720px) { .composer { padding: 12px; }.composer-actions { flex-wrap: wrap; }.quick-trigger { display: inline-flex; }.send-hint { display: none; }.quick-trigger, .send-button, .stop-button { flex: 1; justify-content: center; padding: 0 12px; } }
@media (min-width: 721px) { .composer-box, .composer-actions { width: min(100%, 900px); margin-inline: auto; } }
@media (max-width: 720px) { .composer { padding-bottom: max(13px, env(safe-area-inset-bottom)); }.composer textarea { min-height: 54px; padding-top: 10px; }.composer-meta { padding-bottom: 8px; } }
</style>
