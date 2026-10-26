<script setup lang="ts">
import { computed } from 'vue'
import { useChatStore } from '@/stores/chat'
import { useSystemStore } from '@/stores/system'
import type { ChatBackend } from '@/types/api'

const system = useSystemStore()
const chat = useChatStore()
const disabled = computed(() => !system.agent || system.availableChatBackends.length < 2 || chat.phase === 'thinking' || chat.phase === 'streaming')
function choose(backend: ChatBackend) {
  if (!disabled.value) system.selectChatBackend(backend)
}
</script>

<template>
  <div class="chat-model-selector" :class="{ 'single-model': system.availableChatBackends.length === 1 }" role="group" aria-label="对话模型">
    <button v-for="backend in system.availableChatBackends" :key="backend" class="model-choice" type="button" :data-backend="backend" :aria-pressed="system.chatBackend === backend" :title="backend === 'local' ? '本地 · Qwen3' : '云端 · DeepSeek Flash'" :disabled="disabled" @click="choose(backend)">
      {{ backend === 'local' ? 'Qwen3' : 'DeepSeek Flash' }}
    </button>
  </div>
</template>

<style scoped>
.chat-model-selector { display: grid; grid-template-columns: 1.65fr 1fr; flex: 0 0 auto; width: 224px; max-width: 100%; min-width: 0; height: 38px; padding: 3px; gap: 3px; border: 1px solid #d8e1cc; border-radius: 13px; background: #f4f6e9; }
.model-choice { display: grid; place-items: center; min-width: 0; padding: 0 8px; border: 1px solid transparent; border-radius: 9px; background: transparent; color: #61705b; font: 500 12px/1.2 var(--font-body); white-space: nowrap; cursor: pointer; transition: background-color .15s, color .15s, border-color .15s; }
.model-choice[aria-pressed="true"] { border-color: #c3d0b5; color: #344d39; background: #e2ead6; box-shadow: 0 1px 2px #35482b08; }
.model-choice:hover:not(:disabled):not([aria-pressed="true"]) { background: #ebefdf; color: #344d39; }
.model-choice:focus-visible { outline: 2px solid #597c48; outline-offset: 1px; }
.model-choice:disabled { cursor: default; }.model-choice:disabled:not([aria-pressed="true"]) { color: #7a8473; }
.single-model { width: 142px; grid-template-columns: 1fr; }
@media (max-width: 620px) { .chat-model-selector { width: 100%; max-width: 224px; height: 36px; }.model-choice { padding-inline: 4px; font-size: 11px; }.single-model { max-width: 142px; } }
@media (prefers-reduced-motion: reduce) { .model-choice { transition: none; } }
</style>
