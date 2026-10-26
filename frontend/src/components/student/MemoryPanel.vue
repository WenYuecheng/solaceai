<script setup lang="ts">
import { onMounted, ref, shallowRef } from 'vue'
import { requestJson, request } from '@/core/api'

interface SavedMemory { id: number; content: string }
const memories = ref<SavedMemory[]>([])
const error = shallowRef('')
async function refresh() {
  try { memories.value = await requestJson<SavedMemory[]>('/api/memories'); error.value = '' }
  catch { error.value = '暂时无法读取记忆' }
}
async function remove(id: number) {
  try { await request(`/api/memories/${id}`, { method: 'DELETE' }); memories.value = memories.value.filter(item => item.id !== id) }
  catch { error.value = '删除未完成，请重试' }
}
onMounted(refresh)
</script>

<template>
  <details class="memory-panel">
    <summary>你让我记住的事</summary>
    <p>以“请记住：”开头告诉我你的偏好。你可以随时删除。</p>
    <button type="button" @click="refresh">刷新</button>
    <p v-if="error" role="alert">{{ error }}</p>
    <ul v-if="memories.length">
      <li v-for="memory in memories" :key="memory.id"><span>{{ memory.content }}</span><button type="button" data-delete-memory @click="remove(memory.id)">删除</button></li>
    </ul>
    <p v-else>暂无保存的偏好。</p>
  </details>
</template>

<style scoped>
.memory-panel { padding: 16px 0 0; border: 0; border-top: 1px solid var(--line); border-radius: 0; color: var(--text-soft); background: transparent; font-size: 13px; }
summary { cursor: pointer; font-weight: 600; } p { line-height: 1.7; } ul { padding-left: 18px; } li { margin: 12px 0; overflow-wrap: anywhere; }
button { margin-left: 8px; padding: 5px 9px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface-solid); color: inherit; cursor: pointer; }
</style>
