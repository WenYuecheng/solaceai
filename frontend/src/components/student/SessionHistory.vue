<script setup lang="ts">
import ConfirmDialog from '@/components/shared/ConfirmDialog.vue'
import { onMounted, ref, watch } from 'vue'
import { request } from '@/core/api'
interface SessionSummary { sessionId: string; title: string; updatedAt: string }
const props = defineProps<{ activeSession: string | null; revision?: number }>()
const emit = defineEmits<{ select: [sessionId: string]; removed: [sessionId: string]; removing: [sessionId: string] }>()
const items = ref<SessionSummary[]>([])
const total = ref(0)
const busy = ref(false)
const error = ref('')
const pendingRemoval = ref<SessionSummary | null>(null)
let generation = 0
async function load(more = false) {
  const current = ++generation
  busy.value = true
  error.value = ''
  try {
    const response = await request(`/api/sessions?limit=50&offset=${more ? items.value.length : 0}`)
    const data = await response.json() as { items: SessionSummary[]; total: number }
    if (current !== generation) return
    items.value = more ? [...items.value, ...data.items] : data.items
    total.value = data.total
  } catch (reason) {
    if (current === generation) error.value = reason instanceof Error ? reason.message : '无法加载历史会话'
  } finally { if (current === generation) busy.value = false }
}
function remove(item: SessionSummary) { pendingRemoval.value = item }
async function confirmRemoval() {
  const item = pendingRemoval.value
  if (!item || busy.value) return
  pendingRemoval.value = null
  emit('removing', item.sessionId)
  busy.value = true
  error.value = ''
  try {
    await request(`/api/sessions/${encodeURIComponent(item.sessionId)}`, { method: 'DELETE' })
    emit('removed', item.sessionId)
    await load()
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : '移除失败，请重试'
  } finally { busy.value = false }
}
onMounted(() => { void load() })
watch(() => props.revision, () => { void load() })
</script>

<template>
  <section class="history-panel" aria-labelledby="history-title" :aria-busy="busy">
    <header><h2 id="history-title">历史会话</h2><button type="button" :disabled="busy" @click="load()">刷新</button></header>
    <p v-if="error" role="alert">{{ error }}</p>
    <p v-else-if="!items.length">{{ busy ? '正在加载…' : '还没有历史会话，从今天的一句话开始。' }}</p>
    <ul v-if="items.length">
      <li v-for="item in items" :key="item.sessionId">
        <button class="session-select" type="button" :disabled="busy" :aria-current="activeSession === item.sessionId ? 'true' : undefined" @click="emit('select', item.sessionId)">{{ item.title || '未命名会话' }}</button>
        <button class="remove" type="button" :disabled="busy" :aria-label="`移除会话：${item.title}`" @click="remove(item)">移除</button>
      </li>
    </ul>
    <button v-if="items.length < total" type="button" :disabled="busy" @click="load(true)">加载更多</button>
    <ConfirmDialog :open="!!pendingRemoval" title="移除这条会话？" message="它将从你的历史中隐藏，不能再继续；已授权的心理报告与个案审计记录会保留。" confirm-label="移除会话" @confirm="confirmRemoval" @cancel="pendingRemoval = null" />
  </section>
</template>

<style scoped>
.history-panel { padding: 0; border: 0; background: transparent; }
header, li { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
h2 { margin: 0; font-family: var(--font-playful, var(--font-body)); font-size: 17px; font-weight: 400; } p { font-size: 12px; color: var(--text-soft); }
ul { list-style: none; margin: 12px 0 0; padding: 0; max-height: 230px; overflow: auto; }
button { border: 0; padding: 7px; border-radius: 8px; background: transparent; color: #5f7053; cursor: pointer; font-size: 12px; }
button:disabled { cursor: wait; opacity: .6; } .session-select { min-width: 0; text-align: left; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.session-select[aria-current='true'], .session-select:hover { background: #e1e9ce; color: #4e6750; } .remove { flex-shrink: 0; }
</style>
