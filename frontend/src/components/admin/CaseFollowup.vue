<script setup lang="ts">
import { computed, shallowRef, watch } from 'vue'
import { requestJson } from '@/core/api'
import { formatDateTime } from '@/core/format'
import type { RiskCase } from '@/types/api'
interface CaseNote { id: number; actor: string; note: string; createdAt: string }
const props = defineProps<{ caseItem: RiskCase }>()
const emit = defineEmits<{ updated: [item: RiskCase] }>()
const notes = shallowRef<CaseNote[]>([])
const note = shallowRef(''), reason = shallowRef(''), error = shallowRef('')
const busy = shallowRef(false), loading = shallowRef(false)
const closed = computed(() => props.caseItem.status === 'CLOSED')
let revision = 0
watch(() => props.caseItem.id, async (id, _, onCleanup) => {
  const current = ++revision
  let active = true
  onCleanup(() => { active = false })
  notes.value = []; note.value = ''; reason.value = ''; error.value = ''; loading.value = true
  try {
    const result = await requestJson<CaseNote[]>(`/api/admin/cases/${id}/notes`)
    if (active && current === revision) notes.value = result
  } catch (failure) {
    if (active) error.value = failure instanceof Error ? failure.message : '跟进记录读取失败'
  } finally { if (active) loading.value = false }
}, { immediate: true })
async function act(action: 'acknowledge' | 'notes' | 'close') {
  if (busy.value || closed.value) return
  const id = props.caseItem.id, current = revision
  busy.value = true; error.value = ''
  try {
    const body = action === 'notes' ? { note: note.value.trim() } : action === 'close' ? { reason: reason.value.trim() } : undefined
    const result = await requestJson<RiskCase | CaseNote>(`/api/admin/cases/${id}/${action}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, ...(body ? { body: JSON.stringify(body) } : {}) })
    if (current !== revision) return
    if (action !== 'notes') emit('updated', result as RiskCase)
    else { note.value = ''; emit('updated', { ...props.caseItem }) }
    if (action === 'close') reason.value = ''
    const history = await requestJson<CaseNote[]>(`/api/admin/cases/${id}/notes`)
    if (current === revision) notes.value = history
  } catch (failure) {
    if (current === revision) error.value = failure instanceof Error ? failure.message : '个案操作失败'
  } finally { busy.value = false }
}
</script>

<template>
  <section class="case-followup" aria-label="个案跟进">
    <h4>跟进处理</h4>
    <p v-if="error" role="alert" class="error">{{ error }}</p>
    <p v-if="closed">该个案已结案。</p>
    <template v-else>
      <button v-if="caseItem.status !== 'ACKNOWLEDGED'" data-action="acknowledge" class="secondary-button" :disabled="busy" @click="act('acknowledge')">{{ busy ? '处理中…' : '确认接手' }}</button>
      <label :for="`note-${caseItem.id}`">跟进记录</label>
      <textarea :id="`note-${caseItem.id}`" v-model="note" maxlength="2000" rows="3" :disabled="busy" placeholder="记录访谈与处置进展" />
      <button data-action="notes" class="secondary-button" :disabled="busy || !note.trim()" @click="act('notes')">添加记录</button>
      <label :for="`reason-${caseItem.id}`">结案原因</label>
      <textarea :id="`reason-${caseItem.id}`" v-model="reason" maxlength="2000" rows="2" :disabled="busy || caseItem.status !== 'ACKNOWLEDGED'" placeholder="完成跟进后，说明结案依据" />
      <p v-if="caseItem.status !== 'ACKNOWLEDGED'" class="hint">请先确认接手，再结案。</p>
      <button data-action="close" class="secondary-button" :disabled="busy || caseItem.status !== 'ACKNOWLEDGED' || !reason.trim()" @click="act('close')">结案</button>
    </template>
    <h4>跟进历史</h4>
    <p v-if="loading" role="status">正在读取…</p>
    <p v-else-if="!notes.length">暂无跟进记录</p>
    <ol v-else><li v-for="item in notes" :key="item.id"><div>{{ item.actor }} · {{ formatDateTime(item.createdAt) }}</div><p>{{ item.note }}</p></li></ol>
  </section>
</template>

<style scoped>
.case-followup { margin-top: 24px; border-top: 1px solid var(--line); }.case-followup label { display: block; margin: 18px 0 8px; }.case-followup textarea { box-sizing: border-box; width: 100%; padding: 12px; margin-bottom: 8px; font: inherit; color: var(--text); background: var(--surface-soft); border: 1px solid var(--line); border-radius: 10px; resize: vertical; }.case-followup ol { padding-left: 20px; }.case-followup li { margin-bottom: 16px; }.case-followup li div, .hint { color: var(--text-faint); font-size: 12px; }.case-followup li p { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.6; }.error { color: var(--danger); }
</style>
