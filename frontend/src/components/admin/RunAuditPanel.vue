<script setup lang="ts">
import { computed, onMounted, shallowRef } from 'vue'
import { requestJson } from '@/core/api'
import { ElMessage } from 'element-plus'
import { formatDateTime, statusLabel } from '@/core/format'
type Entry = Record<string, unknown>
interface RunTrace { firstTokenSeconds?: number | null; elapsedSeconds?: number | null; runId?: string; id: number; sessionId: string; intent: string; riskLevel: string; createdAt: string; sanitizedInput: string; agentSteps: Entry[]; retrievedKnowledge: Entry[] }
interface Execution { id: number; runId: string; toolName: string; status: string; error?: string; createdAt: string }
interface Job { id: number; kind: string; status: string; attempts: number; maxAttempts: number; lastError?: string }
interface DeadLetter { id: number; jobId?: number; kind: string; reason: string }
const traces = shallowRef<RunTrace[]>([]), jobs = shallowRef<Job[]>([]), deadLetters = shallowRef<DeadLetter[]>([])
const executions = shallowRef<Execution[]>([]), recovering = shallowRef<number | null>(null)
const selectedId = shallowRef<number | null>(null), loading = shallowRef(false), error = shallowRef('')
const selected = computed(() => traces.value.find(trace => trace.id === selectedId.value))
const steps = computed(() => (selected.value?.agentSteps ?? []).filter(step => typeof step.agent === 'string'))
const observation = computed<Entry | null>(() => {
  const records = selected.value?.agentSteps ?? []
  const artifact = records.find(step => step.kind === 'emotion_observation')?.payload
  if (artifact && typeof artifact === 'object') return artifact as Entry
  const value = records.find(step => step.agent === 'EmotionClassifier')?.observation
  try { const parsed = typeof value === 'string' ? JSON.parse(value) : null; return parsed && typeof parsed === 'object' ? parsed : null } catch { return null }
})
const emotionStatus = computed(() => ({ ok: '已完成', disabled: '未启用', offline: '离线演示', unavailable: '暂不可用' }[String(observation.value?.status)] ?? '未知'))
function text(value: unknown): string { return typeof value === 'string' || typeof value === 'number' ? String(value) : '' }
function agentLabel(value: unknown): string { return (value === 'KnowledgeAgent' || value === 'RetrievalAgent') ? 'KnowledgeAgent（确定性检索）' : text(value) }
async function load() {
  if (loading.value) return
  loading.value = true; error.value = ''
  try {
    const [runs, queue, dead, calls] = await Promise.all([requestJson<RunTrace[]>('/api/admin/agent-traces'), requestJson<Job[]>('/api/admin/tool-jobs'), requestJson<DeadLetter[]>('/api/admin/dead-letters'), requestJson<Execution[]>('/api/admin/tool-executions')])
    traces.value = runs; jobs.value = queue; deadLetters.value = dead; executions.value = calls
    if (!runs.some(run => run.id === selectedId.value)) selectedId.value = runs[0]?.id ?? null
  } catch (failure) { error.value = failure instanceof Error ? failure.message : '运行记录读取失败' }
  finally { loading.value = false }
}
async function recover(id: number) {
  recovering.value = id
  try { await requestJson(`/api/admin/tool-executions/${id}/recover`, {method:'POST'}); ElMessage.success('恢复执行完成'); await load() }
  catch (error) { ElMessage.error(error instanceof Error ? error.message : '恢复失败') }
  finally { recovering.value = null }
}
onMounted(load)
</script>

<template>
  <section class="run-audit" aria-label="运行追踪">
    <header><h3>运行追踪与工具执行</h3><button class="secondary-button" :disabled="loading" @click="load">{{ loading ? '读取中…' : '刷新运行记录' }}</button></header>
    <p v-if="error" role="alert" class="error">{{ error }}</p>
    <p v-if="loading" role="status">正在读取运行记录…</p>
    <template v-else>
      <label for="trace-selector">选择运行记录</label>
      <select id="trace-selector" v-model="selectedId"><option v-if="!traces.length" :value="null">暂无运行记录</option><option v-for="run in traces" :key="run.id" :value="run.id">#{{ run.id }} · {{ run.sessionId }} · {{ formatDateTime(run.createdAt) }}</option></select>
      <article v-if="selected">
        <p>{{ statusLabel(selected.intent) }} · {{ statusLabel(selected.riskLevel) }}</p>
        <p v-if="selected.elapsedSeconds != null">总耗时：{{ selected.elapsedSeconds.toFixed(2) }} 秒<span v-if="selected.firstTokenSeconds != null"> · 首次响应：{{ selected.firstTokenSeconds.toFixed(2) }} 秒</span></p><h4>脱敏输入</h4><p class="body-text">{{ selected.sanitizedInput }}</p>
        <h4>本地情绪分类观察</h4>
        <p v-if="observation">{{ text(observation.label) || '未产生标签' }} · {{ emotionStatus }}<br><span class="hint">模型：{{ text(observation.model) || '未记录' }}</span></p>
        <p v-else>暂无分类观察</p>
        <p class="hint">分类结果用于观察，不作为临床诊断或独立的风险处置依据。</p>
        <h4>执行步骤</h4><ol v-if="steps.length"><li v-for="(step, index) in steps" :key="index"><strong>{{ agentLabel(step.agent) }}</strong> · {{ text(step.action) }}<p class="body-text">{{ text(step.observation) }}</p></li></ol><p v-else>暂无步骤记录</p>
        <h4>检索证据</h4><ul v-if="selected.retrievedKnowledge.length"><li v-for="(evidence, index) in selected.retrievedKnowledge" :key="index"><strong>{{ text(evidence.source) || '未记录来源' }}</strong><p class="body-text">{{ text(evidence.content) || text(evidence.preview) || text(evidence.text) }}</p></li></ul><p v-else>此次运行未召回知识片段</p>
      </article>
      <h4>工具执行（{{ executions.length }}）</h4><div class="table-wrap"><table v-if="executions.length"><thead><tr><th>工具</th><th>运行</th><th>状态</th><th>失败信息</th><th>操作</th></tr></thead><tbody><tr v-for="call in executions" :key="call.id"><td>{{ call.toolName }}</td><td>{{ call.runId }}</td><td>{{ call.status }}</td><td>{{ call.error || '—' }}</td><td><button v-if="['FAILED', 'RUNNING'].includes(call.status)" class="secondary-button" :disabled="recovering !== null" @click="recover(call.id)">{{ recovering === call.id ? '恢复中…' : '恢复执行' }}</button></td></tr></tbody></table><p v-else>暂无工具执行</p></div>
      <h4>历史任务记录（只读，{{ jobs.length }}）</h4><div class="table-wrap"><table v-if="jobs.length"><thead><tr><th>任务</th><th>类型</th><th>状态</th><th>尝试次数</th><th>失败信息</th></tr></thead><tbody><tr v-for="job in jobs" :key="job.id"><td>#{{ job.id }}</td><td>{{ job.kind }}</td><td>{{ job.status }}</td><td>{{ job.attempts }}/{{ job.maxAttempts }}</td><td>{{ job.lastError || '—' }}</td></tr></tbody></table><p v-else>暂无工具任务</p></div>
      <h4>历史失败归档（只读，{{ deadLetters.length }}）</h4><ul v-if="deadLetters.length"><li v-for="record in deadLetters" :key="record.id">#{{ record.jobId ?? record.id }} · {{ record.kind }}<p class="body-text">{{ record.reason }}</p></li></ul><p v-else>暂无历史失败归档</p>
    </template>
  </section>
</template>

<style scoped>
.run-audit header { display: flex; justify-content: space-between; align-items: center; gap: 12px; }.run-audit h3 { font-size: 18px; }.run-audit label { display: block; margin-bottom: 8px; }.run-audit select { width: 100%; padding: 10px; background: var(--surface-soft); color: var(--text); border: 1px solid var(--line); border-radius: 10px; }.run-audit article { margin: 20px 0; padding: 16px; background: var(--surface-soft); border: 1px solid var(--line); border-radius: 14px; }.body-text { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.7; }.hint { color: var(--text-faint); font-size: 12px; }.error { color: var(--danger); }.table-wrap { overflow-x: auto; }table { width: 100%; border-collapse: collapse; font-size: 13px; }th, td { padding: 10px; text-align: left; border-bottom: 1px solid var(--line); }li { margin-bottom: 14px; }@media (max-width: 600px) { .run-audit header { align-items: flex-start; flex-direction: column; } }
</style>
