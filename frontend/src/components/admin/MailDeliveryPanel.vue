<script setup lang="ts">
import { computed, onMounted, shallowRef } from 'vue'
import { requestJson } from '@/core/api'
import type { ToolRecord } from '@/types/api'

const props = defineProps<{ alerts: ToolRecord[] }>()
const emit = defineEmits<{ refresh: [record: DeliveryResult] }>()
interface MailStatus { mode: string; ready: boolean; sender?: string; host?: string; port?: number; missing: string[]; issues: string[]; alertIssues?: string[]; alertRecipients: string[]; message?: string }
interface DeliveryResult extends Partial<ToolRecord> { status: string; message: string }
const status = shallowRef<MailStatus | null>(null)
const error = shallowRef(''), result = shallowRef(''), busy = shallowRef(false)
const retried = shallowRef<number[]>([])
const replacements = shallowRef<ToolRecord[]>([])
const clock = shallowRef(Date.now())
const receipts = computed(() => [...replacements.value, ...props.alerts.filter(record => !replacements.value.some(item => item.id === record.id))].filter(record => !retried.value.includes(record.id)))
const failed = computed(() => receipts.value.filter(record => record.status === 'FAILED' && record.channel === 'email'))
const interrupted = computed(() => receipts.value.filter(record => record.status === 'RUNNING' && record.channel === 'email'))
function expired(record: ToolRecord) {
  try { const expiry = Date.parse(JSON.parse(record.message).mailRetryLease); return Number.isFinite(expiry) && expiry <= clock.value }
  catch { return false }
}
function showResult(response: DeliveryResult) {
  if (response.status === 'FAILED') error.value = response.message
  else result.value = response.message
}
async function load() {
  error.value = ''; clock.value = Date.now()
  try { status.value = await requestJson<MailStatus>('/api/admin/mail/status') }
  catch (reason) { error.value = reason instanceof Error ? reason.message : '邮件配置读取失败' }
}
async function testMail() {
  busy.value = true; error.value = ''; result.value = ''
  try {
    const response = await requestJson<DeliveryResult>('/api/admin/mail/test', { method: 'POST' })
    showResult(response)
  } catch (reason) { error.value = reason instanceof Error ? reason.message : '测试发送失败' }
  finally { busy.value = false }
}
async function retry(id: number, recoverInterrupted = false) {
  busy.value = true; error.value = ''; result.value = ''
  try {
    const options: RequestInit = recoverInterrupted
      ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ recoverInterrupted: true }) }
      : { method: 'POST' }
    const response = await requestJson<DeliveryResult>(`/api/admin/mail/alerts/${id}/retry`, options)
    showResult(response)
    if (response.id != null && response.reportId != null) replacements.value = [...replacements.value, response as ToolRecord]
    retried.value = [...retried.value, id]
    emit('refresh', response)
  } catch (reason) { error.value = reason instanceof Error ? reason.message : '预警重试失败' }
  finally { busy.value = false }
}
onMounted(load)
</script>

<template>
  <section class="mail-panel" aria-label="邮件配置与投递">
    <header><h3>邮件配置与投递</h3><button class="secondary-button" :disabled="busy" @click="load">刷新配置</button></header>
    <template v-if="status">
      <p>{{ status.mode === 'log' ? '仅记录，未发送邮件' : status.ready ? '邮件配置检查通过' : '邮件配置待完善' }}</p>
      <p class="mail-details">发件人：{{ status.sender || '未配置' }} · 邮件服务器：{{ status.host || '未配置' }}{{ status.port ? `:${status.port}` : '' }}</p>
      <p class="mail-details">预警及测试接收人：{{ status.alertRecipients.join('、') || '未配置' }}</p>
      <p v-if="status.missing.length">缺少配置：{{ status.missing.join('、') }}</p>
      <p v-for="issue in [...status.issues, ...(status.alertIssues ?? [])]" :key="issue">{{ issue }}</p>
      <p class="mail-details">{{ status.message }}。测试邮件仅发往上述配置的预警接收人。</p>
      <button data-test="mail-test" class="primary-button" :disabled="busy || !status.alertRecipients.length" @click="testMail">{{ busy ? '正在处理…' : '发送测试邮件' }}</button>
    </template>
    <p v-if="error" role="alert">{{ error }}</p>
    <p v-if="result" role="status">{{ result }}</p>
    <div v-if="interrupted.length" class="mail-retries"><h4>投递中或结果不确定</h4><p class="mail-details">进程中断时，邮件可能已被 SMTP 接受。租约到期后请先核查收件箱，再确认恢复；重新发送可能重复。点击刷新配置检查租约状态。</p><div v-for="record in interrupted" :key="record.id"><span>预警 #{{ record.id }} · {{ expired(record) ? '租约已到期' : '租约未到期，请稍后刷新' }}</span><button v-if="expired(record)" data-test="mail-recover" class="secondary-button" :disabled="busy" @click="retry(record.id, true)">确认可能重复，恢复中断投递</button></div></div>
    <div v-if="failed.length" class="mail-retries"><h4>失败的预警</h4><div v-for="record in failed" :key="record.id"><span>预警 #{{ record.id }} · 报告 #{{ record.reportId }}</span><button data-test="mail-retry" class="secondary-button" :disabled="busy" @click="retry(record.id)">重试此预警</button></div></div>
  </section>
</template>

<style scoped>
.mail-panel { margin:20px 0; padding:18px; border:1px solid var(--line); border-radius:12px; }
.mail-panel header, .mail-retries > div { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:12px; }
.mail-panel h3, .mail-panel h4 { margin:0; }.mail-panel p { overflow-wrap:anywhere; }.mail-details { color:var(--text-soft); font-size:13px; line-height:1.6; }.mail-retries { margin-top:20px; display:grid; gap:10px; }[role=alert] { color:var(--danger); }
</style>
