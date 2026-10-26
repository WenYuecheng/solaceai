<script setup lang="ts">
import { computed, defineAsyncComponent, shallowRef } from 'vue'
import { request, requestJson } from '@/core/api'
import MailDeliveryPanel from './MailDeliveryPanel.vue'
import { ElMessage } from 'element-plus'
const RunAuditPanel = defineAsyncComponent(() => import('./RunAuditPanel.vue'))
import { formatDateTime, statusLabel } from '@/core/format'
import type { ToolRecord } from '@/types/api'

const props = defineProps<{ excel: ToolRecord[]; alerts: ToolRecord[] }>()
const tab = shallowRef<'excel' | 'alerts' | 'traces'>('excel')
const refreshedAlerts = shallowRef<ToolRecord[] | null>(null)
const records = computed(() => tab.value === 'excel' ? props.excel : (refreshedAlerts.value ?? props.alerts))
async function refreshAlerts(record?: Partial<ToolRecord>) {
  if (record?.id != null && record.reportId != null) refreshedAlerts.value = [record as ToolRecord, ...(refreshedAlerts.value ?? props.alerts)]
  try { refreshedAlerts.value = await requestJson<ToolRecord[]>('/api/admin/alerts') }
  catch { ElMessage.error('预警记录刷新失败，请重新打开此页面') }
}
function deliveryLabel(status: string, channel?: string, message?: string) {
  if (status === 'LOG_ONLY' || channel === 'log' || message?.includes('deliveryMode=log')) return '仅记录'
  if (status === 'SUCCESS' && channel === 'email') return 'SMTP 已接受'
  if (status === 'RETRIED') return '已重试'
  if (status === 'RUNNING') return '处理中'
  return statusLabel(status)
}
const start = shallowRef(''), end = shallowRef(''), risk = shallowRef(''), exporting = shallowRef(false)
async function download() {
  exporting.value = true
  try {
    const query = new URLSearchParams(); if (start.value) query.set('start', start.value); if (end.value) query.set('end', end.value); if (risk.value) query.set('risk', risk.value)
    const response = await request(`/api/admin/reports/export?${query}`)
    const url = URL.createObjectURL(await response.blob()); const link = document.createElement('a'); link.href = url; link.download = 'SolaceAI报告台账.xlsx'; link.click(); URL.revokeObjectURL(url)
  } catch (error) { ElMessage.error(error instanceof Error ? error.message : '下载失败') }
  finally { exporting.value = false }
}
</script>

<template>
  <section class="data-view glass"><header><div><h2>工具执行与告警投递</h2><p>导出报告台账，查看工具执行和风险预警记录。</p></div><div v-if="tab !== 'traces'" class="delivery-summary"><span><i class="success"></i>{{ records.filter(item => item.status === 'SUCCESS').length }} 成功</span><span><i class="failed"></i>{{ records.filter(item => item.status === 'FAILED').length }} 失败</span></div></header>
    <el-segmented v-model="tab" :options="[{ label: `历史台账记录 ${excel.length}`, value: 'excel' }, { label: `风险预警 ${alerts.length}`, value: 'alerts' }, { label: '运行追踪', value: 'traces' }]" class="delivery-tabs"/>
    <div class="export-controls"><label>开始日期<input v-model="start" type="date"></label><label>结束日期<input v-model="end" type="date"></label><label>风险等级<select v-model="risk"><option value="">全部</option><option value="LOW">低风险</option><option value="MEDIUM">中风险</option><option value="HIGH">高风险</option></select></label><button class="primary-button" :disabled="exporting" @click="download">{{ exporting ? '正在生成…' : '下载报告台账' }}</button></div>
    <MailDeliveryPanel v-if="tab === 'alerts'" :alerts="refreshedAlerts ?? alerts" @refresh="refreshAlerts" />
    <RunAuditPanel v-if="tab === 'traces'" />
    <el-table v-else :data="records" empty-text="暂无执行记录" row-key="id">
      <el-table-column prop="id" label="记录" width="90"><template #default="scope"><span class="mono">#{{ scope.row.id }}</span></template></el-table-column>
      <el-table-column prop="reportId" label="报告" width="100"><template #default="scope"><span class="mono">#{{ scope.row.reportId }}</span></template></el-table-column>
      <el-table-column label="状态" width="140"><template #default="scope"><el-tag :type="scope.row.status === 'SUCCESS' ? 'success' : 'danger'">{{ tab === 'alerts' ? deliveryLabel(scope.row.status, scope.row.channel, scope.row.message) : statusLabel(scope.row.status) }}</el-tag></template></el-table-column>
      <el-table-column v-if="tab === 'alerts'" prop="channel" label="渠道" width="110"/>
      <el-table-column v-if="tab === 'alerts'" prop="recipient" label="接收人" min-width="180" show-overflow-tooltip/>
      <el-table-column v-else prop="filePath" label="文件" min-width="220" show-overflow-tooltip/>
      <el-table-column prop="message" label="执行信息" min-width="260" show-overflow-tooltip/>
      <el-table-column label="时间" width="180"><template #default="scope">{{ formatDateTime(scope.row.createdAt) }}</template></el-table-column>
    </el-table>
  </section>
</template>

<style scoped>
.export-controls { display:flex; gap:12px; flex-wrap:wrap; margin:16px 0; align-items:end; } .export-controls label { display:grid; gap:6px; font-size:12px; } .export-controls input, .export-controls select { padding:8px; border:1px solid var(--line); border-radius:8px; background:var(--surface-soft); color:var(--text); }
.data-view { min-width: 0; padding: 22px; border-radius: var(--radius-lg); }.data-view > header { display: flex; align-items: flex-end; justify-content: space-between; gap: 20px; }.data-view h2 { margin: 0; font-family: var(--font-display); font-size: 30px; letter-spacing: -.045em; }.data-view header p:last-child { margin: 8px 0 0; color: var(--text-soft); }.delivery-summary { display: flex; gap: 14px; color: var(--text-faint); font-size: 11px; }.delivery-summary span { display: inline-flex; align-items: center; gap: 7px; }.delivery-summary i { width: 7px; height: 7px; border-radius: 50%; }.delivery-summary .success { background: var(--success); }.delivery-summary .failed { background: var(--danger); }.delivery-tabs { margin: 24px 0 16px; }
@media (max-width: 620px) { .data-view { padding: 16px; }.delivery-summary { display: none; } }
</style>
