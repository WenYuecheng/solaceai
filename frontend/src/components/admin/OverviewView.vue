<script setup lang="ts">
import { defineAsyncComponent } from 'vue'
import MetricGrid from '@/components/admin/MetricGrid.vue'
import { formatDateTime, statusLabel } from '@/core/format'
import type { Report, RiskCase, ToolRecord } from '@/types/api'

interface DashboardMetrics { reports: number; highRisk: number; cases: number; openCases: number; excel: number; excelSuccess: number; alerts: number; alertSuccess: number }
defineProps<{ displayName?: string; metrics: DashboardMetrics; reports: Report[]; cases: RiskCase[]; excel: ToolRecord[]; alerts: ToolRecord[] }>()
const emit = defineEmits<{ openReport: [report: Report] }>()
const AnalyticsCharts = defineAsyncComponent(() => import('@/components/admin/AnalyticsCharts.vue'))
</script>

<template>
  <div class="overview-view">
    <section class="garden-welcome" aria-label="教师工作台欢迎信息">
      <div class="garden-welcome-copy"><p class="welcome-date">校园关怀，从倾听开始</p><h2>{{ displayName || '老师' }}，你好</h2><p>在这里看见学生的近况，<br />让需要的支持，及时发生。</p></div>
    </section>
    <MetricGrid :metrics="metrics" />
    <section class="section-heading"><div><h2>咨询与风险数据洞察</h2></div><span>近 7 日及累计分布</span></section>
    <Suspense><AnalyticsCharts :reports="reports" :excel="excel" :alerts="alerts" /><template #fallback><div class="chart-skeleton glass">正在读取图表…</div></template></Suspense>
    <section class="overview-lists">
      <article class="summary-panel glass"><div class="panel-head"><div><h3>最近咨询报告</h3></div><span>{{ reports.length }} 条</span></div><div v-if="!reports.length" class="empty-state"><strong>暂无报告</strong><p>完成咨询后，可在这里查看生成的报告。</p></div><button v-for="report in reports.slice(0, 5)" :key="report.id" type="button" class="summary-row" @click="emit('openReport', report)"><span><strong>{{ report.displayName || report.username }}</strong><small>{{ report.summary }}</small></span><span><b :class="`risk-${report.riskLevel.toLowerCase()}`">{{ statusLabel(report.riskLevel) }}</b><small>{{ formatDateTime(report.createdAt) }}</small></span></button></article>
      <article class="summary-panel glass"><div class="panel-head"><div><h3>待跟进个案</h3></div><span>{{ cases.filter(item => item.status === 'OPEN').length }} 个</span></div><div v-if="!cases.length" class="empty-state"><strong>暂无个案</strong><p>中高风险报告会自动创建风险个案。</p></div><div v-for="item in cases.slice(0, 5)" :key="item.id" class="case-row"><span class="case-index mono">#{{ item.id }}</span><span><strong>{{ item.summary || '待补充摘要' }}</strong><small>负责人：{{ item.owner || '未分配' }}</small></span><b>{{ statusLabel(item.status) }}</b></div></article>
    </section>
  </div>
</template>

<style scoped>
.garden-welcome { position: relative; min-height: 208px; overflow: hidden; border-radius: 24px 24px 64px 24px; background: #e1ecd8; }.garden-welcome > img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; object-position: center 54%; }.garden-welcome-copy { position: relative; width: max-content; max-width: 64%; margin: 20px; padding: 20px 28px; background: rgba(255, 253, 246, .96); border-radius: 16px 32px 32px 16px; }.garden-welcome-copy .welcome-date { margin: 0 0 9px; color: var(--lagoon); font-size: 12px; }.garden-welcome-copy h2 { overflow-wrap: anywhere; margin: 0; font-size: clamp(23px, 2.4vw, 30px); font-weight: 600; letter-spacing: -.03em; }.garden-welcome-copy > p:last-child { margin: 10px 0 0; color: var(--text-soft); font-size: 13px; line-height: 1.8; }
.overview-view { display: grid; gap: 22px; }.section-heading { display: flex; align-items: flex-end; justify-content: space-between; gap: 20px; margin-top: 4px; }.section-heading h2 { margin: 0; font-family: var(--font-display); font-size: 20px; letter-spacing: -.04em; }.section-heading > span { color: var(--text-faint); font-size: 11px; }.chart-skeleton { min-height: 250px; display: grid; place-items: center; border-radius: var(--radius-md); color: var(--text-faint); }.overview-lists { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }.summary-panel { padding: 18px; border-radius: var(--radius-md); }.panel-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 15px; padding-bottom: 14px; border-bottom: 1px solid var(--line); }.panel-head h3 { margin: 0; font-family: var(--font-display); }.panel-head > span { color: var(--text-faint); font-size: 11px; }.summary-row { width: 100%; min-height: 68px; display: flex; align-items: center; justify-content: space-between; gap: 14px; padding: 11px 4px; border: 0; border-bottom: 1px solid var(--line); color: var(--text); background: transparent; text-align: left; }.summary-row:hover { background: var(--surface-soft); }.summary-row > span { min-width: 0; display: grid; gap: 4px; }.summary-row > span:last-child { justify-items: end; }.summary-row strong, .summary-row small { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }.summary-row small, .case-row small { color: var(--text-faint); font-size: 11px; }.summary-row b { padding: 4px 8px; border-radius: var(--radius-pill); color: var(--lagoon); background: var(--surface-soft); font-size: 11px; }.summary-row b.risk-high { color: var(--danger); }.summary-row b.risk-medium { color: var(--amber); }.case-row { min-height: 68px; display: grid; grid-template-columns: auto minmax(0, 1fr) auto; align-items: center; gap: 12px; border-bottom: 1px solid var(--line); }.case-row > span:nth-child(2) { min-width: 0; display: grid; gap: 4px; }.case-row strong { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }.case-row > b { color: var(--amber); font-size: 11px; }.case-index { color: var(--lagoon); }.empty-state { min-height: 170px; display: grid; place-content: center; text-align: center; color: var(--text-soft); }.empty-state p { margin: 8px 0 0; color: var(--text-faint); }
@media (max-width: 760px) { .garden-welcome { min-height: 205px; border-bottom-right-radius: 40px; }.garden-welcome-copy { margin: 14px; padding: 18px; max-width: 82%; }.garden-welcome-copy h2 { overflow-wrap: anywhere; } .overview-lists { grid-template-columns: 1fr; } }
.garden-welcome { min-height: 190px; background: transparent; border: 1px solid #cbd7bf; }.garden-welcome-copy { background: #fffef9dc; }.garden-welcome-copy h2 { font-family: var(--font-playful); font-weight: 400; }
</style>
