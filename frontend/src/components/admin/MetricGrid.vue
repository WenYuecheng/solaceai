<script setup lang="ts">
import { percentage } from '@/core/format'

const props = defineProps<{ metrics: { reports: number; highRisk: number; cases: number; openCases: number; excel: number; excelSuccess: number; alerts: number; alertSuccess: number } }>()
const cards = [
  { key: 'reports', label: '咨询报告', tone: 'blue', meta: () => '累计报告' },
  { key: 'highRisk', label: '高风险报告', tone: 'red', meta: () => `占比 ${percentage(props.metrics.highRisk, props.metrics.reports)}` },
  { key: 'cases', label: '风险个案', tone: 'amber', meta: () => `待跟进 ${props.metrics.openCases}` },
  { key: 'excel', label: '台账写入', tone: 'teal', meta: () => `成功 ${props.metrics.excelSuccess}` },
  { key: 'alerts', label: '风险预警', tone: 'violet', meta: () => `发送成功 ${props.metrics.alertSuccess}` },
] as const
</script>

<template>
  <section class="metric-grid" aria-label="核心运营指标">
    <article v-for="card in cards" :key="card.key" class="metric-card glass" :class="`tone-${card.tone}`">
      <div><span>{{ card.label }}</span></div><strong class="mono">{{ metrics[card.key] }}</strong><small>{{ card.meta() }}</small>
    </article>
  </section>
</template>

<style scoped>
.metric-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 0; }.metric-card { min-width: 0; min-height: 114px; padding: 18px 20px; border: 0; border-right: 1px solid var(--line); border-radius: 0; } .metric-grid { border: 1px solid var(--line); border-radius: 20px; overflow: hidden; background: var(--surface); }.metric-card:last-child { border-right: 0; }.metric-card > div { display: flex; justify-content: space-between; gap: 10px; color: var(--text-soft); font-size: 12px; font-weight: 700; }.metric-card strong { display: block; margin-top: 12px; font-family: var(--font-display); font-size: 30px; color: var(--tone); line-height: 1; letter-spacing: -.05em; }.metric-card small { display: block; margin-top: 9px; color: var(--text-faint); font-size: 11px; }.tone-blue { --tone: var(--lagoon); }.tone-red { --tone: var(--danger); }.tone-amber { --tone: var(--amber); }.tone-teal { --tone: var(--success); }.tone-violet { --tone: var(--lagoon); }
@media (max-width: 1200px) { .metric-grid { grid-template-columns: repeat(3, 1fr); } }@media (max-width: 650px) { .metric-grid { grid-template-columns: repeat(2, 1fr); }.metric-card:last-child { grid-column: 1 / -1; } }
</style>
