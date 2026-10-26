<script setup lang="ts">
import { onBeforeUnmount, onMounted, shallowRef, useTemplateRef, watch } from 'vue'
import * as echarts from 'echarts/core'
import { BarChart, LineChart, PieChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { buildDistributionData, buildSevenDayTrend } from '@/core/chart-options'
import type { Report, ToolRecord } from '@/types/api'

echarts.use([BarChart, LineChart, PieChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer])
const props = defineProps<{ reports: Report[]; excel: ToolRecord[]; alerts: ToolRecord[] }>()
const trendEl = useTemplateRef<HTMLDivElement>('trend')
const riskEl = useTemplateRef<HTMLDivElement>('risk')
const emotionEl = useTemplateRef<HTMLDivElement>('emotion')
const deliveryEl = useTemplateRef<HTMLDivElement>('delivery')
const instances = shallowRef<echarts.ECharts[]>([])
let observer: ResizeObserver | null = null

function palette() {
  const styles = getComputedStyle(document.documentElement)
  return { text: styles.getPropertyValue('--text').trim(), soft: styles.getPropertyValue('--text-soft').trim(), line: styles.getPropertyValue('--line').trim(), lagoon: styles.getPropertyValue('--lagoon').trim(), aqua: styles.getPropertyValue('--aqua').trim(), violet: styles.getPropertyValue('--violet').trim(), danger: styles.getPropertyValue('--danger').trim(), amber: styles.getPropertyValue('--amber').trim() }
}

function render() {
  if (!trendEl.value || !riskEl.value || !emotionEl.value || !deliveryEl.value) return
  if (!instances.value.length) instances.value = [trendEl.value, riskEl.value, emotionEl.value, deliveryEl.value].map((element) => echarts.init(element))
  const [trend, risk, emotion, delivery] = instances.value
  const colors = palette()
  const axis = { axisLine: { lineStyle: { color: colors.line } }, axisLabel: { color: colors.soft }, splitLine: { lineStyle: { color: colors.line } } }
  const seven = buildSevenDayTrend(props.reports)
  const animate = !window.matchMedia('(prefers-reduced-motion: reduce)').matches
  trend.setOption({ animation: animate, animationDuration: 300, color: [colors.aqua], tooltip: { trigger: 'axis' }, grid: { top: 18, left: 40, right: 16, bottom: 32 }, xAxis: { type: 'category', data: seven.map((item) => item.label), ...axis }, yAxis: { type: 'value', minInterval: 1, ...axis }, series: [{ type: 'line', data: seven.map((item) => item.value), smooth: .42, symbolSize: 8, areaStyle: { opacity: .12 }, lineStyle: { width: 3 } }] })
  const makeDonut = (data: Array<{ name: string; value: number }>, colorsList: string[]) => ({ animation: animate, animationDuration: 300, color: colorsList, tooltip: { trigger: 'item' }, legend: { bottom: 0, textStyle: { color: colors.soft } }, series: [{ type: 'pie', radius: ['48%', '70%'], center: ['50%', '43%'], label: { color: colors.text, formatter: '{b}\n{c}' }, itemStyle: { borderRadius: 6, borderWidth: 2, borderColor: 'transparent' }, data }] })
  risk.setOption(makeDonut(buildDistributionData(props.reports, 'riskLevel', [['LOW', '低风险'], ['MEDIUM', '中风险'], ['HIGH', '高风险']]), [colors.lagoon, colors.amber, colors.danger]))
  emotion.setOption(makeDonut(buildDistributionData(props.reports, 'emotion', [['NORMAL', '平稳'], ['ANXIETY', '焦虑'], ['DEPRESSED', '低落'], ['HIGH_RISK', '高风险']]), [colors.lagoon, colors.amber, colors.violet, colors.danger]))
  const deliveryData = [props.excel.filter((item) => item.status === 'SUCCESS').length, props.excel.filter((item) => item.status === 'FAILED').length, props.alerts.filter((item) => item.status === 'SUCCESS').length, props.alerts.filter((item) => item.status === 'FAILED').length]
  delivery.setOption({ animation: animate, animationDuration: 300, color: [colors.violet], tooltip: { trigger: 'axis' }, grid: { top: 15, left: 76, right: 18, bottom: 20 }, xAxis: { type: 'value', minInterval: 1, ...axis }, yAxis: { type: 'category', data: ['台账成功', '台账失败', '预警成功', '预警失败'], ...axis }, series: [{ type: 'bar', data: deliveryData, barWidth: 13, itemStyle: { borderRadius: [0, 8, 8, 0] } }] })
}

onMounted(() => { render(); observer = new ResizeObserver(() => instances.value.forEach((chart) => chart.resize())); [trendEl.value, riskEl.value, emotionEl.value, deliveryEl.value].forEach((element) => element && observer?.observe(element)) })
watch(() => [props.reports, props.excel, props.alerts], render, { deep: true })
onBeforeUnmount(() => { observer?.disconnect(); instances.value.forEach((chart) => chart.dispose()) })
</script>

<template>
  <section class="analytics-grid" aria-label="咨询与风险数据图表">
    <article class="chart-card glass chart-wide"><div><h3>近 7 日报告趋势</h3></div><div ref="trend" class="chart" role="img" :aria-label="`近 7 日共 ${reports.length} 条报告`"></div></article>
    <article class="chart-card glass"><div><h3>风险等级</h3></div><div ref="risk" class="chart" role="img" aria-label="风险等级分布图"></div></article>
    <article class="chart-card glass"><div><h3>情绪分布</h3></div><div ref="emotion" class="chart" role="img" aria-label="情绪分布图"></div></article>
    <article class="chart-card glass chart-wide"><div><h3>工具执行状态</h3></div><div ref="delivery" class="chart" role="img" aria-label="工具执行状态图"></div></article>
  </section>
</template>

<style scoped>
.analytics-grid { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 16px; }.chart-card { min-width: 0; padding: 20px; border-radius: var(--radius-md); }.chart-card h3 { margin: 0; font-family: var(--font-display); font-size: 16px; }.chart { width: 100%; height: 208px; margin-top: 8px; }.chart-wide { grid-column: span 1; }
@media (max-width: 700px) { .analytics-grid { grid-template-columns: 1fr; }.chart { height: 230px; } }
</style>
