<script setup lang="ts">
import { computed, onMounted, onUnmounted, shallowRef } from 'vue'
import type { AnswerTrace, AnswerTraceStatus, AnswerTraceStep } from '@/types/api'
import RetrievalDetails from './RetrievalDetails.vue'
import { actorForStep, phaseForStep, tracePhases } from '@/core/answer-trace'

const props = defineProps<{ trace?: AnswerTrace | null; active?: boolean }>()
const labels: Record<AnswerTraceStatus, string> = { running: '进行中', completed: '已完成', skipped: '已跳过', failed: '未完成', degraded: '已降级', cancelled: '已停止' }
const now = shallowRef(Date.now())
let timer: ReturnType<typeof setInterval> | undefined
onMounted(() => { timer = setInterval(() => { if (props.trace?.status === 'running') now.value = Date.now() }, 250) })
onUnmounted(() => clearInterval(timer))
const steps = computed(() => props.trace?.steps.map(step => ({ ...step, phase: phaseForStep(step), actor: actorForStep(step) })) ?? [])
const rounds = computed(() => Math.max(0, ...steps.value.map(step => step.round ?? 0)))
const cycle = ['reason', 'act', 'observe', 'answer'] as const
const seenPhases = computed(() => new Set(steps.value.map(step => step.phase)))
const total = computed(() => props.trace?.steps.filter(step => step.status !== 'running').length ?? 0)
function duration(step: AnswerTraceStep) {
  const milliseconds = step.durationMs ?? Math.max(0, now.value - Date.parse(step.startedAt))
  if (!Number.isFinite(milliseconds)) return '计时中'
  return milliseconds < 1000 ? `${Math.round(milliseconds)} ms` : `${(milliseconds / 1000).toFixed(1)} s`
}
</script>

<template>
  <details v-if="trace" class="answer-trace" open>
    <summary class="trace-summary"><span>回答执行过程</span><span class="trace-progress">{{ labels[trace.status] }}<template v-if="rounds"> · {{ rounds }} 轮</template> · {{ total }}/{{ trace.steps.length }} 步</span></summary>
    <div class="trace-body" tabindex="0" role="region" aria-label="执行步骤记录">
    <div class="react-cycle" aria-label="ReAct 阶段说明">
      <span v-for="phase in cycle" :key="phase" class="cycle-phase" :class="{ visited: seenPhases.has(phase) }"><b>{{ tracePhases[phase].label }}</b><span>{{ tracePhases[phase].description }}</span></span>
    </div>
    <p class="trace-note">按实际执行顺序展示决策摘要、动作与结果；同一轮中的工具结果供后续决策使用。</p>
    <ol class="trace-steps">
      <li v-for="step in steps" :key="step.id" class="trace-step" :data-status="step.status" :data-phase="step.phase">
        <div class="step-heading"><span class="step-index">{{ step.sequence }}</span><span class="phase-badge">{{ tracePhases[step.phase].label }}</span><strong>{{ step.title }}</strong><span class="step-status">{{ labels[step.status] }}</span><time class="step-duration">{{ duration(step) }}</time></div>
        <div class="step-meta"><span v-if="step.round">第 {{ step.round }} 轮</span><span>执行者：{{ step.actor }}</span><code v-if="step.tool">工具：{{ step.tool }}</code><code v-if="step.nextAction">下一步：{{ step.nextAction }}</code></div>
        <p class="step-why">{{ step.why }}</p>
        <p v-if="step.result" class="step-result">{{ step.result }}<span v-if="step.count !== undefined"> · 数量：{{ step.count }}</span></p>
        <RetrievalDetails v-if="step.retrievalDetails?.length" :groups="step.retrievalDetails" />
      </li>
    </ol>
    </div>
  </details>
  <p v-else class="trace-unavailable">{{ active ? '正在准备本次回答的执行记录…' : '此历史回复未记录执行过程' }}</p>
</template>

<style scoped>
.answer-trace { margin: 0 0 22px; border: 1px solid #b7c8ad; border-radius: 12px; background: transparent; font-size: 12px; color: #43583f; }
.trace-summary { cursor: pointer; padding: 12px 14px; font-size: 12px; font-weight: 600; color: #43583f; }.trace-progress { float: right; font-family: var(--font-mono); font-size: 10px; font-weight: 400; margin-left: 10px; }.trace-body { max-height: 240px; overflow-y: auto; padding: 0 14px 14px; border-top: 1px solid #c5d2bb; overscroll-behavior: contain; }.trace-note, .trace-unavailable { color: #43583f; font-size: 10px; line-height: 1.7; }.trace-unavailable { margin: 0 0 14px; }
.react-cycle { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 5px; margin-top: 12px; }.cycle-phase { display: grid; gap: 3px; padding: 6px; border: 1px solid #b7c8ad; border-radius: 6px; color: #43583f; opacity: 1; }.cycle-phase b { font-family: var(--font-mono); font-size: 10px; }.cycle-phase > span { font-size: 10px; line-height: 1.5; }.cycle-phase.visited { opacity: 1; color: #43583f; background: transparent; }
.phase-badge { padding: 3px 6px; border-radius: 4px; background: transparent; color: #43583f; font-family: var(--font-mono); font-size: 10px; }.trace-step[data-phase="reason"] .phase-badge { background: transparent; color: #43583f; }.trace-step[data-phase="act"] .phase-badge { background: transparent; color: #43583f; }.trace-step[data-phase="observe"] .phase-badge { background: transparent; color: #43583f; }.trace-step[data-phase="answer"] .phase-badge { background: transparent; color: #43583f; }
.step-meta { display: flex; flex-wrap: wrap; gap: 4px 10px; margin: 6px 0 0 24px; color: #43583f; font: 11px/1.6 var(--font-mono); }.step-meta code { overflow-wrap: anywhere; font: inherit; }.trace-steps { list-style: none; padding: 0; margin: 12px 0 0; }.trace-step { border-left: 2px solid #b2c3a7; padding: 0 0 14px 10px; margin-left: 7px; }.trace-step:last-child { padding-bottom: 0; }.step-heading { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; color: #43583f; }.step-heading strong { font-size: 12px; font-weight: 500; }.step-index { display: inline-grid; place-content: center; min-width: 18px; height: 18px; border-radius: 50%; background: transparent; border: 1px solid #b7c8ad; font: 10px var(--font-mono); }.step-status { margin-left: auto; color: #43583f; font-size: 10px; }.step-duration { color: #43583f; font: 10px var(--font-mono); }.step-why, .step-result { line-height: 1.7; margin: 5px 0 0 24px; color: #43583f; font-size: 12px; }.step-why { color: #43583f; }.trace-step[data-status="running"] { border-left-color: #77926a; }.trace-step[data-status="failed"], .trace-step[data-status="degraded"] { border-left-color: #b88c88; }
@media (max-width: 520px) { .react-cycle { grid-template-columns: repeat(2, minmax(0, 1fr)); }.trace-progress { float: none; display: block; margin: 5px 0 0 15px; }.trace-body { padding-inline: 10px; } }
</style>
