<script setup lang="ts">
import { computed, shallowRef } from 'vue'
import { filterReports } from '@/core/admin-view'
import { formatDateTime, statusLabel } from '@/core/format'
import type { Conversation, Report } from '@/types/api'

const props = defineProps<{ reports: Report[]; conversation: Conversation | null; detailLoading: boolean }>()
const emit = defineEmits<{ open: [report: Report]; close: [] }>()
const query = shallowRef(''), risk = shallowRef('ALL'), intent = shallowRef('ALL'), page = shallowRef(1), drawerOpen = shallowRef(false)
const pageSize = 10
const filtered = computed(() => filterReports(props.reports, query.value, risk.value, intent.value))
const visible = computed(() => filtered.value.slice((page.value - 1) * pageSize, page.value * pageSize))
function open(report: Report) { drawerOpen.value = true; emit('open', report) }
</script>

<template>
  <section class="data-view glass"><header><div><h2>报告档案</h2><p>按学生、风险和意图筛选，并在侧边审阅完整历史消息。</p></div><span class="count-chip mono">{{ filtered.length }} 条报告</span></header>
    <div class="filters"><el-input v-model="query" clearable placeholder="搜索学生或摘要…" aria-label="搜索咨询报告"/><el-select v-model="risk" aria-label="筛选风险等级"><el-option label="全部风险" value="ALL"/><el-option label="低风险" value="LOW"/><el-option label="中风险" value="MEDIUM"/><el-option label="高风险" value="HIGH"/></el-select><el-select v-model="intent" aria-label="筛选意图"><el-option label="全部意图" value="ALL"/><el-option label="日常交流" value="CHAT"/><el-option label="心理咨询" value="CONSULT"/><el-option label="风险干预" value="RISK"/></el-select></div>
    <el-table :data="visible" empty-text="暂无符合条件的报告" row-key="id" @row-click="open">
      <el-table-column prop="displayName" label="学生" width="140"><template #default="scope">{{ scope.row.displayName || scope.row.username }}</template></el-table-column>
      <el-table-column label="风险" width="100"><template #default="scope"><el-tag :type="scope.row.riskLevel === 'HIGH' ? 'danger' : scope.row.riskLevel === 'MEDIUM' ? 'warning' : 'success'">{{ statusLabel(scope.row.riskLevel) }}</el-tag></template></el-table-column>
      <el-table-column label="意图" width="110"><template #default="scope">{{ statusLabel(scope.row.intent) }}</template></el-table-column>
      <el-table-column prop="summary" label="摘要" min-width="260" show-overflow-tooltip/>
      <el-table-column label="时间" width="180"><template #default="scope">{{ formatDateTime(scope.row.createdAt) }}</template></el-table-column>
    </el-table>
    <el-pagination v-if="filtered.length > pageSize" v-model:current-page="page" background layout="prev, pager, next" :page-size="pageSize" :total="filtered.length"/>
    <el-drawer v-model="drawerOpen" direction="rtl" size="min(620px, 96vw)" title="只读会话档案" destroy-on-close @closed="emit('close')">
      <div v-if="detailLoading" class="drawer-loading">正在读取会话…</div>
      <template v-else-if="conversation"><div class="archive-heading"><div><strong>{{ conversation.title || conversation.sessionId }}</strong><small>{{ conversation.messages.length }} 条消息</small></div><span class="mono">{{ conversation.sessionId }}</span></div><div class="conversation-list"><article v-for="(message, index) in conversation.messages" :key="`${message.createdAt}-${index}`" :class="['archive-message', message.role.toLowerCase()]"><div><strong>{{ message.role.toUpperCase() === 'USER' ? '学生' : 'SolaceAI' }}</strong><time>{{ formatDateTime(message.createdAt) }}</time></div><p>{{ message.content }}</p></article></div></template>
      <el-empty v-else description="未找到会话内容"/>
    </el-drawer>
  </section>
</template>

<style scoped>
.data-view { min-width: 0; padding: 22px; border-radius: var(--radius-lg); }.data-view > header { display: flex; align-items: flex-end; justify-content: space-between; gap: 20px; }.data-view h2 { margin: 0; font-family: var(--font-display); font-size: 30px; letter-spacing: -.045em; }.data-view header p:last-child { margin: 8px 0 0; color: var(--text-soft); }.count-chip { padding: 8px 11px; border: 1px solid var(--line); border-radius: var(--radius-pill); color: var(--text-faint); font-size: 11px; }.filters { display: grid; grid-template-columns: minmax(220px, 1fr) 170px 170px; gap: 10px; margin: 24px 0 16px; }.el-pagination { justify-content: flex-end; margin-top: 18px; }.drawer-loading { min-height: 300px; display: grid; place-items: center; color: var(--text-faint); }.archive-heading { display: flex; justify-content: space-between; gap: 15px; padding-bottom: 16px; border-bottom: 1px solid var(--line); }.archive-heading div { display: grid; gap: 5px; }.archive-heading small, .archive-heading > span { color: var(--text-faint); font-size: 11px; }.conversation-list { display: grid; gap: 16px; margin-top: 20px; }.archive-message { max-width: 90%; }.archive-message.user { margin-left: auto; }.archive-message > div { display: flex; justify-content: space-between; gap: 10px; margin: 0 5px 5px; color: var(--text-faint); font-size: 11px; }.archive-message p { margin: 0; padding: 13px 15px; border: 1px solid var(--line); border-radius: 5px 17px 17px 17px; background: var(--surface-soft); line-height: 1.7; white-space: pre-wrap; overflow-wrap: anywhere; }.archive-message.user p { border-radius: 17px 5px 17px 17px; color: white; background: var(--lagoon-deep); }
@media (max-width: 720px) { .data-view { padding: 16px; }.filters { grid-template-columns: 1fr; }.count-chip { display: none; } }
</style>
