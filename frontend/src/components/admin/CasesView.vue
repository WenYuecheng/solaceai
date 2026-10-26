<script setup lang="ts">
import { computed, shallowRef, watch } from 'vue'
import CaseFollowup from './CaseFollowup.vue'
import { formatDateTime, statusLabel } from '@/core/format'
import type { RiskCase } from '@/types/api'

const props = defineProps<{ cases: RiskCase[] }>()
const emit = defineEmits<{ updated: [item: RiskCase] }>()
function updateCase(item: RiskCase) { selected.value = item; emit('updated', item) }
watch(() => props.cases, (cases) => { if (selected.value) selected.value = cases.find(item => item.id === selected.value?.id) ?? selected.value })
const query = shallowRef(''), risk = shallowRef('ALL'), status = shallowRef('ALL'), page = shallowRef(1)
const selected = shallowRef<RiskCase | null>(null)
const pageSize = 10
const filtered = computed(() => props.cases.filter((item) => (!query.value || `${item.id} ${item.summary} ${item.owner}`.toLocaleLowerCase('zh-CN').includes(query.value.toLocaleLowerCase('zh-CN'))) && (risk.value === 'ALL' || item.riskLevel === risk.value) && (status.value === 'ALL' || item.status === status.value)))
const visible = computed(() => filtered.value.slice((page.value - 1) * pageSize, page.value * pageSize))
</script>

<template>
  <section class="data-view glass"><header><div><h2>个案列表</h2><p>集中查看风险等级、跟进状态和交接摘要。</p></div><span class="count-chip mono">{{ filtered.length }} 个个案</span></header>
    <div class="filters"><el-input v-model="query" clearable placeholder="搜索摘要、负责人或个案编号…" aria-label="搜索风险个案" /><el-select v-model="risk" aria-label="筛选风险等级"><el-option label="全部风险" value="ALL"/><el-option label="低风险" value="LOW"/><el-option label="中风险" value="MEDIUM"/><el-option label="高风险" value="HIGH"/></el-select><el-select v-model="status" aria-label="筛选个案状态"><el-option label="全部状态" value="ALL"/><el-option label="待跟进" value="OPEN"/><el-option label="已预警" value="ALERT_SENT"/><el-option label="已确认" value="ACKNOWLEDGED"/><el-option label="已结案" value="CLOSED"/></el-select></div>
    <el-table :data="visible" empty-text="暂无符合条件的个案" row-key="id" @row-click="selected = $event">
      <el-table-column prop="id" label="个案" width="90"><template #default="scope"><span class="mono">#{{ scope.row.id }}</span></template></el-table-column>
      <el-table-column label="风险" width="100"><template #default="scope"><el-tag :type="scope.row.riskLevel === 'HIGH' ? 'danger' : scope.row.riskLevel === 'MEDIUM' ? 'warning' : 'success'" effect="light">{{ statusLabel(scope.row.riskLevel) }}</el-tag></template></el-table-column>
      <el-table-column prop="summary" label="摘要" min-width="260" show-overflow-tooltip />
      <el-table-column prop="owner" label="负责人" width="130" />
      <el-table-column label="状态" width="110"><template #default="scope">{{ statusLabel(scope.row.status) }}</template></el-table-column>
      <el-table-column label="更新时间" width="180"><template #default="scope">{{ formatDateTime(scope.row.updatedAt) }}</template></el-table-column>
    </el-table>
    <el-pagination v-if="filtered.length > pageSize" v-model:current-page="page" background layout="prev, pager, next" :page-size="pageSize" :total="filtered.length" />
    <el-drawer :model-value="Boolean(selected)" direction="rtl" size="min(520px, 92vw)" title="个案交接详情" destroy-on-close @close="selected = null">
      <template v-if="selected"><div class="drawer-meta"><el-tag :type="selected.riskLevel === 'HIGH' ? 'danger' : 'warning'">{{ statusLabel(selected.riskLevel) }}</el-tag><span>{{ statusLabel(selected.status) }}</span><span>{{ formatDateTime(selected.updatedAt) }}</span></div><h3>个案 #{{ selected.id }}</h3><p class="drawer-summary">{{ selected.summary || '暂无摘要' }}</p><h4>交接摘要</h4><pre>{{ selected.handoffSummary || '暂无交接摘要' }}</pre><dl><div><dt>负责人</dt><dd>{{ selected.owner || '未分配' }}</dd></div><div><dt>关联报告</dt><dd>#{{ selected.reportId }}</dd></div><div><dt>确认人</dt><dd>{{ selected.acknowledgedBy || '尚未确认' }}</dd></div></dl><CaseFollowup :case-item="selected" @updated="updateCase" /></template>
    </el-drawer>
  </section>
</template>

<style scoped>
.data-view { min-width: 0; padding: 22px; border-radius: var(--radius-lg); }.data-view > header { display: flex; align-items: flex-end; justify-content: space-between; gap: 20px; }.data-view h2 { margin: 0; font-family: var(--font-display); font-size: 30px; letter-spacing: -.045em; }.data-view header p:last-child { margin: 8px 0 0; color: var(--text-soft); }.count-chip { padding: 8px 11px; border: 1px solid var(--line); border-radius: var(--radius-pill); color: var(--text-faint); font-size: 11px; }.filters { display: grid; grid-template-columns: minmax(220px, 1fr) 170px 170px; gap: 10px; margin: 24px 0 16px; }.el-pagination { justify-content: flex-end; margin-top: 18px; }.drawer-meta { display: flex; align-items: center; gap: 10px; color: var(--text-faint); font-size: 12px; }.drawer-summary { line-height: 1.7; }.data-view pre { padding: 16px; overflow-wrap: anywhere; white-space: pre-wrap; border: 1px solid var(--line); border-radius: 14px; color: var(--text-soft); background: var(--surface-soft); line-height: 1.7; }.data-view dl div { display: flex; justify-content: space-between; gap: 14px; padding: 10px 0; border-bottom: 1px solid var(--line); }.data-view dt { color: var(--text-faint); }.data-view dd { margin: 0; }
@media (max-width: 720px) { .data-view { padding: 16px; }.filters { grid-template-columns: 1fr; }.data-view > header { align-items: flex-start; }.count-chip { display: none; } }
</style>
