<script setup lang="ts">
import { applyTheme } from '@/core/theme'
import { computed, defineAsyncComponent, onBeforeUnmount, onMounted, shallowRef } from 'vue'
import { storeToRefs } from 'pinia'
import { ElMessage } from 'element-plus'
import AdminSidebar from '@/components/admin/AdminSidebar.vue'
import SystemStatus from '@/components/shared/SystemStatus.vue'
import { parseAdminSection, updateAdminSection, type AdminSection } from '@/core/admin-view'
import { formatDateTime } from '@/core/format'
import { useAdminStore } from '@/stores/admin'
import { useAuthStore } from '@/stores/auth'
import { useSystemStore } from '@/stores/system'
import type { Report } from '@/types/api'

const auth = useAuthStore(), admin = useAdminStore(), system = useSystemStore()
const OverviewView = defineAsyncComponent(() => import('@/components/admin/OverviewView.vue'))
const CasesView = defineAsyncComponent(() => import('@/components/admin/CasesView.vue'))
const ReportsView = defineAsyncComponent(() => import('@/components/admin/ReportsView.vue'))
const DeliveryView = defineAsyncComponent(() => import('@/components/admin/DeliveryView.vue'))
const KnowledgeView = defineAsyncComponent(() => import('@/components/admin/KnowledgeView.vue'))
const { profile } = storeToRefs(auth)
const { reports, cases, excel, alerts, knowledge, conversation, detailLoading, metrics, loading, error, updatedAt } = storeToRefs(admin)
const section = shallowRef<AdminSection>(parseAdminSection(window.location.search))
const knowledgeBusy = shallowRef(false)
const knowledgeActionBusy = shallowRef<'rebuild-vector' | 'backup' | null>(null)
const displayName = computed(() => profile.value?.displayName || profile.value?.username || '管理员')
const titles: Record<AdminSection, { title: string; note: string }> = {
  overview: { title: '工作概览', note: '理解咨询变化，及时关注需要跟进的风险。' },
  cases: { title: '风险个案', note: '跟进风险状态与交接信息。' },
  reports: { title: '咨询报告', note: '查找报告并审阅完整会话档案。' },
  delivery: { title: '工具执行', note: '核对台账写入和预警投递结果。' },
  knowledge: { title: '知识库维护', note: '保持校园心理支持知识准确可用。' },
}

function navigate(next: AdminSection) { section.value = next; window.history.pushState({ view: next }, '', updateAdminSection(window.location.href, next)) }
function syncHistory() { section.value = parseAdminSection(window.location.search) }
async function logout() {
  try { await auth.logout(); window.location.assign('/') }
  catch { window.alert(auth.error || '退出失败，请稍后重试') }
}
async function openReport(report: Report) { navigate('reports'); await admin.loadConversation(report.sessionId) }
async function uploadKnowledge(file: File) { knowledgeBusy.value = true; try { const result = await admin.uploadKnowledge(file); ElMessage.success(`${result.source} 已写入 ${result.chunks} 个片段`) } catch (reason) { ElMessage.error(reason instanceof Error ? reason.message : '上传失败') } finally { knowledgeBusy.value = false } }
async function runKnowledgeAction(action: 'rebuild-vector' | 'backup') { knowledgeActionBusy.value = action; try { const result = await admin.runKnowledgeAction(action); ElMessage.success(action === 'backup' ? `备份完成：${result.snapshot}` : `重建完成：${result.indexedChunks} 个片段`) } catch (reason) { ElMessage.error(reason instanceof Error ? reason.message : '操作失败') } finally { knowledgeActionBusy.value = null } }

onMounted(async () => {
  applyTheme()
  const nextProfile = await auth.resume()
  if (!nextProfile) return window.location.replace('/')
  if (!auth.admin) return window.location.replace('/student.html')
  window.addEventListener('popstate', syncHistory)
  void system.load()
  void admin.loadDashboard()
})
onBeforeUnmount(() => window.removeEventListener('popstate', syncHistory))
</script>

<template>
  <a class="skip-link" href="#admin-content">跳到主要内容</a>
  <div class="admin-page">
    <AdminSidebar :section="section" :display-name="displayName" @navigate="navigate" @logout="logout" />
    <div class="admin-workspace page-enter">
      <header class="workspace-header">
        <div><p class="workspace-context">校园心理支持 / 管理工作台</p><h1>{{ titles[section].title }}</h1><p>{{ titles[section].note }}</p></div>
        <div class="workspace-actions"><SystemStatus /><span v-if="updatedAt" class="updated-time">更新于 {{ formatDateTime(updatedAt.toISOString()) }}</span><button class="secondary-button" type="button" :disabled="loading" @click="admin.loadDashboard">{{ loading ? '刷新中…' : '刷新数据' }}</button><button class="quiet-button mobile-logout" type="button" @click="logout">退出</button></div>
      </header>
      <p v-if="error" class="admin-error" role="alert">数据读取失败：{{ error }}。请检查服务状态后重新刷新。</p>
      <main id="admin-content" class="workspace-content">
        <OverviewView v-if="section === 'overview'" :display-name="displayName" :metrics="metrics" :reports="reports" :cases="cases" :excel="excel" :alerts="alerts" @open-report="openReport" />
        <CasesView v-else-if="section === 'cases'" :cases="cases" @updated="admin.loadDashboard" />
        <ReportsView v-else-if="section === 'reports'" :reports="reports" :conversation="conversation" :detail-loading="detailLoading" @open="admin.loadConversation($event.sessionId)" />
        <DeliveryView v-else-if="section === 'delivery'" :excel="excel" :alerts="alerts" />
        <KnowledgeView v-else :knowledge="knowledge" :busy="knowledgeBusy" :action-busy="knowledgeActionBusy" @upload="uploadKnowledge" @action="runKnowledgeAction" />
      </main>
    </div>
  </div>
</template>

<style scoped>
.admin-page { position: relative; min-height: 100vh; display: grid; grid-template-columns: 224px minmax(0, 1fr); gap: 0; padding: 0; isolation: isolate; }.admin-workspace { position: relative; z-index: 2; }.admin-workspace { min-width: 0; padding: 28px clamp(20px, 3vw, 44px); max-width: 1700px; width: 100%; }.workspace-header { min-height: 96px; display: flex; align-items: center; justify-content: space-between; gap: 24px; padding: 0 0 24px; margin-bottom: 22px; border-bottom: 1px solid var(--line); }.workspace-header h1 { margin: 0; font-family: var(--font-display); font-size: clamp(26px, 3vw, 34px); letter-spacing: -.025em; }.workspace-header > div:first-child > p:last-child { margin: 5px 0 0; color: var(--text-soft); font-size: 12px; }.workspace-actions { display: flex; align-items: center; justify-content: flex-end; gap: 10px; flex-wrap: wrap; }.updated-time { color: var(--text-faint); font-family: var(--font-mono); font-size: 11px; }.workspace-content { min-width: 0; padding-bottom: 20px; scroll-margin-top: 16px; }.admin-error { margin: 0 0 14px; padding: 12px 15px; border: 1px solid color-mix(in srgb, var(--danger) 32%, transparent); border-radius: 13px; color: var(--danger); background: color-mix(in srgb, var(--danger) 8%, var(--surface)); }.mobile-logout { display: none; }.workspace-context { margin: 0 0 12px; font-size: 12px; color: var(--text-faint); }
@media (max-width: 980px) { .admin-page { grid-template-columns: 1fr; }.workspace-header { padding: 10px 5px 16px; }.updated-time { display: none; }.mobile-logout { display: inline-flex; } }
@media (max-width: 650px) { .admin-page { padding: 0; gap: 0; }.admin-workspace { padding: 20px 14px; }.workspace-header { align-items: flex-start; flex-direction: column; }.workspace-actions { width: 100%; justify-content: flex-start; }.workspace-actions :deep(.status-chip.model) { display: none; } }
.admin-page::before { content: ''; position: fixed; inset: 0; z-index: -1; background: url('/images/teacher-garden.webp') center / cover; opacity: .23; pointer-events: none; }
</style>
