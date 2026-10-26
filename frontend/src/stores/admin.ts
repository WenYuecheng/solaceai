import { computed, ref, shallowRef } from 'vue'
import { defineStore } from 'pinia'
import { buildDashboardMetrics } from '@/core/analytics'
import { request, requestJson } from '@/core/api'
import type { Conversation, KnowledgeStatus, Report, RiskCase, ToolRecord } from '@/types/api'

export const useAdminStore = defineStore('admin', () => {
  const reports = ref<Report[]>([])
  const cases = ref<RiskCase[]>([])
  const excel = ref<ToolRecord[]>([])
  const alerts = ref<ToolRecord[]>([])
  const knowledge = shallowRef<KnowledgeStatus | null>(null)
  const conversation = shallowRef<Conversation | null>(null)
  const loading = shallowRef(false)
  const detailLoading = shallowRef(false)
  const error = shallowRef('')
  const updatedAt = shallowRef<Date | null>(null)
  const metrics = computed(() => buildDashboardMetrics(reports.value, cases.value, excel.value, alerts.value))

  async function loadDashboard() {
    loading.value = true
    error.value = ''
    try {
      const [nextReports, nextCases, nextExcel, nextAlerts, nextKnowledge] = await Promise.all([
        requestJson<Report[]>('/api/admin/reports'),
        requestJson<RiskCase[]>('/api/admin/cases'),
        requestJson<ToolRecord[]>('/api/admin/excel-records'),
        requestJson<ToolRecord[]>('/api/admin/alerts'),
        requestJson<KnowledgeStatus>('/api/admin/knowledge/status'),
      ])
      reports.value = nextReports
      cases.value = nextCases
      excel.value = nextExcel
      alerts.value = nextAlerts
      knowledge.value = nextKnowledge
      updatedAt.value = new Date()
    } catch (reason) {
      error.value = reason instanceof Error ? reason.message : '管理数据读取失败'
    } finally { loading.value = false }
  }

  async function loadConversation(sessionId: string) {
    detailLoading.value = true
    try { conversation.value = await requestJson<Conversation>(`/api/admin/conversations/${encodeURIComponent(sessionId)}`) }
    finally { detailLoading.value = false }
  }

  async function uploadKnowledge(file: File) {
    const data = new FormData()
    data.append('file', file)
    const response = await request('/api/admin/knowledge/file', { method: 'POST', body: data })
    const result = await response.json() as { source: string; chunks: number }
    knowledge.value = await requestJson<KnowledgeStatus>('/api/admin/knowledge/status')
    return result
  }

  async function runKnowledgeAction(action: 'rebuild-vector' | 'backup') {
    const result = await requestJson<Record<string, string | number>>(`/api/admin/knowledge/${action}`, { method: 'POST' })
    knowledge.value = await requestJson<KnowledgeStatus>('/api/admin/knowledge/status')
    return result
  }

  return { reports, cases, excel, alerts, knowledge, conversation, loading, detailLoading, error, updatedAt, metrics, loadDashboard, loadConversation, uploadKnowledge, runKnowledgeAction }
})
