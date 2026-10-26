<script setup lang="ts">
import { computed, shallowRef, onMounted, watch } from 'vue'
import { requestJson } from '@/core/api'
import { ElMessage, ElMessageBox, type UploadFile } from 'element-plus'
import type { KnowledgeStatus } from '@/types/api'

const props = defineProps<{ knowledge: KnowledgeStatus | null; busy?: boolean; actionBusy?: 'rebuild-vector' | 'backup' | null }>()
const milvusBackend = computed(() => props.knowledge?.backend === 'milvus_bge')
const emit = defineEmits<{ upload: [file: File]; action: [action: 'rebuild-vector' | 'backup'] }>()
const file = shallowRef<File | null>(null)
function selectFile(uploadFile: UploadFile) {
  const raw = uploadFile.raw
  if (!raw) return
  const allowed = ['application/pdf', 'text/plain', 'text/markdown', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'].includes(raw.type) || /(\.(pdf|docx|md|markdown|txt)|\.metadata\.json)$/i.test(raw.name)
  if (!allowed) { file.value = null; ElMessage.error('请选择 PDF、Word、Markdown 或 txt 文件'); return }
  if (raw.size > 10 * 1024 * 1024) { file.value = null; ElMessage.error('文件不能超过 10 MB'); return }
  file.value = raw; preview.value = null
}
function upload() { if (!file.value || !preview.value) return ElMessage.warning('请先预览并确认解析内容'); emit('upload', file.value) }
const preview = shallowRef<{markdown: string; parser: string; source: string} | null>(null), previewing = shallowRef(false)
const documents = shallowRef<{source: string; document_id: string; chunks: number; version: string; status: string}[]>([])
async function parsePreview() {
  if (!file.value) return
  previewing.value = true
  try { const data = new FormData(); data.append('file', file.value); preview.value = await requestJson('/api/admin/knowledge/preview', {method: 'POST', body: data}) }
  catch (error) { ElMessage.error(error instanceof Error ? error.message : '解析失败') }
  finally { previewing.value = false }
}
const contentOpen = shallowRef(false), contentLoading = shallowRef(false)
const contentTitle = shallowRef(''), contentDocumentId = shallowRef('')
const contentChunks = shallowRef<{id: number; content: string}[]>([]), contentTotal = shallowRef(0)
async function viewDocument(document: {document_id: string; source: string}, more = false) {
  if (contentLoading.value) return
  contentOpen.value = true; contentLoading.value = true
  if (!more) { contentChunks.value = []; contentTitle.value = document.source; contentDocumentId.value = document.document_id }
  try {
    const data = await requestJson<{total: number; chunks: {id: number; content: string}[]}>(`/api/admin/knowledge/document-content?document_id=${encodeURIComponent(document.document_id)}&offset=${contentChunks.value.length}`)
    contentChunks.value = [...contentChunks.value, ...data.chunks]; contentTotal.value = data.total
  } catch (error) { ElMessage.error(error instanceof Error ? error.message : '内容读取失败') }
  finally { contentLoading.value = false }
}
async function loadDocuments() { try { documents.value = await requestJson('/api/admin/knowledge/documents') } catch (error) { ElMessage.error(error instanceof Error ? error.message : '文档读取失败') } }
async function removeDocument(documentId: string) {
  await ElMessageBox.confirm('删除后该文档将不再参与检索，历史引用仍保留。', '删除文档', {type: 'warning'})
  try { await requestJson(`/api/admin/knowledge/documents?document_id=${encodeURIComponent(documentId)}`, {method:'DELETE'}); await loadDocuments(); ElMessage.success('文档已删除') }
  catch (error) { ElMessage.error(error instanceof Error ? error.message : '删除失败') }
}
watch(() => props.busy, (busy, previous) => { if (previous && !busy) void loadDocuments() })
onMounted(loadDocuments)
async function confirmAction(action: 'rebuild-vector' | 'backup') { if (action === 'rebuild-vector') await ElMessageBox.confirm('重建会重新生成全部向量索引，期间检索可能短暂降级。', '确认重建索引', { confirmButtonText: '开始重建', cancelButtonText: '取消', type: 'warning' }); emit('action', action) }
</script>

<template>
  <section class="knowledge-view"><article class="knowledge-hero glass"><div><h2>知识库状态</h2><p>上传资料、查看来源，并维护检索索引。</p><div class="knowledge-stats"><span><strong class="mono">{{ knowledge?.databaseChunks ?? 0 }}</strong><small>数据库片段</small></span><span><strong class="mono">{{ knowledge?.vectorChunks ?? 0 }}</strong><small>向量片段</small></span><span><strong>{{ knowledge?.vectorAvailable ? '可用' : '待恢复' }}</strong><small>向量状态</small></span></div></div><img class="knowledge-botanical" src="/images/teacher-leaves.webp" alt="" loading="lazy" /></article>
    <div class="knowledge-grid"><article class="upload-panel glass"><h3>上传知识文件</h3><el-upload drag :auto-upload="false" :limit="1" accept=".pdf,.docx,.pptx,.xlsx,.md,.markdown,.txt,.html,.csv,.png,.jpg,.jpeg,.metadata.json" :on-change="selectFile"><div class="upload-symbol" aria-hidden="true">⇧</div><div class="el-upload__text">拖放文件到这里，或<em>点击选择</em></div><template #tip><div class="el-upload__tip">支持 PDF、Word、PPTX、XLSX、文本、HTML、CSV、图片及转换元数据，最大 10 MB</div></template></el-upload><button class="secondary-button upload-action" type="button" :disabled="!file || previewing" @click="parsePreview">{{ previewing ? '正在解析…' : '解析并预览' }}</button><details v-if="preview" open><summary>解析预览 · {{ preview.parser }}</summary><pre class="preview-text">{{ preview.markdown }}</pre></details><button class="primary-button upload-action" type="button" :disabled="busy || !file || !preview" @click="upload">{{ busy ? '正在上传…' : '确认并写入知识库' }}</button></article>
      <article class="maintenance-panel glass"><h3>向量索引维护</h3><div class="maintenance-action"><span><strong>重建向量索引</strong><small>重新处理数据库中的全部知识片段</small></span><button class="secondary-button" type="button" :disabled="Boolean(actionBusy)" @click="confirmAction('rebuild-vector')">{{ actionBusy === 'rebuild-vector' ? '重建中…' : '开始重建' }}</button></div><div class="maintenance-action"><span><strong>备份当前索引</strong><small>{{ milvusBackend ? 'Milvus 使用持久化卷备份，当前入口暂不支持' : '创建可恢复的 Chroma 快照' }}</small></span><button class="secondary-button" type="button" :disabled="Boolean(actionBusy) || milvusBackend" @click="confirmAction('backup')">{{ milvusBackend ? '使用卷备份' : actionBusy === 'backup' ? '备份中…' : '立即备份' }}</button></div><div class="boundary-note"><strong>文档处理</strong><p>{{ milvusBackend ? '本机使用 Docling 进行版面解析与 OCR，可直接上传 PDF、Office 文件或扫描图片。' : '轻量版支持导入本机转换后的 Markdown 或 metadata.json。' }}更新失败时保留原有知识版本。</p></div></article></div>
    <article class="upload-panel glass"><header><h3>知识文档</h3><button class="secondary-button" @click="loadDocuments">刷新文档</button></header><div class="documents-table"><table v-if="documents.length"><thead><tr><th>来源</th><th>片段数</th><th>状态</th><th>版本</th><th>操作</th></tr></thead><tbody><tr v-for="document in documents" :key="document.document_id"><td><button class="secondary-button" @click="viewDocument(document)">{{ document.source }}</button></td><td>{{ document.chunks }}</td><td>{{ document.status }}</td><td>{{ document.version?.slice(0,12) }}</td><td><button class="secondary-button" @click="removeDocument(document.document_id)">删除</button></td></tr></tbody></table><p v-else>暂无知识文档</p></div></article>
    <el-drawer v-model="contentOpen" :title="contentTitle" size="min(760px, 96vw)">
      <p v-if="contentLoading">正在读取最新内容…</p>
      <article v-for="chunk in contentChunks" :key="chunk.id"><small>片段 #{{ chunk.id }}</small><pre class="preview-text">{{ chunk.content }}</pre></article>
      <button v-if="contentChunks.length < contentTotal" class="secondary-button" :disabled="contentLoading" @click="viewDocument({document_id: contentDocumentId, source: contentTitle}, true)">加载更多</button>
    </el-drawer>
  </section>
</template>

<style scoped>
.documents-table { overflow-x:auto; } .documents-table table { width:100%; border-collapse:collapse; } .documents-table th, .documents-table td { text-align:left; padding:12px; border-bottom:1px solid var(--line); }
.preview-text { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 320px; overflow:auto; padding:12px; background:var(--surface-soft); line-height:1.7; }
.knowledge-view { display: grid; gap: 14px; }.knowledge-hero { position: relative; min-height: 0; display: flex; align-items: center; overflow: hidden; padding: 24px; border-radius: var(--radius-lg); }.knowledge-botanical { width: 26%; max-width: 260px; height: 186px; object-fit: cover; margin-left: auto; border-radius: 65% 65% 16px 16px; opacity: .9; }.knowledge-hero > div:first-child { position: relative; z-index: 2; max-width: 680px; }.knowledge-hero h2 { max-width: 640px; margin: 0; font-family: var(--font-display); font-size: 26px; line-height: 1.02; letter-spacing: -.055em; }.knowledge-hero p { max-width: 600px; color: var(--text-soft); line-height: 1.7; }.knowledge-stats { display: flex; gap: 12px; margin-top: 20px; flex-wrap: wrap; }.knowledge-stats > span { min-width: 130px; display: grid; padding: 12px 14px; border: 1px solid var(--line); border-radius: 14px; background: var(--surface-soft); }.knowledge-stats strong { font-family: var(--font-display); font-size: 22px; }.knowledge-stats small { margin-top: 3px; color: var(--text-faint); }.knowledge-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }.upload-panel, .maintenance-panel { padding: 24px; border-radius: var(--radius-lg); }.knowledge-grid h3 { margin: 0 0 20px; font-family: var(--font-display); font-size: 24px; }.upload-symbol { margin-bottom: 10px; color: var(--lagoon); font-size: 32px; }.upload-action { width: 100%; margin-top: 16px; }.maintenance-action { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 18px 0; border-bottom: 1px solid var(--line); }.maintenance-action span { display: grid; gap: 5px; }.maintenance-action small { color: var(--text-faint); }.boundary-note { margin-top: 22px; padding: 15px; border-radius: 14px; color: var(--text-soft); background: var(--surface-soft); }.boundary-note p { margin: 6px 0 0; font-size: 12px; line-height: 1.6; }
@media (max-width: 860px) { .knowledge-botanical { display: none; } .knowledge-grid { grid-template-columns: 1fr; } }
</style>
