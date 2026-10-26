<script setup lang="ts">
import type { RetrievalDetail } from '@/types/api'
defineProps<{ groups: RetrievalDetail[] }>()
const labels: Record<string, string> = { dense: '密集向量召回（Dense）', sparse: '学习型稀疏召回（Sparse）', bm25: 'BM25 词法召回', milvus_rrf: 'Milvus RRF 融合', rrf: '跨查询 RRF 候选集', rerank: '重排（Reranker）', evidence: '最终参考证据' }
const methods: Record<string, string> = { cosine: '余弦相似度', inner_product: '内积', bm25: 'BM25', rrf: 'RRF 分数', semantic_local_cross_encoder: '本地 Cross-Encoder', local_cross_encoder: '本地 Cross-Encoder', semantic_api: '语义重排', lexical: '词法重排', selection: '证据选择' }
function score(value: number) { return Number.isFinite(value) ? value.toFixed(6) : '—' }
</script>

<template>
  <div class="retrieval-details">
    <details v-for="(group, index) in groups" :key="`${group.channel}-${group.queryIndex}-${index}`" class="retrieval-lane" open>
      <summary>{{ labels[group.channel] || group.channel }} · {{ group.total }} 条 <small>查询 {{ group.queryIndex }} · {{ methods[group.method] || group.method }}</small></summary>
      <p v-if="!group.candidates.length" class="empty">本阶段无命中资料。</p>
      <ol v-else class="candidates">
        <li v-for="candidate in group.candidates" :key="`${candidate.rank}-${candidate.chunkId}`">
          <div class="candidate-heading"><b>#{{ candidate.rank }} {{ candidate.source }}</b><span>切片 {{ candidate.chunkId }} · {{ score(candidate.score) }}</span></div>
          <p>{{ candidate.content }}</p>
        </li>
      </ol>
      <p v-if="group.total > group.candidates.length" class="empty">展示前 {{ group.candidates.length }} 条，共 {{ group.total }} 条。</p>
    </details>
  </div>
</template>

<style scoped>
.retrieval-details { margin: 8px 0 0 24px; }.retrieval-lane { margin-top: 8px; border: 1px solid #b7c8ad; border-radius: 6px; padding: 8px; }.retrieval-lane summary { cursor: pointer; font-weight: 600; line-height: 1.7; }.retrieval-lane small { font-weight: 400; }.candidates { max-height: 320px; overflow-y: auto; list-style: none; padding: 0 6px 0 0; margin: 8px 0 0; }.candidates li { padding: 8px 0; border-top: 1px solid #c5d2bb; }.candidate-heading { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 5px; overflow-wrap: anywhere; }.candidate-heading span { font: 10px/1.7 var(--font-mono); }.candidates p, .empty { margin: 5px 0 0; white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.7; }
</style>
