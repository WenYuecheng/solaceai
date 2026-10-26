<script setup lang="ts">
import { computed } from 'vue'
import type { EvidenceCitation } from '@/types/api'
const props = defineProps<{ citations: EvidenceCitation[]; interactive: boolean }>()
const emit = defineEmits<{ evidence: [citation: EvidenceCitation] }>()
const groups = computed(() => {
  const seen = new Set<string>()
  const unique = props.citations.filter((item, index) => {
    const key = item.chunkId != null ? String(item.chunkId) : `${item.source}:${item.preview || index}`
    if (seen.has(key)) return false
    seen.add(key); return true
  })
  return [
    { label: '正文已引用', items: unique.filter(item => item.cited === true) },
    { label: '本轮检索参考', items: unique.filter(item => item.cited === false) },
    { label: '参考依据', items: unique.filter(item => item.cited == null) },
  ].filter(group => group.items.length)
})
</script>
<template>
  <details class="message-sources" open>
    <summary>参考来源 · {{ groups.reduce((count, group) => count + group.items.length, 0) }} 条</summary>
    <section v-for="group in groups" :key="group.label">
      <h4>{{ group.label }} · {{ group.items.length }} 条</h4>
      <p v-for="(citation, index) in group.items" :key="citation.chunkId ?? index">
        <button type="button" class="source-link" :disabled="!interactive || citation.chunkId == null" @click="emit('evidence', citation)">{{ citation.title || citation.source }}</button>
        <span v-if="citation.title && citation.title !== citation.source"> · {{ citation.source }}</span><br>
        <span v-if="citation.sectionPath?.length">{{ citation.sectionPath.join(' / ') }}<br></span>{{ citation.preview }}
      </p>
    </section>
  </details>
</template>
<style scoped>
.message-sources { color: var(--text-soft); font-size: 12px; line-height: 1.6; margin: 8px 4px; }
summary { cursor: pointer; }
h4 { font-size: inherit; margin: 12px 0 4px; }
.source-link { color: inherit; background: transparent; border: 0; cursor: pointer; text-decoration: underline; }
.source-link:disabled { cursor: default; text-decoration: none; }
</style>
