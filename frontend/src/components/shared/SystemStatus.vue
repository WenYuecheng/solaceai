<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { useSystemStore } from '@/stores/system'

const props = withDefaults(defineProps<{ showModel?: boolean }>(), { showModel: true })
const system = useSystemStore()
const { healthy, modelLabel } = storeToRefs(system)
</script>

<template>
  <div class="system-status" aria-live="polite">
    <span class="status-chip" :class="healthy === true ? 'ok' : healthy === false ? 'danger' : 'pending'">
      <i aria-hidden="true"></i>{{ healthy === true ? '服务正常' : healthy === false ? '服务中断' : '连接中' }}
    </span>
    <span v-if="props.showModel" class="status-chip model"><i aria-hidden="true"></i>{{ modelLabel }}</span>
  </div>
</template>

<style scoped>
.system-status { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.status-chip { min-height: 34px; display: inline-flex; align-items: center; gap: 8px; padding: 0 12px; border: 1px solid var(--line); border-radius: var(--radius-pill); color: var(--text-soft); background: var(--surface-soft); font-family: var(--font-body); font-size: 11px; font-weight: 500; white-space: nowrap; }
.status-chip i { width: 7px; height: 7px; border-radius: 50%; background: var(--text-faint);  }
.status-chip.ok i { background: var(--success); }.status-chip.danger i { background: var(--danger); }.status-chip.model i { background: var(--violet); }
</style>
