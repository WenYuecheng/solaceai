<script setup lang="ts">
import { shallowRef } from 'vue'
import AccountCenter from '@/components/auth/AccountCenter.vue'
import BrandMark from '@/components/shared/BrandMark.vue'
import SystemStatus from '@/components/shared/SystemStatus.vue'
import ChatModelSelector from '@/components/student/ChatModelSelector.vue'

const accountOpen = shallowRef(false)
defineProps<{ displayName: string; sidebarOpen?: boolean }>()
const emit = defineEmits<{ logout: []; reset: []; openSidebar: [] }>()
</script>

<template>
  <header class="student-header">
    <div class="header-brand"><button class="icon-button sidebar-toggle" type="button" aria-label="打开会话侧栏" aria-controls="student-navigation" :aria-expanded="sidebarOpen" @click="emit('openSidebar')"><span aria-hidden="true">☰</span></button><BrandMark compact context="校园心理支持" /></div>
    <div class="header-actions">
      <SystemStatus :show-model="false" />
      <ChatModelSelector />
      <div class="student-account"><strong>{{ displayName }}</strong><span>学生端</span></div>
      <button class="secondary-button reset-header" type="button" @click="emit('reset')">新会话</button>
      <button class="quiet-button" type="button" @click="accountOpen = true">个人中心</button>
      <button class="quiet-button" type="button" @click="emit('logout')">退出</button>
    </div>
  </header>
  <AccountCenter v-if="accountOpen" @close="accountOpen = false" />
</template>

<style scoped>
.student-header { position: relative; z-index: 5; min-height: 72px; display: flex; align-items: center; justify-content: space-between; gap: 20px; padding: 13px 28px; border-bottom: 1px solid #e1e6d3; background: #fafbef; }
.header-actions { display: flex; align-items: center; justify-content: flex-end; gap: 9px; }.student-account { min-width: 100px; display: grid; text-align: right; }.student-account strong { max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }.student-account span { color: var(--text-faint); font-size: 11px; }
@media (max-width: 880px) { .header-actions :deep(.status-chip.model), .reset-header, .student-account { display: none; } }
@media (max-width: 620px) { .student-header { align-items: flex-start; }.header-actions { gap: 5px; }.student-account, .header-actions :deep(.system-status) { display: none; }.quiet-button { padding: 0 8px; } }
.header-brand { display: flex; align-items: center; gap: 12px; }.sidebar-toggle { display: none; background: transparent; border: 0; font-size: 20px; }
@media (max-width: 1020px) { .sidebar-toggle { display: inline-grid; } }
@media (max-width: 720px) { .student-header { min-height: 68px; padding: 10px 16px 10px 10px; align-items: center; }.header-brand { gap: 6px; }.header-brand :deep(.brand-copy small) { display: none; }.header-brand :deep(.brand-symbol) { display: none; } }
@media (max-width: 620px) { .student-header { flex-wrap: wrap; gap: 8px; }.header-actions { flex: 1 1 100%; display: grid; grid-template-columns: minmax(0, 1fr) auto auto; align-items: center; gap: 8px; }.header-actions :deep(.chat-model-selector) { justify-self: start; } }
</style>
