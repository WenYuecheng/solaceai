<script setup lang="ts">
import type { AdminSection } from '@/core/admin-view'

defineProps<{ section: AdminSection; displayName: string }>()
const emit = defineEmits<{ navigate: [section: AdminSection]; logout: [] }>()
const items: Array<{ key: AdminSection; icon: string; label: string }> = [
  { key: 'overview', icon: '⌁', label: '数据概览' },
  { key: 'cases', icon: '◇', label: '风险个案' },
  { key: 'reports', icon: '▤', label: '咨询报告' },
  { key: 'delivery', icon: '⇄', label: '工具执行' },
  { key: 'knowledge', icon: '◫', label: '知识库' },
]
</script>

<template>
  <aside class="admin-sidebar">
    <a class="admin-brand" href="/admin.html"><span class="brand-initial" aria-hidden="true">S</span><span><strong>SolaceAI</strong><small>校园心理支持</small></span></a>
    <nav aria-label="管理工作台">
      <button v-for="item in items" :key="item.key" type="button" :class="{ active: section === item.key }" :aria-label="item.label" :aria-current="section === item.key ? 'page' : undefined" @click="emit('navigate', item.key)">
        <span class="nav-icon" aria-hidden="true">{{ item.icon }}</span><span><strong>{{ item.label }}</strong></span>
      </button>
    </nav>
    <div class="sidebar-bottom">
      <p class="botanical-caption">让每一份关心，都有回应。</p>
      <div class="admin-profile"><span>{{ displayName.slice(0, 1).toUpperCase() }}</span><div><strong>{{ displayName }}</strong><small>心理教师 · 管理员</small></div></div>
      <div class="bottom-actions"><button class="quiet-button" type="button" @click="emit('logout')">退出</button></div>
    </div>
  </aside>
</template>

<style scoped>
.admin-sidebar { position: sticky; top: 0; height: 100vh; height: 100dvh; align-self: start; overflow-y: auto; overscroll-behavior: contain; display: flex; flex-direction: column; padding: 30px 18px 22px; border-right: 1px solid var(--line); background: #edf3e9; isolation: isolate; }
.admin-brand { display: flex; gap: 10px; align-items: center; color: var(--text); text-decoration: none; padding: 0 8px; }.brand-initial { width: 34px; height: 38px; display: grid; place-items: center; border-radius: 8px; background: var(--lagoon); color: white; font-family: Georgia, serif; font-size: 26px; }.admin-brand strong { font-family: var(--font-display); font-size: 19px; }.admin-brand small { display: block; margin-top: 3px; font-size: 11px; color: var(--text-faint); }
nav { display: grid; gap: 6px; margin-top: 38px; }nav button { min-height: 44px; display: flex; align-items: center; gap: 12px; padding: 10px 12px; border: 0; border-radius: 13px; color: var(--text-soft); background: transparent; text-align: left; }nav button:hover { color: var(--text); background: var(--surface-soft); }nav button.active { color: var(--lagoon); background: #dce9d7; box-shadow: none; }.nav-icon { width: 22px; text-align: center; font-size: 20px; }nav strong { font-size: 13px; font-weight: 600; }
.sidebar-garden { margin: 0 0 4px; overflow: hidden; border-radius: 48px 48px 14px 14px; background: #e0eadb; }.sidebar-garden img { width: 100%; height: 112px; display: block; object-fit: cover; }.sidebar-garden figcaption { padding: 11px 8px; color: var(--text-soft); font-size: 11px; text-align: center; }
.sidebar-bottom { margin-top: auto; display: grid; gap: 16px; padding: 26px 8px 0; }.admin-profile { display: flex; align-items: center; gap: 10px; }.admin-profile > span { width: 32px; height: 32px; display: grid; place-items: center; border-radius: 50%; background: var(--surface-soft); color: var(--lagoon); }.admin-profile div { min-width: 0; display: grid; gap: 3px; }.admin-profile strong { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px; }.admin-profile small { color: var(--text-faint); font-size: 10px; }.bottom-actions { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
@media (max-width: 980px) { .admin-sidebar { position: sticky; top: 0; z-index: 30; width: 100%; height: auto; overflow: visible; padding: 16px 20px; flex-direction: row; flex-wrap: wrap; align-items: center; gap: 18px; border-right: 0; border-bottom: 1px solid var(--line); }.admin-sidebar nav { display: flex; min-width: 0; margin: 0; overflow-x: auto; flex: 1; scrollbar-width: thin; }.admin-sidebar nav button { flex-shrink: 0; padding: 8px 12px; }.nav-icon { display: none; }.sidebar-bottom { display: none; margin: 0; padding: 0; border: 0; }.sidebar-garden, .admin-profile, .bottom-actions .quiet-button { display: none; } }
@media (max-width: 650px) { .admin-sidebar nav { order: 3; flex-basis: 100%; }.sidebar-bottom { margin-left: auto; }.admin-brand { padding: 0; } }
.admin-sidebar::before { content: ''; position: absolute; inset: 0; z-index: -1; background: url('/images/teacher-leaves.webp') 35% center / cover; opacity: .22; pointer-events: none; }
.botanical-caption { margin: 0; padding: 14px 8px; border-radius: 14px; color: #4d6048; background: #fffef9b8; font-size: 11px; line-height: 1.7; text-align: center; }
</style>
