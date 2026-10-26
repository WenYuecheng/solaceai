<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, useTemplateRef, watch } from 'vue'

const open = defineModel<boolean>('open', { default: false })
const mobile = ref(window.innerWidth <= 1020)
const dialog = useTemplateRef<HTMLDialogElement>('dialog')
function resize() { mobile.value = window.innerWidth <= 1020; if (!mobile.value) open.value = false }
watch([open, mobile], async () => {
  await nextTick()
  if (open.value && mobile.value && !dialog.value?.open) dialog.value?.showModal()
  else if (dialog.value?.open) dialog.value.close()
})
onMounted(() => window.addEventListener('resize', resize))
onBeforeUnmount(() => window.removeEventListener('resize', resize))
</script>

<template>
  <aside v-if="!mobile" id="student-navigation" class="student-aside" aria-label="会话与支持"><slot /></aside>
  <dialog v-else id="student-navigation" ref="dialog" class="student-aside mobile-sidebar" aria-label="会话与支持" @close="open = false" @click.self="open = false">
    <div class="sidebar-heading"><strong>会话与支持</strong><button class="icon-button" type="button" aria-label="关闭会话侧栏" @click="open = false">×</button></div>
    <slot />
  </dialog>
</template>

<style scoped>
.student-aside { grid-area: aside; display: flex; flex-direction: column; gap: 25px; min-width: 0; min-height: 0; overflow-y: auto; padding: 26px 18px 18px; border-right: 1px solid var(--line); background: #eef2df; position: relative; isolation: isolate; }
.mobile-sidebar { position: fixed; inset: 0 auto 0 0; width: min(300px, calc(100vw - 32px)); height: 100dvh; max-height: 100dvh; margin: 0; border: 0; border-right: 1px solid var(--line); color: var(--text); overscroll-behavior: contain; }
.mobile-sidebar:not([open]) { display: none; }.mobile-sidebar::backdrop { background: #1f181066; }.sidebar-heading { display: flex; align-items: center; justify-content: space-between; margin-bottom: 2px; }.sidebar-heading strong { font-size: 16px; font-weight: 500; }
.student-aside::before { content: ''; position: absolute; inset: 0; z-index: -1; background: url('/images/student-clouds.webp') 27% center / cover; opacity: .21; pointer-events: none; }
.mobile-sidebar { position: fixed; }
</style>
