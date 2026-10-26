<script setup lang="ts">
import { nextTick, onBeforeUnmount, ref, useId, useTemplateRef, watch } from 'vue'

const props = defineProps<{ open: boolean; title: string; message: string; confirmLabel: string }>()
const emit = defineEmits<{ confirm: []; cancel: [] }>()
const dialog = useTemplateRef<HTMLDialogElement>('dialog')
const titleId = useId(), descriptionId = useId()
const settled = ref(false)
function finish(confirmed: boolean) {
  if (settled.value) return
  settled.value = true
  dialog.value?.close?.()
  if (confirmed) emit('confirm')
  else emit('cancel')
}
watch(() => props.open, async open => {
  await nextTick()
  if (open) {
    settled.value = false
    if (dialog.value && !dialog.value.open) dialog.value.showModal?.()
  } else if (dialog.value?.open) {
    settled.value = true
    dialog.value.close?.()
  }
}, { immediate: true })
onBeforeUnmount(() => { settled.value = true; if (dialog.value?.open) dialog.value.close?.() })
</script>

<template>
  <dialog ref="dialog" class="confirm-dialog" :aria-labelledby="titleId" :aria-describedby="descriptionId" @cancel.prevent="finish(false)" @close="finish(false)" @click.self="finish(false)">
    <div class="confirm-decoration" aria-hidden="true">❀</div>
    <h2 :id="titleId">{{ title }}</h2>
    <p :id="descriptionId">{{ message }}</p>
    <div class="confirm-actions"><button type="button" class="secondary-button" autofocus @click="finish(false)">取消</button><button type="button" class="primary-button" data-confirm @click="finish(true)">{{ confirmLabel }}</button></div>
  </dialog>
</template>

<style scoped>
.confirm-dialog { width: min(440px, calc(100vw - 40px)); max-height: calc(100dvh - 48px); overflow-y: auto; overscroll-behavior: contain; padding: 32px; border: 1px solid #c6d3b9; border-radius: 28px 28px 12px 28px; color: #344b35; background: #f8faee; box-shadow: 0 20px 65px #263c2933; isolation: isolate; }
.confirm-dialog::before { content: ''; position: absolute; inset: 0; z-index: -1; background: url('/images/student-clouds.webp') center / cover; opacity: .14; pointer-events: none; }
.confirm-dialog::backdrop { background: #30452d66; backdrop-filter: blur(3px); }
.confirm-decoration { width: 46px; height: 46px; display: grid; place-items: center; margin-bottom: 18px; border-radius: 16px 16px 5px 16px; color: #597c48; background: #e2ebce; font-size: 32px; }
h2 { margin: 0; font: 400 27px/1.4 var(--font-playful); }
p { margin: 16px 0 28px; color: #4e6049; font-size: 14px; line-height: 1.9; overflow-wrap: anywhere; }
.confirm-actions { display: flex; justify-content: flex-end; gap: 12px; }.confirm-actions button { min-height: 44px; padding-inline: 22px; border-radius: 14px; font-family: var(--font-playful); font-size: 17px; font-weight: 400; }.secondary-button { background: #fffef9df; }
@media (max-width: 520px) { .confirm-dialog { padding: 26px; }.confirm-actions button { flex: 1; padding-inline: 12px; } }
</style>
