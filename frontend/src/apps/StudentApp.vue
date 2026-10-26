<script setup lang="ts">
import ConfirmDialog from '@/components/shared/ConfirmDialog.vue'
import { computed, nextTick, onMounted, shallowRef, useTemplateRef, watch } from 'vue'
import { storeToRefs } from 'pinia'
import ChatComposer from '@/components/student/ChatComposer.vue'
import MessageList from '@/components/student/MessageList.vue'
import QuickCheckin from '@/components/student/QuickCheckin.vue'
import StudentHeader from '@/components/student/StudentHeader.vue'
import SessionHistory from '@/components/student/SessionHistory.vue'
import MemoryPanel from '@/components/student/MemoryPanel.vue'
import ResponsiveSidebar from '@/components/student/ResponsiveSidebar.vue'
import { useAuthStore } from '@/stores/auth'
import { useChatStore } from '@/stores/chat'
import { useSystemStore } from '@/stores/system'
import { useThemeStore } from '@/stores/theme'

const auth = useAuthStore(), chat = useChatStore(), system = useSystemStore(), theme = useThemeStore()
const { profile } = storeToRefs(auth)
const { messages, phase, error } = storeToRefs(chat)
const draft = shallowRef('')
const quickOpen = shallowRef(false)
const sidebarOpen = shallowRef(false)
const resetOpen = shallowRef(false)
const historyRevision = shallowRef(0)
const composer = useTemplateRef<InstanceType<typeof ChatComposer>>('composer')
const quickDialog = useTemplateRef<HTMLDialogElement>('quickDialog')
const displayName = computed(() => profile.value?.displayName === 'Demo Student' ? 'Student' : profile.value?.displayName || profile.value?.username || 'Student')
const greeting = computed(() => {
  const hour = new Date().getHours()
  return hour < 6 ? '夜深了' : hour < 11 ? '早上好' : hour < 14 ? '中午好' : hour < 18 ? '下午好' : '晚上好'
})

function choosePrompt(message: string) { draft.value = message; quickOpen.value = false; sidebarOpen.value = false; nextTick(() => composer.value?.focus()) }
function resetSession() { if (!messages.value.length) chat.reset(); else resetOpen.value = true }
function confirmReset() { resetOpen.value = false; chat.reset(); nextTick(() => composer.value?.focus()) }
function selectSession(id: string) { sidebarOpen.value = false; draft.value = ''; void chat.loadSession(id) }
function removedSession(id: string) {
  if (chat.sessionId === id || chat.pendingSessionId === id) { chat.invalidateSession(id); draft.value = '' }
}
watch(phase, (value) => { if (value === 'done') historyRevision.value++ })
async function logout() {
  try { await auth.logout(); window.location.assign('/') }
  catch { window.alert(auth.error || '退出失败，请稍后重试') }
}

watch(quickOpen, async (open) => {
  await nextTick()
  const dialog = quickDialog.value
  if (open && dialog && !dialog.open) dialog.showModal()
  if (!open && dialog?.open) dialog.close()
})

onMounted(async () => {
  theme.initialize()
  const nextProfile = await auth.resume()
  if (!nextProfile || nextProfile.requiresSetup) return window.location.replace('/')
  if (auth.admin) return window.location.replace('/admin.html')
  void system.load()
})
</script>

<template>
  <a class="skip-link" href="#chat-workspace">跳到对话</a>
  <div class="student-page">
    <StudentHeader :display-name="displayName" @logout="logout" @reset="resetSession" :sidebar-open="sidebarOpen" @open-sidebar="sidebarOpen = true" />
    <main class="student-layout">
      <ResponsiveSidebar v-model:open="sidebarOpen">
        <div class="aside-intro"><span class="library-mark" aria-hidden="true">❀</span><h2>我的心情手账</h2><p>把今天，轻轻放在这里。</p><button class="secondary-button aside-new" type="button" @click="resetSession(); sidebarOpen = false">＋ 开始新会话</button></div>
        <SessionHistory :active-session="chat.pendingSessionId || chat.sessionId" :revision="historyRevision" @select="selectSession" @removed="removedSession" @removing="removedSession" />
        <MemoryPanel />
        <p class="sidebar-footnote">每一种心情，都有被听见的位置。</p>
      </ResponsiveSidebar>

      <section id="chat-workspace" class="chat-workspace" aria-labelledby="chat-title">
        <header class="chat-heading">
          <div><h1 id="chat-title">{{ greeting }}，{{ displayName }} <span aria-hidden="true">☀</span></h1><p>来坐一会儿，说说你的今天。</p></div>
          <span class="session-state"><i aria-hidden="true"></i>{{ phase === 'thinking' ? '思考中' : phase === 'streaming' ? '回复中' : phase === 'error' ? '连接中断' : '我在这里' }}</span>
        </header>
        <div class="chat-paper">
          <MessageList :messages="messages" :phase="phase" :error="error" @quick="choosePrompt" @retry="chat.retry" />
          <ChatComposer ref="composer" v-model="draft" :phase="phase" @send="chat.send" @stop="chat.stop" @open-quick="quickOpen = true" />
        </div>
        <div class="trust-strip" aria-label="对话特点"><span>耐心倾听</span><span>尊重隐私</span><span>风险时建议线下支持</span></div>
      </section>

      <aside class="garden-corner" aria-label="小憩角落">
        <div class="corner-heading"><span aria-hidden="true">✿</span><h2>小憩角落</h2></div>
        <p class="corner-caption">不用急着变好，<br>先让自己歇一歇。</p>
        <QuickCheckin @select="choosePrompt" />
        <section class="support-note"><strong>给自己一点温柔</strong><p>对话用于心理支持，不替代专业诊断；如遇紧急危险，请立即联系身边可信任的人或当地紧急服务。</p></section>
      </aside>
    </main>

    <ConfirmDialog :open="resetOpen" title="开始一段新对话？" message="当前页面中的对话会被清空。你仍可以在历史会话里找到已保存的内容。" confirm-label="开始新会话" @confirm="confirmReset" @cancel="resetOpen = false" />
    <dialog ref="quickDialog" class="quick-dialog" aria-labelledby="quick-dialog-title" @click.self="quickOpen = false" @close="quickOpen = false">
      <div class="dialog-head"><h2 id="quick-dialog-title">选择一个起点</h2><button class="icon-button" type="button" aria-label="关闭快捷入口" @click="quickOpen = false">×</button></div>
      <QuickCheckin @select="choosePrompt" />
    </dialog>
  </div>
</template>

<style scoped>
.student-page { height: 100dvh; min-height: 480px; display: flex; flex-direction: column; background: #f6f5e9; color: #354937; }
.student-layout { flex: 1; min-height: 0; display: grid; grid-template-areas: 'aside chat corner'; grid-template-columns: 220px minmax(0, 1fr) 240px; }
.chat-workspace { grid-area: chat; min-width: 0; min-height: 0; display: grid; grid-template-rows: auto minmax(0, 1fr) auto; padding: 0 24px; }
.chat-heading { display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 23px 3px 20px; }.chat-heading h1 { margin: 0; font-family: var(--font-playful, var(--font-body)); font-size: clamp(23px, 2.1vw, 31px); font-weight: 400; }.chat-heading h1 span { color: #c3a24d; display: inline-block; margin-left: 5px; }.chat-heading p { margin: 9px 0 0; color: #5f7053; font-size: 12px; }.session-state { display: inline-flex; align-items: center; gap: 6px; color: #5f7053; font-size: 11px; white-space: nowrap; }.session-state i { width: 7px; height: 7px; border-radius: 50%; background: #89a877; box-shadow: 0 0 0 4px #e8eddb; }
.chat-paper { display: grid; grid-template-rows: minmax(0, 1fr) auto; min-height: 0; overflow: hidden; border: 1px solid #e1e5d4; border-radius: 25px; background: #fffef9; box-shadow: 0 7px 0 #e9ecdba6; }
.trust-strip { display: flex; justify-content: center; gap: 20px; padding: 14px 5px; color: #5f7053; font-size: 10px; }.trust-strip span::before { content: '✧'; margin-right: 5px; color: #8a9b75; }
.aside-intro h2 { margin: 11px 0 8px; font-family: var(--font-playful, var(--font-body)); font-size: 22px; font-weight: 400; }.aside-intro p { margin: 0 0 22px; color: #5f7053; font-size: 11px; }.library-mark { display: grid; place-items: center; width: 35px; height: 35px; color: #648459; background: #e0e9c9; border-radius: 12px 12px 12px 3px; font-size: 27px; }.aside-new { width: 100%; min-height: 42px; border-color: #4e6750; border-radius: 13px; color: #fffef6; background: #4e6750; font-size: 12px; }.sidebar-footnote { margin: -15px 0 0; text-align: center; color: #5f7053; font-size: 10px; line-height: 1.7; }
.garden-corner { grid-area: corner; min-height: 0; overflow-y: auto; padding: 26px 20px 20px 0; }.corner-heading { display: flex; align-items: center; gap: 8px; }.corner-heading > span { color: #bd9a54; font-size: 24px; }.corner-heading h2 { margin: 0; font-family: var(--font-playful, var(--font-body)); font-size: 23px; font-weight: 400; }.corner-caption { margin: 12px 0 25px; color: #5f7053; text-align: center; font-size: 12px; line-height: 1.8; }.support-note { margin-top: 26px; padding: 16px 14px; border: 1px dashed #d4d9bf; border-radius: 4px 18px 18px 18px; background: #f5efcd; color: #5f7053; }.support-note strong { font-family: var(--font-playful, var(--font-body)); font-size: 15px; font-weight: 400; }.support-note p { margin: 8px 0 0; font-size: 10px; line-height: 1.8; }
.quick-dialog { width: min(92vw, 480px); max-height: 85dvh; padding: 24px; overflow: auto; border: 1px solid #dbe0cc; border-radius: 24px; color: var(--text); background: #fffef7; }.quick-dialog::backdrop { background: #2d432c66; }.dialog-head { display: flex; align-items: center; justify-content: space-between; padding-bottom: 20px; }.dialog-head h2 { margin: 0; font-family: var(--font-playful, var(--font-body)); font-size: 23px; font-weight: 400; }
@media (max-width: 1200px) { .student-layout { grid-template-areas: 'aside chat'; grid-template-columns: 220px minmax(0, 1fr); }.garden-corner { display: none; } }
@media (max-width: 1020px) { .student-layout { grid-template-areas: 'chat'; grid-template-columns: minmax(0, 1fr); } }
@media (max-width: 720px) { .student-page { min-height: 0; }.chat-workspace { padding: 0 10px; }.chat-heading { padding: 16px 7px; }.chat-heading h1 { font-size: 23px; }.chat-heading p { margin-top: 6px; font-size: 11px; }.session-state { font-size: 10px; }.chat-paper { border-radius: 20px; }.trust-strip { gap: 12px; padding: 10px 0; font-size: 9px; }.quick-dialog { width: calc(100vw - 24px); } }
@media (max-height: 600px) { .chat-heading { padding-top: 10px; padding-bottom: 10px; }.chat-heading p, .trust-strip { display: none; }.chat-workspace { grid-template-rows: auto minmax(0, 1fr); padding-bottom: 8px; } }

.chat-paper { position: relative; isolation: isolate; background: #f4f4e8; }
.chat-paper::before { content: ''; position: absolute; z-index: -1; inset: 0; background: url('/images/student-meadow.webp') center / cover; opacity: .28; pointer-events: none; }
.chat-paper > * { position: relative; }
.garden-corner { position: relative; isolation: isolate; padding: 26px 18px 20px; border-radius: 26px 0 0 26px; }
.garden-corner::before { content: ''; position: absolute; z-index: -1; inset: 0; background: url('/images/student-reading.webp') 48% center / cover; opacity: .17; pointer-events: none; }
.corner-caption { margin: 28px 0; padding: 18px 10px; background: #fffef9c9; border-radius: 24px 6px 24px 6px; }
.sidebar-footnote { margin-top: auto; padding-top: 20px; }
.support-note { background: #fbf5dceb; }
@media (max-width: 1200px) { .garden-corner { display: none; } }
</style>
