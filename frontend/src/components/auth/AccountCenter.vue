<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, shallowRef, useTemplateRef } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { ApiError, requestJson } from '@/core/api'
import type { Profile } from '@/types/api'
const emit = defineEmits<{ close: [] }>()
const auth = useAuthStore()
const dialog = useTemplateRef<HTMLDialogElement>('dialog')
const mode = shallowRef<'password' | 'email'>('password')
const fields = reactive({ currentPassword: '', password: '', confirmPassword: '', email: '', code: '' })
const busy = shallowRef(false), error = shallowRef(''), notice = shallowRef(''), remaining = shallowRef(0)
const bound = computed(() => !!auth.profile?.email)
let timer: ReturnType<typeof setInterval> | undefined
let previousFocus: HTMLElement | null = null
onMounted(() => { previousFocus = document.activeElement as HTMLElement; dialog.value?.showModal?.() })
onBeforeUnmount(() => { clearInterval(timer); dialog.value?.close?.(); previousFocus?.focus(); clearFields() })
function clearFields() { fields.currentPassword = ''; fields.password = ''; fields.confirmPassword = ''; fields.email = ''; fields.code = '' }
function switchMode(next: typeof mode.value) { mode.value = next; clearFields(); error.value = ''; notice.value = '' }
function close() { if (!busy.value) emit('close') }
async function post<T>(path: string, body: unknown) { return requestJson<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }) }
async function perform(action: () => Promise<void>) {
  if (busy.value) return
  busy.value = true; error.value = ''; notice.value = ''
  try { await action() }
  catch (reason) { error.value = reason instanceof ApiError ? reason.message : '网络连接异常，请稍后重试' }
  finally { busy.value = false }
}
async function sendCode() {
  if (remaining.value > 0) return
  if (!fields.currentPassword || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(fields.email.trim())) { error.value = '请填写当前密码和有效邮箱'; return }
  await perform(async () => {
    await post('/api/auth/email/code', { email: fields.email.trim(), currentPassword: fields.currentPassword })
    remaining.value = 60
    clearInterval(timer)
    timer = setInterval(() => { remaining.value--; if (remaining.value <= 0) clearInterval(timer) }, 1000)
    notice.value = '验证码已发送，5 分钟内有效。'
  })
}
async function submit() {
  if (mode.value === 'password' && fields.password !== fields.confirmPassword) { error.value = '两次输入的密码不一致'; return }
  await perform(async () => {
    const body = mode.value === 'password'
      ? { currentPassword: fields.currentPassword, password: fields.password, confirmPassword: fields.confirmPassword }
      : { currentPassword: fields.currentPassword, email: fields.email.trim(), code: fields.code }
    const profile = await post<Profile>(mode.value === 'password' ? '/api/auth/password' : '/api/auth/email', body)
    auth.acceptSession('', profile)
    clearFields()
    notice.value = mode.value === 'password' ? '密码已修改，其他设备需要重新登录。' : '邮箱已绑定，现在可以用邮箱和密码登录。'
  })
}
</script>

<template>
  <dialog ref="dialog" class="account-dialog" aria-labelledby="account-title" @cancel.prevent="close" @click.self="close">
    <header><div><span class="decoration" aria-hidden="true">❀</span><h2 id="account-title">个人中心</h2></div><button class="quiet-button" type="button" aria-label="关闭个人中心" :disabled="busy" @click="close">关闭</button></header>
    <dl class="account-details"><div><dt>用户名</dt><dd>{{ auth.profile?.username }}</dd></div><div><dt>邮箱</dt><dd>{{ auth.profile?.email || '尚未绑定' }}</dd></div></dl>
    <div class="account-tabs" role="group" aria-label="账号设置"><button type="button" :aria-pressed="mode === 'password'" :disabled="busy" @click="switchMode('password')">修改密码</button><button v-if="!bound" type="button" :aria-pressed="mode === 'email'" :disabled="busy" @click="switchMode('email')">绑定邮箱</button></div>
    <form v-if="mode === 'password' || !bound" class="account-form" @submit.prevent="submit">
      <label>当前密码<input v-model="fields.currentPassword" name="currentPassword" type="password" autocomplete="current-password" maxlength="128" required :disabled="busy" placeholder="输入当前账号密码"></label>
      <template v-if="mode === 'password'">
        <label>新密码<input v-model="fields.password" name="password" type="password" autocomplete="new-password" minlength="8" maxlength="128" required :disabled="busy" placeholder="至少 8 位密码"></label>
        <label>确认新密码<input v-model="fields.confirmPassword" name="confirmPassword" type="password" autocomplete="new-password" minlength="8" maxlength="128" required :disabled="busy" placeholder="再次输入新密码"></label>
      </template>
      <template v-else>
        <label>邮箱<input v-model="fields.email" name="email" type="email" autocomplete="email" maxlength="254" required :disabled="busy" placeholder="输入要绑定的邮箱"></label>
        <label>邮箱验证码<span class="code-row"><input v-model="fields.code" name="code" autocomplete="one-time-code" inputmode="numeric" pattern="[0-9]{6}" maxlength="6" required :disabled="busy" placeholder="6 位验证码"><button type="button" class="secondary-button" :disabled="busy || remaining > 0" @click="sendCode">{{ remaining > 0 ? `${remaining} 秒后重发` : '发送验证码' }}</button></span></label>
      </template>
      <p v-if="error" class="form-error" role="alert">{{ error }}</p>
      <p v-if="notice" class="notice" role="status">{{ notice }}</p>
      <button type="submit" class="primary-button" :disabled="busy">{{ busy ? '正在处理…' : mode === 'password' ? '保存新密码' : '验证并绑定' }}</button>
    </form>
    <p v-else-if="notice" class="success" role="status">{{ notice }}</p>
  </dialog>
</template>

<style scoped>
.account-dialog { width: min(460px, calc(100vw - 32px)); max-height: calc(100dvh - 40px); overflow-y: auto; overscroll-behavior: contain; padding: 28px; border: 1px solid var(--line); border-radius: 28px 28px 12px 28px; color: var(--text); background: var(--surface, #f8faee); box-shadow: 0 20px 65px #263c2933; }
.account-dialog::backdrop { background: #30452d66; backdrop-filter: blur(3px); }header, header > div { display: flex; align-items: center; gap: 12px; }header { justify-content: space-between; }h2 { margin: 0; font: 400 27px/1.4 var(--font-playful); }.decoration { color: #597c48; font-size: 30px; }.account-details { margin: 22px 0; padding: 16px; border: 1px dashed var(--line); border-radius: 14px; display: grid; gap: 10px; font-size: 13px; }.account-details div { display: grid; grid-template-columns: 54px minmax(0, 1fr); gap: 12px; }dt { color: var(--text-soft); }dd { margin: 0; overflow-wrap: anywhere; }.account-tabs { display: flex; gap: 8px; margin-bottom: 20px; padding: 4px; border-radius: 12px; background: color-mix(in srgb, var(--aqua) 12%, var(--surface)); }.account-tabs button { flex: 1; border: 0; border-radius: 9px; padding: 10px; color: var(--text-soft); background: transparent; cursor: pointer; }.account-tabs button[aria-pressed=true] { background: #4e6750; color: #fffef9; }.success { color: var(--text-soft); font-size: 13px; line-height: 1.8; }
@media (max-width: 520px) { .account-dialog { padding: 22px; } }
</style>
<style src="./account-form.css"></style>
