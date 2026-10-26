<script setup lang="ts">
import CredentialSetup from './CredentialSetup.vue'
import { computed, onUnmounted, reactive, shallowRef } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { ApiError, requestJson } from '@/core/api'
import type { Profile } from '@/types/api'

const emit = defineEmits<{ authenticated: [profile: Profile] }>()
const auth = useAuthStore()
const mode = shallowRef<'password' | 'email'>('email')
const fields = reactive({ username: '', email: '', password: '', code: '' })
const formError = shallowRef(''), notice = shallowRef(''), sending = shallowRef(false), remaining = shallowRef(0)
const busy = computed(() => auth.loading || sending.value)
const title = computed(() => mode.value === 'email' ? '邮箱登录 / 注册' : '欢迎回到 SolaceAI')
let timer: ReturnType<typeof setInterval> | undefined
onUnmounted(() => clearInterval(timer))
function switchMode(next: typeof mode.value) {
  mode.value = next
  formError.value = ''; notice.value = ''; auth.error = ''
  fields.password = ''; fields.code = ''
}
async function sendCode() {
  if (sending.value || remaining.value > 0) return
  formError.value = ''; notice.value = ''
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(fields.email.trim())) { formError.value = '请先填写有效的邮箱地址'; return }
  sending.value = true
  try {
    await requestJson('/api/auth/code', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email: fields.email.trim(), purpose: 'login' }) })
    notice.value = '验证码请求已提交，5 分钟内有效。新邮箱验证成功后需设置用户名和密码。'
    remaining.value = 60
    clearInterval(timer)
    timer = setInterval(() => { remaining.value--; if (remaining.value <= 0) clearInterval(timer) }, 1000)
  } catch (reason) {
    formError.value = reason instanceof ApiError ? reason.message : '网络连接异常，请稍后重试'
  } finally { sending.value = false }
}
async function submit() {
  if (busy.value) return
  formError.value = ''
  try {
    const profile = mode.value === 'password' ? await auth.login(fields.username, fields.password)
      : await auth.emailLogin(fields.email, fields.code)
    fields.password = ''; fields.code = ''
    if (!profile.requiresSetup) emit('authenticated', profile)
  } catch { /* auth store 内部已统一捕获并设置安全的用户端提示文案 */ }
}
</script>

<template>
  <CredentialSetup v-if="auth.profile?.requiresSetup" @authenticated="emit('authenticated', $event)" />
  <div v-else class="access-form">
    <h2 id="login-title">{{ title }}</h2>
    <p class="intro">{{ mode === 'email' ? '未注册邮箱验证成功后将自动创建学生账号，再设置用户名和密码。心理教师请使用密码登录。' : '使用用户名或已绑定邮箱和密码登录。' }}</p>
    <div class="mode-switch" role="group" aria-label="选择登录方式">
      <button type="button" data-mode="email" :aria-pressed="mode === 'email'" :disabled="busy" @click="switchMode('email')">邮箱登录 / 注册</button>
      <button type="button" data-mode="password" :aria-pressed="mode === 'password'" :disabled="busy" @click="switchMode('password')">用户名 / 邮箱密码</button>
    </div>
    <form id="login-form" class="form" @submit.prevent="submit">
      <template v-if="mode === 'password'">
        <div class="field">
          <label for="username">用户名或邮箱</label>
          <input id="username" v-model="fields.username" name="username" autocomplete="username" required maxlength="254" spellcheck="false" placeholder="输入用户名或已绑定邮箱" :disabled="busy">
        </div>
        <div class="field">
          <label for="password">密码</label>
          <input id="password" v-model="fields.password" name="password" type="password" autocomplete="current-password" maxlength="128" required placeholder="输入密码…" :disabled="busy">
        </div>
      </template>
      <template v-else>
        <div class="field">
          <label for="email">邮箱</label>
          <input id="email" v-model="fields.email" name="email" type="email" autocomplete="email" required maxlength="254" placeholder="输入你的邮箱地址" :disabled="busy">
        </div>
        <div class="field">
          <label for="code">邮箱验证码</label>
          <div class="code-row">
            <input id="code" v-model="fields.code" name="code" autocomplete="one-time-code" inputmode="numeric" pattern="[0-9]{6}" minlength="6" maxlength="6" required placeholder="6 位验证码" :disabled="busy">
            <button type="button" class="secondary-button" :disabled="busy || remaining > 0" @click="sendCode">{{ sending ? '正在发送…' : remaining > 0 ? `${remaining} 秒后重发` : '发送验证码' }}</button>
          </div>
        </div>
      </template>
      <p v-if="formError || auth.error" class="form-error" role="alert">{{ formError || auth.error }}</p>
      <p v-if="notice" class="notice" role="status">{{ notice }}</p>
      <button class="primary-button submit" type="submit" :disabled="busy">{{ auth.loading ? '正在验证…' : '安全登录' }}</button>
    </form>
  </div>
</template>

<style scoped>
h2 { margin: 0; font-family: var(--font-display); font-size: clamp(25px, 2.3vw, 32px); font-weight: 500; letter-spacing: -.025em; }
.intro { margin: 12px 0 22px; font-size: 12px; color: var(--text-soft); line-height: 1.7; }
.mode-switch { display: flex; padding: 0; gap: 8px; margin-bottom: 24px; padding: 4px; border: 0; border-radius: 12px; background: #eaf0e3; }
.mode-switch button { flex: 1; padding: 10px 4px; border: 0; border-radius: 9px; background: transparent; color: var(--text-soft); font-size: 13px; cursor: pointer; }
.mode-switch button[aria-pressed="true"] { color: #fffef9; background: #4e6750; box-shadow: none; font-weight: 700; }
.form { display: grid; gap: 16px; }
.field { display: grid; gap: 8px; }
.field label { font-weight: 500; font-size: 13px; }
.field input { width: 100%; min-width: 0; min-height: 50px; padding: 0 14px; border: 1px solid var(--line); border-radius: var(--radius-sm); color: var(--text); background: #ffffffb8; }
.field input:focus { border-color: var(--aqua); outline: 2px solid color-mix(in srgb, var(--aqua) 30%, transparent); }
.code-row { display: flex; gap: 10px; }
.code-row input { flex: 1; }
.code-row button { white-space: nowrap; font-size: 12px; padding: 0 12px; }
.form-error { margin: 0; color: var(--danger); line-height: 1.6; }
.notice { margin: 0; color: var(--text-soft); font-size: 12px; line-height: 1.7; }
.submit { display: flex; align-items: center; justify-content: center; width: 100%; padding: 0 18px; margin-top: 6px; }
button:disabled { opacity: .6; cursor: wait; }
@media (max-width: 380px) { .code-row { flex-wrap: wrap; }.code-row button { min-height: 40px; } }
input:autofill { box-shadow: 0 0 0 100px #f4f7ec inset; -webkit-text-fill-color: #344137; }
</style>
