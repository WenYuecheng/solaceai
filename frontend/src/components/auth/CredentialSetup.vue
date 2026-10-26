<script setup lang="ts">
import { reactive, shallowRef } from 'vue'
import { useAuthStore } from '@/stores/auth'
import { ApiError, requestJson } from '@/core/api'
import type { Profile } from '@/types/api'
const emit = defineEmits<{ authenticated: [profile: Profile] }>()
const auth = useAuthStore()
const fields = reactive({ username: '', password: '', confirmPassword: '' })
const busy = shallowRef(false), error = shallowRef('')
async function submit() {
  if (busy.value) return
  error.value = ''
  if (fields.password !== fields.confirmPassword) { error.value = '两次输入的密码不一致'; return }
  busy.value = true
  try {
    const profile = await requestJson<Profile>('/api/auth/setup', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(fields) })
    fields.password = ''; fields.confirmPassword = ''
    auth.acceptSession('', profile)
    emit('authenticated', profile)
  } catch (reason) { error.value = reason instanceof ApiError ? reason.message : '网络连接异常，请稍后重试' }
  finally { busy.value = false }
}
async function leave() { if (!busy.value) { await auth.logout().catch(() => { error.value = '退出失败，请稍后重试' }) } }
</script>

<template>
  <section class="credential-setup">
    <h2 id="login-title">设置你的账号</h2>
    <p class="intro">邮箱已验证。再设置一个熟悉的用户名和密码，就可以开始了。</p>
    <form id="login-form" class="account-form" @submit.prevent="submit">
      <label>用户名<input v-model="fields.username" name="username" autocomplete="username" minlength="3" maxlength="64" pattern="[A-Za-z0-9_\-]+" title="3–64 位字母、数字、下划线或短横线" required :disabled="busy" placeholder="3–64 位字母、数字、下划线或短横线"></label>
      <label>密码<input v-model="fields.password" name="password" type="password" autocomplete="new-password" minlength="8" maxlength="128" required :disabled="busy" placeholder="至少 8 位密码"></label>
      <label>确认密码<input v-model="fields.confirmPassword" name="confirmPassword" type="password" autocomplete="new-password" minlength="8" maxlength="128" required :disabled="busy" placeholder="再次输入密码"></label>
      <p v-if="error" role="alert" class="form-error">{{ error }}</p>
      <button class="primary-button" type="submit" :disabled="busy">{{ busy ? '正在保存…' : '完成设置并进入' }}</button>
      <button class="quiet-button" type="button" :disabled="busy" @click="leave">退出并返回登录</button>
    </form>
  </section>
</template>

<style scoped>
h2 { margin: 0; font: 500 clamp(25px, 2.3vw, 32px) var(--font-display); }.intro { margin: 12px 0 22px; font-size: 12px; color: var(--text-soft); line-height: 1.7; }
</style>
<style src="./account-form.css"></style>
