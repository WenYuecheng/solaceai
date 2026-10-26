<script setup lang="ts">
import { onMounted } from 'vue'
import BrandMark from '@/components/shared/BrandMark.vue'
import SystemStatus from '@/components/shared/SystemStatus.vue'
import AccessForm from '@/components/auth/AccessForm.vue'
import { useAuthStore } from '@/stores/auth'
import { useSystemStore } from '@/stores/system'
import { useThemeStore } from '@/stores/theme'
import { isAdmin } from '@/core/auth'
import type { Profile } from '@/types/api'

const auth = useAuthStore(), system = useSystemStore(), theme = useThemeStore()
function routeProfile(profile: Profile | null) {
  if (profile && !profile.requiresSetup) window.location.assign(isAdmin(profile) ? '/admin.html' : '/student.html')
}
onMounted(async () => {
  theme.initialize()
  void system.load(false)
  routeProfile(await auth.resume())
})
</script>

<template>
  <a class="skip-link" href="#login-form">跳到登录表单</a>
  <main class="login-page">
    <img class="login-garden" src="/images/login-garden.webp" alt="" width="1536" height="1024" fetchpriority="high" aria-hidden="true">
    <header class="login-top"><BrandMark context="校园心理支持" /><span>不赶时间，我们慢慢聊。</span></header>
    <section class="login-story" aria-labelledby="story-title">
      <div class="story-content"><span class="paper-note">给今天的你，一点小小的留白</span><h1 id="story-title">把烦恼，<br>暂时放一放。</h1><p>关于压力、睡眠，或一件放不下的小事。<br>在这里，从一句“今天有点累”开始也很好。</p></div>
      <figure class="campus-postcard"><picture><source media="(max-width: 720px)" srcset="/images/campus-mobile.webp"><img src="/images/campus-corner.webp" alt="树荫下安静的校园长椅" width="1536" height="1024"></picture><figcaption>校园里，总有一个可以歇脚的角落。</figcaption></figure>
    </section>
    <section class="login-access" aria-labelledby="login-title">
      <div class="access-body"><div class="access-greeting"><span aria-hidden="true">✿</span><span>很高兴，在这里遇见你</span></div><AccessForm @authenticated="routeProfile" /><footer class="access-footer"><SystemStatus :show-model="false" /><p>对话用于心理支持，不替代专业诊断与治疗。</p></footer></div>
    </section>
    <footer class="login-bottom">SolaceAI · 给情绪一个可以停靠的地方</footer>
  </main>
</template>

<style scoped>
.login-page { position: relative; isolation: isolate; min-height: 100dvh; display: grid; grid-template-columns: minmax(0, 1fr) minmax(420px, .92fr); grid-template-rows: auto 1fr auto; gap: 28px 7vw; padding: 30px clamp(28px, 6vw, 100px) 22px; background: #e3e9d8; }
.login-garden { position: absolute; z-index: -2; inset: 0; width: 100%; height: 100%; object-fit: cover; object-position: center; }.login-page::before { content: ''; position: absolute; inset: 0; z-index: -1; background: linear-gradient(90deg, #f6f5eeda 5%, #f4f5e4ad 45%, #eef2e529); }
.login-top { grid-column: 1 / -1; display: flex; justify-content: space-between; align-items: center; gap: 20px; }.login-top > span { font-size: 13px; color: #3c513e; background: #fffef9e8; padding: 10px 18px; border-radius: 24px; }
.login-story { align-self: center; max-width: 620px; padding-block: 24px; }.paper-note { display: inline-block; padding: 8px 16px; color: #4c5f45; background: #f6eabf; border-radius: 4px 15px 5px 12px; font-size: 12px; transform: rotate(-2deg); }.story-content h1 { margin: 28px 0 20px; font-family: var(--font-playful); font-size: clamp(48px, 5.8vw, 82px); font-weight: 400; line-height: 1.18; letter-spacing: .015em; color: #344d39; }.story-content p { color: #445440; font-size: 15px; line-height: 1.95; margin: 0; }
.campus-postcard { width: min(82%, 380px); margin: 34px 0 0 15px; padding: 10px 10px 0; background: #fffef8; transform: rotate(-3deg); border-radius: 4px; box-shadow: 0 10px 30px #2e48201c; }.campus-postcard picture { display: block; }.campus-postcard img { display: block; width: 100%; height: 155px; object-fit: cover; object-position: center 65%; }.campus-postcard figcaption { padding: 12px 8px 14px; font-size: 11px; color: #606951; }
.login-access { min-width: 0; display: flex; align-items: center; justify-content: center; padding-block: 12px; }.access-body { width: min(100%, 490px); padding: 30px 34px 24px; border: 1px solid #ffffffe0; border-radius: 32px 32px 10px 32px; background: #fffef9ed; box-shadow: 0 15px 55px #35482b16; }.access-greeting { display: flex; align-items: center; gap: 10px; margin-bottom: 28px; color: #526b4c; font-size: 12px; }.access-greeting > span:first-child { font-size: 25px; }.access-footer { margin-top: 28px; padding-top: 16px; border-top: 1px dashed #cbd6c0; color: #59665b; font-size: 10px; }.access-footer p { line-height: 1.7; margin: 10px 0 0; }.login-bottom { grid-column: 1 / -1; text-align: center; color: #3d503a; font-size: 11px; }
@media (max-width: 1000px) { .login-page { gap: 22px; padding-inline: 28px; grid-template-columns: minmax(0, 1fr) minmax(370px, 1fr); }.story-content h1 { font-size: 55px; }.story-content p { font-size: 13px; }.campus-postcard { width: 90%; margin-left: 0; }.access-body { padding: 26px; } }
@media (max-width: 720px) { .login-page { display: flex; flex-direction: column; gap: 18px; padding: 22px 20px; }.login-garden { position: fixed; height: 100dvh; }.login-top > span { display: none; }.login-top :deep(.brand-symbol) { width: 35px; height: 35px; flex-basis: 35px; }.login-story { align-self: stretch; padding: 12px 8px 0; }.paper-note { font-size: 11px; padding: 6px 12px; }.story-content h1 { font-size: 42px; margin: 20px 0 12px; }.story-content h1 br { display: none; }.story-content p { font-size: 12px; }.campus-postcard { display: none; }.login-access { padding: 0; }.access-body { width: 100%; border-radius: 24px; padding: 24px; }.access-greeting { margin-bottom: 18px; }.login-bottom { margin-top: 4px; font-size: 10px; } }
</style>
