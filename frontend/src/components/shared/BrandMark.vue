<script setup lang="ts">
withDefaults(defineProps<{ compact?: boolean; context?: string }>(), { compact: false, context: '校园心理支持' })
</script>

<template>
  <div class="brand-lockup" :class="{ compact }">
    <span class="brand-symbol" aria-hidden="true"><b>SA</b></span>
    <span class="brand-copy">
      <small>{{ context }}</small>
      <strong>SolaceAI</strong>
    </span>
  </div>
</template>

<style scoped>
.brand-lockup { display: inline-flex; align-items: center; gap: 13px; min-width: 0; }
.brand-symbol { position: relative; width: 48px; height: 48px; flex: 0 0 48px; display: grid; place-items: center; overflow: hidden; border-radius: 50%; color: white; background: #4e6750; box-shadow: none; }
.brand-symbol i { position: absolute; width: 36px; height: 36px; border-radius: 50%; background: rgba(127, 224, 226, .28); filter: blur(8px); transform: translate(11px, -10px); }
.brand-symbol b { position: relative; font-family: var(--font-display); letter-spacing: -.04em; }
.brand-copy { min-width: 0; display: grid; }
.brand-copy small { overflow: hidden; color: var(--text-faint); font-family: var(--font-body); font-size: 11px; font-weight: 700; letter-spacing: .02em; text-overflow: ellipsis; white-space: nowrap; }
.brand-copy strong { font-family: var(--font-display); font-size: 21px; letter-spacing: -.035em; }
.compact .brand-symbol { width: 42px; height: 42px; flex-basis: 42px; border-radius: 50%; }
.compact .brand-copy strong { font-size: 18px; }
@media (max-width: 520px) { .brand-copy small { font-size: 11px; } .brand-copy strong { font-size: 18px; } }
</style>
