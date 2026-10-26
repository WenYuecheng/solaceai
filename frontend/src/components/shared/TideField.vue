<script setup lang="ts">
withDefaults(defineProps<{ state?: 'idle' | 'input' | 'thinking' | 'reply'; intensity?: 'soft' | 'vivid' }>(), {
  state: 'idle', intensity: 'soft',
})
</script>

<template>
  <div class="tide-field" :class="[`state-${state}`, `intensity-${intensity}`]" aria-hidden="true">
    <span class="orb orb-a"></span><span class="orb orb-b"></span><span class="orb orb-c"></span>
    <span class="tide-grid"></span><span class="tide-noise"></span>
  </div>
</template>

<style scoped>
.tide-field { position: absolute; inset: 0; overflow: hidden; pointer-events: none; contain: paint; }
.orb { position: absolute; border-radius: 50%; filter: blur(70px); will-change: transform, opacity; }
.orb-a { width: 46%; aspect-ratio: 1; top: -12%; left: -8%; background: color-mix(in srgb, var(--aqua) 50%, transparent); animation: drift-a 16s ease-in-out infinite alternate; }
.orb-b { width: 48%; aspect-ratio: 1; right: -12%; bottom: -20%; background: color-mix(in srgb, var(--violet) 42%, transparent); animation: drift-b 19s ease-in-out infinite alternate; }
.orb-c { width: 32%; aspect-ratio: 1; top: 30%; left: 42%; background: color-mix(in srgb, var(--lagoon) 38%, transparent); animation: drift-c 13s ease-in-out infinite alternate; }
.tide-grid { position: absolute; inset: 0; opacity: .16; background-image: linear-gradient(rgba(104, 176, 188, .16) 1px, transparent 1px), linear-gradient(90deg, rgba(104, 176, 188, .16) 1px, transparent 1px); background-size: 44px 44px; mask-image: radial-gradient(circle at 50% 45%, black, transparent 78%); }
.tide-noise { position: absolute; inset: -20%; opacity: .08; background-image: repeating-radial-gradient(circle at 20% 30%, currentColor 0 1px, transparent 1px 7px); transform: rotate(8deg); }
.intensity-vivid .orb { filter: blur(58px); opacity: .92; }
.state-input .orb-c { animation-duration: 7s; }
.state-thinking .orb { animation-duration: 6s; }
.state-reply .orb-a { animation-duration: 9s; }
@keyframes drift-a { to { transform: translate3d(22%, 18%, 0) scale(1.1); } }
@keyframes drift-b { to { transform: translate3d(-18%, -12%, 0) scale(.92); } }
@keyframes drift-c { to { transform: translate3d(-14%, 20%, 0) scale(1.16); } }
@media (prefers-reduced-motion: reduce) { .orb { animation: none; } }
</style>
