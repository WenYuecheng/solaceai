<script setup lang="ts">
import { useId } from 'vue'
const titleId = useId()
const emit = defineEmits<{ select: [message: string] }>()
const prompts = [
  { icon: '☾', title: '压力与睡眠', note: '总是紧绷或睡不好', message: '我最近压力很大，晚上总是睡不着。' },
  { icon: '☁', title: '焦虑与考试', note: '心里乱，难以静下来', message: '我感觉很焦虑，考试前完全静不下来。' },
  { icon: '♡', title: '低落与孤独', note: '没力气，也不想说话', message: '我最近情绪很低落，不太想和别人说话。' },
  { icon: '♧', title: '关系与困扰', note: '家人、朋友或亲密关系', message: '我想聊聊最近和家人或朋友的矛盾。' },
]
</script>

<template>
  <section class="quick-panel" :aria-labelledby="titleId">
    <div class="quick-head"><div><h2 :id="titleId">最近感觉怎么样？</h2></div></div>
    <div class="quick-grid">
      <button v-for="prompt in prompts" :key="prompt.title" class="quick-card" type="button" @click="emit('select', prompt.message)">
        <span class="quick-icon" aria-hidden="true">{{ prompt.icon }}</span>
        <span><strong>{{ prompt.title }}</strong><small>{{ prompt.note }}</small></span>
      </button>
    </div>
  </section>
</template>

<style scoped>
.quick-panel { padding: 0; }.quick-head h2 { margin: 0; color: #52684b; font-family: var(--font-playful, var(--font-body)); font-size: 18px; font-weight: 400; }.quick-grid { display: grid; gap: 9px; margin-top: 14px; }.quick-card { min-height: 60px; display: flex; align-items: center; gap: 10px; padding: 10px; border: 1px solid #e2e5d0; border-radius: 15px; color: #4b5f47; background: #fffef5; text-align: left; transition: background-color var(--duration-fast); }.quick-card:hover { background: #e9efdc; border-color: #a6b88e; }.quick-icon { width: 33px; height: 36px; flex: 0 0 33px; display: grid; place-items: center; border-radius: 11px 11px 5px 11px; color: #6e875c; background: #eaf0d7; font-size: 23px; }.quick-card:nth-child(2) .quick-icon { color: #6d8b96; background: #e6eff0; }.quick-card:nth-child(3) .quick-icon { color: #ab8578; background: #f6e8de; }.quick-card:nth-child(4) .quick-icon { color: #a18f53; background: #f6efce; }.quick-card strong, .quick-card small { display: block; }.quick-card strong { font-size: 12px; font-weight: 600; }.quick-card small { margin-top: 5px; color: #5f7053; font-size: 10px; line-height: 1.4; }
</style>
