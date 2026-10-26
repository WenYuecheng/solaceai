import { createApp } from 'vue'
import { createPinia } from 'pinia'
import LoginApp from '@/apps/LoginApp.vue'
import '@/styles/base.css'

createApp(LoginApp).use(createPinia()).mount('#app')
