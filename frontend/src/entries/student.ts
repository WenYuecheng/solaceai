import 'element-plus/dist/index.css'
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import StudentApp from '@/apps/StudentApp.vue'
import '@/styles/base.css'

createApp(StudentApp).use(createPinia()).mount('#app')
