import { createApp } from 'vue'
import { createPinia } from 'pinia'
import {
  ElDrawer,
  ElEmpty,
  ElInput,
  ElOption,
  ElPagination,
  ElSegmented,
  ElSelect,
  ElTable,
  ElTableColumn,
  ElTag,
  ElUpload,
} from 'element-plus'
import AdminApp from '@/apps/AdminApp.vue'
import 'element-plus/dist/index.css'
import '@/styles/base.css'
import '@/styles/admin.css'

const app = createApp(AdminApp).use(createPinia())
;[ElDrawer, ElEmpty, ElInput, ElOption, ElPagination, ElSegmented, ElSelect, ElTable, ElTableColumn, ElTag, ElUpload]
  .forEach((component) => app.component(component.name!, component))
app.mount('#app')
