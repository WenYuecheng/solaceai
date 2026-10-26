import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8080',
      '/actuator': 'http://127.0.0.1:8080',
    },
  },
  build: {
    target: 'es2022',
    outDir: '../packages/core/app/static',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: (id) => id.includes('node_modules') ? 'vendor' : undefined,
      },
      input: {
        login: fileURLToPath(new URL('./index.html', import.meta.url)),
        student: fileURLToPath(new URL('./student.html', import.meta.url)),
        admin: fileURLToPath(new URL('./admin.html', import.meta.url)),
      },
    },
  },
})
