import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30_000,
  use: { baseURL: 'http://127.0.0.1:4174', trace: 'retain-on-failure' },
  webServer: {
    command: 'python3 -m http.server 4174 --directory ../packages/core/app/static',
    url: 'http://127.0.0.1:4174',
    reuseExistingServer: true,
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'mobile', use: { ...devices['iPhone 13'], browserName: 'chromium' } },
  ],
})
