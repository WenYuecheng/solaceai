import { expect, test } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

test('model switch stays aligned and stationary at every header breakpoint', async ({ page }, info) => {
  await page.route('**/api/**', route => route.fulfill({ json: [] }))
  await page.route('**/api/profile', route => route.fulfill({ json: { id: 1, username: 'student', displayName: '林同学', roles: [{ authority: 'ROLE_USER' }] } }))
  await page.route('**/actuator/health', route => route.fulfill({ json: { status: 'UP' } }))
  await page.route('**/api/agent/status', route => route.fulfill({ json: { provider: 'openai', model: 'deepseek-flash', realModelEnabled: true, availableChatBackends: ['cloud','local'], defaultChatBackend: 'cloud', chatModels: { cloud: { provider: 'openai', model: 'deepseek-flash' }, local: { provider: 'vllm-metal', model: 'solace-response', label: 'Qwen3-4B · 回答 LoRA' } } } }))
  await page.goto('/student.html')
  await page.evaluate(() => document.fonts.ready)
  const group = page.getByRole('group', { name: '对话模型', exact: true })
  for (const width of [1440, 1024, 900, 880, 768, 720, 620, 390, 320]) {
    await page.setViewportSize({ width, height: 900 })
    await group.getByRole('button', { name: 'DeepSeek Flash', exact: true }).click()
    const before = await group.boundingBox()
    const header = await page.locator('.student-header').boundingBox()
    const account = await page.getByRole('button', { name: '个人中心', exact: true }).boundingBox()
    await group.getByRole('button', { name: 'Qwen3', exact: true }).click()
    await expect(group.getByRole('button', { name: 'Qwen3', exact: true })).toHaveAttribute('aria-pressed', 'true')
    expect(await group.boundingBox(), `selector shifts at ${width}px`).toEqual(before)
    expect(await page.locator('.student-header').boundingBox(), `header shifts at ${width}px`).toEqual(header)
    expect(await page.getByRole('button', { name: '个人中心', exact: true }).boundingBox()).toEqual(account)
    expect(before!.x).toBeGreaterThanOrEqual(0)
    expect(before!.x + before!.width).toBeLessThanOrEqual(width)
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(width)
    if (width === 1440 || width === 390) await page.locator('.student-header').screenshot({ path: `../output/model-selector-${width}-${info.project.name}.png` })
  }
  expect(await group.textContent()).not.toMatch(/openai|vllm|LoRA|solace-response/)
  const result = await new AxeBuilder({ page }).include('.student-header').analyze()
  expect(result.violations.filter(v => ['critical','serious'].includes(v.impact || ''))).toEqual([])
})
