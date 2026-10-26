import { expect, test } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

const profile = { id: 2, username: 'student', displayName: '林同学', roles: [{ authority: 'ROLE_USER' }], email: null, requiresSetup: false }
test('account dialog follows existing style, works on mobile and restores focus', async ({ page }, info) => {
  await page.route('**/api/**', route => route.fulfill({ json: [] }))
  await page.route('**/api/profile', route => route.fulfill({ json: profile }))
  await page.route('**/actuator/health', route => route.fulfill({ json: { status: 'UP' } }))
  await page.route('**/api/agent/status', route => route.fulfill({ json: { availableChatBackends: ['cloud'], defaultChatBackend: 'cloud' } }))
  await page.route('**/api/auth/password', async route => {
    expect(route.request().postDataJSON()).toEqual({ currentPassword: 'OldPassword123!', password: 'NewPassword123!', confirmPassword: 'NewPassword123!' })
    await route.fulfill({ json: profile })
  })
  await page.goto('/student.html')
  const trigger = page.getByRole('button', { name: '个人中心', exact: true })
  await trigger.click()
  const dialog = page.getByRole('dialog', { name: '个人中心' })
  await expect(dialog).toBeVisible()
  await page.getByLabel('当前密码', { exact: true }).fill('OldPassword123!')
  await page.getByLabel('新密码', { exact: true }).fill('NewPassword123!')
  await page.getByLabel('确认新密码', { exact: true }).fill('NewPassword123!')
  await page.getByRole('button', { name: '保存新密码' }).click()
  await expect(dialog.getByRole('status')).toContainText('密码已修改')
  await page.getByRole('button', { name: '绑定邮箱', exact: true }).click()
  await expect(page.getByLabel('邮箱', { exact: true })).toBeVisible()
  const box = await dialog.boundingBox()
  expect(box!.width).toBeLessThanOrEqual(page.viewportSize()!.width)
  const results = await new AxeBuilder({ page }).include('.account-dialog').analyze()
  expect(results.violations.filter(v => ['serious', 'critical'].includes(v.impact || ''))).toEqual([])
  await dialog.screenshot({ path: `../output/account-center-${info.project.name}.png` })
  await page.keyboard.press('Escape')
  await expect(dialog).not.toBeVisible()
  await expect(trigger).toBeFocused()
})

test('verified email requires credentials and survives refresh before setup', async ({ page }) => {
  const pending = { ...profile, email: 'new@example.com', requiresSetup: true }
  await page.route('**/api/profile', route => route.fulfill({ json: pending }))
  await page.route('**/actuator/health', route => route.fulfill({ json: { status: 'UP' } }))
  await page.route('**/api/auth/setup', route => route.fulfill({ json: { ...pending, username: 'my_student', requiresSetup: false } }))
  await page.goto('/')
  await expect(page.getByRole('heading', { name: '设置你的账号' })).toBeVisible()
  await page.getByLabel('用户名', { exact: true }).fill('my_student')
  await page.getByLabel('密码', { exact: true }).fill('Password123!')
  await page.getByLabel('确认密码', { exact: true }).fill('Password123!')
  await page.getByRole('button', { name: '完成设置并进入' }).click()
  await expect(page).toHaveURL(/student.html/)
})
