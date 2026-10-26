import { expect, test, type Page } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

async function setup(page: Page, theme: string, role: 'login' | 'student' | 'admin') {
  await page.addInitScript(({ theme, role }) => {
    localStorage.setItem('solaceai.theme', theme)
    if (role !== 'login') sessionStorage.setItem('solaceai.auth', JSON.stringify({ token: 'preview-only', username: role, displayName: role === 'admin' ? '心理中心' : '林同学', roles: [{ authority: role === 'admin' ? 'ROLE_ADMIN' : 'ROLE_STUDENT' }] }))
  }, { theme, role })
  await page.route('**/actuator/health', r => r.fulfill({ json: { status: 'UP' } }))
  await page.route('**/api/**', r => {
    const url = new URL(r.request().url())
    if (url.pathname === '/api/profile') return role === 'login' ? r.fulfill({ status: 401, json: { detail: 'unauthorized' } }) : r.fulfill({ json: { id: 1, username: role, displayName: role === 'admin' ? '心理中心' : '林同学', roles: [{ authority: role === 'admin' ? 'ROLE_ADMIN' : 'ROLE_STUDENT' }] } })
    if (url.pathname === '/api/agent/status') return r.fulfill({ json: { model: 'campus-support', provider: 'ollama', realModelEnabled: true } })
    if (url.pathname === '/api/sessions') return r.fulfill({ json: { items: [{ sessionId: 'demo-1', title: '考试前，想让自己放松一些', updatedAt: '2026-10-21T08:00:00Z' }], total: 1 } })
    if (url.pathname === '/api/admin/reports') return r.fulfill({ json: [{ id: 1, sessionId: 'demo-1', username: 'demo', displayName: '示例同学', content: '演示数据', intent: 'CONSULT', emotion: 'ANXIETY', emotionScore: .5, riskLevel: 'LOW', summary: '考试前的紧张与睡眠安排（演示）', createdAt: '2026-10-21T08:00:00Z' }] })
    if (url.pathname === '/api/admin/knowledge/status') return r.fulfill({ json: { backend: 'milvus_bge', databaseChunks: 128, vectorAvailable: true, vectorChunks: 128 } })
    return r.fulfill({ json: [] })
  })
}

for (const width of [1440, 768, 390]) {
  for (const theme of ['light']) {
    for (const role of ['login', 'student', 'admin'] as const) {
      test(`${role} ${theme} at ${width}px: layout, access and screenshot`, async ({ page }, testInfo) => {
        test.skip(testInfo.project.name !== 'chromium', 'Explicit viewport matrix runs once')
        await page.setViewportSize({ width, height: 960 })
        await page.emulateMedia({ reducedMotion: 'reduce' })
        await setup(page, theme, role)
        await page.goto(role === 'login' ? '/' : `/${role}.html`)
        await expect(page.locator('html')).toHaveAttribute('data-theme', 'light')
        await expect(page.getByRole('heading', { level: 1 })).toBeVisible()
        await page.evaluate(async () => { await document.fonts.ready; await Promise.all(Array.from(document.images).filter(img => img.loading !== 'lazy' && img.getClientRects().length > 0).map(img => img.decode().catch(() => {}))) })
        await page.locator('.skip-link').focus()
        await expect(page.locator('.skip-link')).toBeInViewport()
        await page.locator('.skip-link').blur()
        if (role === 'student') {
          const send = page.getByRole('button', { name: '发送消息' })
          await expect(send).toBeInViewport()
          if (width <= 1020) {
            const toggle = page.getByRole('button', { name: '打开会话侧栏' })
            await toggle.click()
            await expect(page.getByRole('dialog', { name: '会话与支持' })).toBeVisible()
            await expect(page.getByRole('heading', { name: '历史会话' })).toBeVisible()
            await page.keyboard.press('Escape')
            await expect(page.getByRole('dialog', { name: '会话与支持' })).not.toBeVisible()
            await expect(toggle).toBeFocused()
            await toggle.blur()
          }
        }
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
        await page.locator('body').evaluate(el => el.getAnimations({ subtree: true }).filter(a => a.effect?.getComputedTiming().iterations !== Infinity).forEach(a => a.finish()))
        await page.screenshot({ path: `../output/redesign-v2/${role}-${theme}-${width}.png`, fullPage: true, animations: 'disabled' })
        const violations = (await new AxeBuilder({ page }).analyze()).violations.filter(v => v.impact === 'critical' || v.impact === 'serious')
        expect(violations.map(v => ({ id: v.id, nodes: v.nodes.map(n => ({ target: n.target, summary: n.failureSummary })) }))).toEqual([])
      })
    }
  }
}

for (const width of [1440, 390]) {
  test(`admin secondary workspaces stay usable at ${width}px`, async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== 'chromium', 'Explicit viewport matrix runs once')
    await page.setViewportSize({ width, height: 960 })
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await setup(page, 'light', 'admin')
    await page.goto('/admin.html')
    for (const [label, view] of [['风险个案', 'cases'], ['咨询报告', 'reports'], ['工具执行', 'delivery'], ['知识库', 'knowledge']]) {
      await page.getByRole('button', { name: label, exact: true }).click()
      await expect(page).toHaveURL(new RegExp(`view=${view}`))
      await expect(page.getByRole('heading', { level: 1 })).toHaveText(label === '知识库' ? '知识库维护' : label!)
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
      await page.screenshot({ path: `../output/redesign-v2/admin-${view}-${width}.png`, fullPage: true, animations: 'disabled' })
    }
  })
}

for (const role of ['login', 'student', 'admin'] as const) {
  test(`${role} uses one palette even with old dark preference`, async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== 'chromium', 'Migration check runs once')
    await setup(page, 'dark', role)
    await page.emulateMedia({ colorScheme: 'dark', reducedMotion: 'reduce' })
    await page.goto(role === 'login' ? '/' : `/${role}.html`)
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'light')
    await expect(page.getByRole('button', { name: /切换主题/ })).toHaveCount(0)
    expect(await page.evaluate(() => localStorage.getItem('solaceai.theme'))).toBeNull()
  })
}

test('teacher navigation stays on screen after scrolling the complete overview', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'chromium', 'Desktop sticky navigation check')
  await page.setViewportSize({ width: 1440, height: 800 })
  await setup(page, 'light', 'admin')
  await page.goto('/admin.html')
  await expect(page.getByRole('heading', { name: '最近咨询报告' })).toBeVisible()
  await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight))
  await expect(page.getByRole('button', { name: '数据概览', exact: true })).toBeInViewport()
  await expect(page.getByRole('button', { name: '知识库', exact: true })).toBeInViewport()
  await page.screenshot({ path: '../output/redesign-v2/admin-scrolled-1440.png', fullPage: false })
})

for (const width of [1440, 390]) {
  test(`execution records and the reply remain distinct at ${width}px`, async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== 'chromium', 'Explicit viewport check')
    await page.setViewportSize({ width, height: 960 })
    await setup(page, 'light', 'student')
    const steps = Array.from({ length: 5 }, (_, index) => ({ id: `s${index}`, sequence: index + 1, stage: 'sanitize', title: ['输入隐私处理', '读取会话上下文', '输入安全检查', '整理参考资料', '完成回复准备'][index], why: '在当前授权范围内整理本次对话所需的信息。', status: 'completed', result: '已完成处理。', startedAt: '2026-10-21T08:00:00Z', durationMs: 12 }))
    await page.route('**/api/chat/stream', r => r.fulfill({ contentType: 'text/event-stream', body: [{ type: 'meta', sessionId: 'preview-session', runId: 'preview-trace' }, ...steps.map(step => ({ type: 'step', sessionId: 'preview-session', runId: 'preview-trace', step })), { type: 'token', content: '听起来，这段时间你一直在努力撑着。\n\n我们可以先给今天留十分钟的休息，再一起看看最需要照顾的是哪一件事。' }, { type: 'done', sessionId: 'preview-session', messageId: 7, runId: 'preview-trace', traceStatus: 'completed' }].map(e => `data: ${JSON.stringify(e)}\n\n`).join('') }))
    await page.goto('/student.html')
    await page.getByRole('button', { name: '最近有件事放不下' }).click()
    await page.getByRole('button', { name: '发送消息' }).click()
    await expect(page.getByText('给你的回复', { exact: false })).toBeVisible()
    await expect(page.locator('.answer-trace')).toHaveAttribute('open', '')
    await expect(page.locator('.trace-step')).toHaveCount(5)
    expect(await page.locator('.trace-body').evaluate(el => el.scrollHeight > el.clientHeight)).toBe(true)
    await page.getByRole('region', { name: '执行步骤记录' }).focus()
    await page.keyboard.press('End')
    await expect.poll(() => page.locator('.trace-body').evaluate(el => el.scrollTop)).toBeGreaterThan(0)
    await page.locator('.trace-body').evaluate(el => el.scrollTop = 0)
    await page.locator('.trace-body').blur()
    await expect(page.locator('.welcome-photo')).toHaveCount(0)
    await expect(page.getByRole('button', { name: '发送消息' })).toBeInViewport()
    await page.locator('.message-list').evaluate(el => el.scrollTop = 0)
    await page.screenshot({ path: `../output/redesign-v2/student-reply-${width}.png`, fullPage: true })
    const violations = (await new AxeBuilder({ page }).analyze()).violations.filter(v => v.impact === 'critical' || v.impact === 'serious')
    expect(violations.map(v => ({ id: v.id, nodes: v.nodes.map(n => ({ target: n.target, summary: n.failureSummary })) }))).toEqual([])
  })
}

for (const width of [1440, 390]) {
  test(`themed removal and new-conversation confirmations at ${width}px`, async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== 'chromium', 'Explicit viewport check')
    await page.setViewportSize({ width, height: 960 })
    await setup(page, 'light', 'student')
    const nativeDialogs: string[] = []
    page.on('dialog', async dialog => { nativeDialogs.push(dialog.message()); await dialog.dismiss() })
    let deletes = 0
    await page.route('**/api/sessions?*', r => r.fulfill({ json: { items: deletes ? [] : [{ sessionId: 'demo-1', title: '考试前，想让自己放松一些' }], total: deletes ? 0 : 1 } }))
    await page.route('**/api/sessions/demo-1', r => { deletes++; return r.fulfill({ json: { deleted: true } }) })
    await page.route('**/api/chat/stream', r => r.fulfill({ contentType: 'text/event-stream', body: 'data: {"type":"meta","sessionId":"new-demo"}\n\ndata: {"type":"token","content":"我们可以慢慢来，先照顾好今天的心情。"}\n\ndata: {"type":"done","messageId":1}\n\n' }))
    await page.goto('/student.html')
    const sidebar = page.getByRole('button', { name: '打开会话侧栏' })
    if (width === 390) await sidebar.click()
    const remove = page.getByRole('button', { name: '移除会话：考试前，想让自己放松一些' })
    await remove.click()
    const removal = page.getByRole('dialog', { name: '移除这条会话？' })
    await expect(removal).toBeVisible()
    await expect(removal.getByRole('button', { name: '取消', exact: true })).toBeFocused()
    await page.screenshot({ path: `../output/redesign-v2/remove-confirm-${width}.png`, fullPage: true })
    await page.keyboard.press('Escape')
    await expect(removal).not.toBeVisible()
    expect(deletes).toBe(0)
    await expect(remove).toBeFocused()
    await remove.click()
    await removal.getByRole('button', { name: '移除会话', exact: true }).click()
    await expect.poll(() => deletes).toBe(1)
    await expect(remove).toHaveCount(0)
    if (width === 390) await page.getByRole('button', { name: '关闭会话侧栏' }).click()
    await page.getByRole('button', { name: '最近有件事放不下' }).click()
    await page.getByRole('button', { name: '发送消息' }).click()
    await expect(page.getByText('我们可以慢慢来，先照顾好今天的心情。')).toBeVisible()
    if (width === 390) { await sidebar.click(); await page.getByRole('button', { name: '＋ 开始新会话' }).click() }
    else await page.getByRole('button', { name: '新会话', exact: true }).click()
    const reset = page.getByRole('dialog', { name: '开始一段新对话？' })
    await expect(reset).toBeVisible()
    await page.screenshot({ path: `../output/redesign-v2/new-confirm-${width}.png`, fullPage: true })
    const violations = (await new AxeBuilder({ page }).analyze()).violations.filter(v => v.impact === 'critical' || v.impact === 'serious')
    expect(violations.map(v => ({ id: v.id, targets: v.nodes.map(n => n.target) }))).toEqual([])
    await reset.getByRole('button', { name: '取消', exact: true }).click()
    await expect(page.getByText('我们可以慢慢来，先照顾好今天的心情。')).toBeVisible()
    if (width === 390) { await sidebar.click(); await page.getByRole('button', { name: '＋ 开始新会话' }).click() }
    else await page.getByRole('button', { name: '新会话', exact: true }).click()
    await reset.getByRole('button', { name: '开始新会话', exact: true }).click()
    await expect(page.getByRole('heading', { name: '今天的心情，想从哪里说起？' })).toBeVisible()
    expect(nativeDialogs).toEqual([])
  })
}
