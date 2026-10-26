import { expect, test, type Page } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

async function mockHealth(page: Page) {
  await page.route('**/actuator/health', (route) => route.fulfill({ json: { status: 'UP' } }))
}

async function authenticate(page: Page, admin = false) {
  await page.addInitScript((isAdmin) => {
    sessionStorage.setItem('solaceai.auth', JSON.stringify({ token: 'demo', username: isAdmin ? 'admin' : 'student', displayName: isAdmin ? '心理中心' : '林同学', roles: [{ authority: isAdmin ? 'ROLE_ADMIN' : 'ROLE_STUDENT' }] }))
  }, admin)
  await page.route('**/api/profile', (route) => route.fulfill({ json: { id: 1, username: admin ? 'admin' : 'student', displayName: admin ? '心理中心' : '林同学', roles: [{ authority: admin ? 'ROLE_ADMIN' : 'ROLE_STUDENT' }] } }))
  await page.route('**/api/agent/status', (route) => route.fulfill({ json: { model: 'solaceai-qwen2.5-7b-ft:latest', provider: 'ollama', realModelEnabled: true } }))
  await mockHealth(page)
}

test('login offers one accessible sign-in task', async ({ page }) => {
  await mockHealth(page)
  await page.goto('/')
  await expect(page.getByRole('heading', { name: /欢迎回到 SolaceAI/ })).toBeVisible()
  await expect(page.getByLabel('用户名')).toBeVisible()
  await page.locator('.login-access').evaluate((element) => element.getAnimations().map((animation) => animation.finish()))
  const results = await new AxeBuilder({ page }).analyze()
  expect(results.violations.filter((item) => item.impact === 'critical' || item.impact === 'serious')).toEqual([])
})

test('student can start, stream and stop from the mobile-first workspace', async ({ page }, testInfo) => {
  await authenticate(page)
  await page.route('**/api/chat/stream', (route) => route.fulfill({ status: 200, contentType: 'text/event-stream', body: 'data: {"type":"meta","sessionId":"s-1"}\n\ndata: {"type":"token","content":"我在这里，我们慢慢来。"}\n\n' }))
  await page.goto('/student.html')
  await expect(page.getByRole('heading', { name: /林同学/ })).toBeVisible()
  await page.getByRole('button', { name: '最近有件事放不下' }).click()
  await page.getByRole('button', { name: /发送消息/ }).click()
  await expect(page.getByText('我在这里，我们慢慢来。')).toBeVisible()
  if (testInfo.project.name === 'mobile') {
    await page.getByRole('button', { name: '情绪快捷入口' }).click()
    await expect(page.getByRole('dialog')).toBeVisible()
  }
})

test('admin overview and knowledge workspace use existing APIs', async ({ page }) => {
  await authenticate(page, true)
  const reports = [{ id: 1, sessionId: 's1', username: 'lin', displayName: '林同学', content: '考试焦虑', intent: 'CONSULT', emotion: 'ANXIETY', emotionScore: .7, riskLevel: 'MEDIUM', summary: '近期考试压力明显', createdAt: '2026-08-26T08:00:00Z' }]
  await page.route('**/api/admin/reports', (route) => route.fulfill({ json: reports }))
  await page.route('**/api/admin/cases', (route) => route.fulfill({ json: [] }))
  await page.route('**/api/admin/excel-records', (route) => route.fulfill({ json: [] }))
  await page.route('**/api/admin/alerts', (route) => route.fulfill({ json: [] }))
  await page.route('**/api/admin/knowledge/documents', (route) => route.fulfill({ json: [] }))
  await page.route('**/api/admin/knowledge/status', (route) => route.fulfill({ json: { databaseChunks: 128, vectorAvailable: true, vectorChunks: 128 } }))
  await page.goto('/admin.html')
  await expect(page.getByRole('heading', { name: '工作概览' })).toBeVisible()
  await expect(page.getByText('近期考试压力明显')).toBeVisible()
  await page.getByRole('button', { name: /知识库/ }).click()
  await expect(page).toHaveURL(/view=knowledge/)
  await expect(page.getByRole('heading', { name: '知识库状态' })).toBeVisible()
})

test('student registration and email login expose verified email fields', async ({ page }) => {
  await mockHealth(page)
  await page.route('**/api/profile', route => route.fulfill({ status: 401, json: { detail: 'unauthorized' } }))
  await page.route('**/api/auth/code', route => route.fulfill({ json: { resendAfter: 60, expiresIn: 300 } }))
  await page.goto('/')
  await page.getByRole('button', { name: '学生注册', exact: true }).click()
  await expect(page.getByRole('heading', { name: '创建学生账号' })).toBeVisible()
  await page.getByLabel('邮箱', { exact: true }).fill('demo@qq.com')
  await page.getByRole('button', { name: '发送验证码', exact: true }).click()
  await expect(page.getByRole('status')).toContainText('请查收验证码')
  await expect(page.getByRole('button', { name: /秒后重发/ })).toBeDisabled()
  await expect(page.getByLabel('确认密码')).toBeVisible()
  await page.getByRole('button', { name: '邮箱验证码', exact: true }).click()
  await expect(page.getByRole('heading', { name: '邮箱验证码登录' })).toBeVisible()
  await expect(page.getByLabel('密码', { exact: true })).toHaveCount(0)
  await expect(page.getByLabel('邮箱验证码', { exact: true })).toBeVisible()
})

test('answer trace is expanded, can collapse and survives history loading', async ({ page }) => {
  await authenticate(page)
  const step = { id: 'step-1', sequence: 1, stage: 'sanitize', title: '输入隐私处理', why: '减少敏感信息进入后续处理。', status: 'completed', result: '已完成输入隐私处理。', startedAt: new Date().toISOString(), durationMs: 12 }
  const answerTrace = { runId: 'run-demo', messageId: 42, status: 'completed', steps: [step] }
  await page.route('**/api/sessions?*', route => route.fulfill({ json: { items: [{ sessionId: 's-1', title: '演示对话', updatedAt: new Date().toISOString() }], total: 1 } }))
  await page.route('**/api/sessions', route => route.fulfill({ json: { items: [{ sessionId: 's-1', title: '演示对话', updatedAt: new Date().toISOString() }], total: 1 } }))
  await page.route('**/api/sessions/s-1', route => route.fulfill({ json: { sessionId: 's-1', messages: [{ id: '42', role: 'assistant', content: '演示回复。', messageId: 42, runId: 'run-demo', answerTrace }] } }))
  const events = [{ type: 'meta', sessionId: 's-1', runId: 'run-demo' }, { type: 'step', sessionId: 's-1', runId: 'run-demo', step }, { type: 'token', content: '演示回复。' }, { type: 'done', messageId: 42, runId: 'run-demo', traceStatus: 'completed' }]
  await page.route('**/api/chat/stream', route => route.fulfill({ contentType: 'text/event-stream', body: events.map(event => `data: ${JSON.stringify(event)}\n\n`).join('') }))
  await page.goto('/student.html')
  await page.getByRole('button', { name: '最近有件事放不下' }).click()
  await page.getByRole('button', { name: /发送消息/ }).click()
  await expect(page.getByText('输入隐私处理', { exact: true })).toBeVisible()
  await expect(page.locator('.answer-trace')).toHaveAttribute('open', '')
  await page.locator('.trace-summary').click()
  await expect(page.locator('.answer-trace')).not.toHaveAttribute('open', '')
  await page.locator('.trace-summary').click()
  await expect(page.getByText('12 ms')).toBeVisible()
  await page.reload()
  const sidebarToggle = page.getByRole('button', { name: '打开会话侧栏' })
  if (await sidebarToggle.isVisible()) await sidebarToggle.click()
  await page.getByRole('button', { name: /演示对话/ }).first().click()
  await expect(page.getByText('输入隐私处理', { exact: true })).toBeVisible()
  await expect(page.getByText('演示回复。')).toBeVisible()
})

test('ReAct phases show rounds, agents and observed tool outcomes', async ({ page }, testInfo) => {
  await authenticate(page)
  await page.route('**/api/memories', route => route.fulfill({ json: [] }))
  await page.route('**/api/sessions?*', route => route.fulfill({ json: { items: [], total: 0 } }))
  const base = { startedAt: new Date().toISOString(), status: 'completed', durationMs: 120, round: 1 }
  const steps = [
    { ...base, id: 'r', sequence: 1, stage: 'agent', phase: 'reason', actor: 'MainAgent', title: 'Agent 决策摘要', why: '结合已有依据选择下一步动作。', result: '下一步调用 search_knowledge（检索参考资料）。', nextAction: 'search_knowledge' },
    { ...base, id: 'a', sequence: 2, stage: 'retrieval', phase: 'act', actor: 'RetrievalAgent', title: '检索参考资料', why: '寻找与当前问题有关的知识资料。', result: '已完成资料检索。', tool: 'search_knowledge', count: 3 },
    { ...base, id: 'o', sequence: 3, stage: 'observation', phase: 'observe', actor: 'MainAgent', title: '观察工具结果', why: '整理本次结果，供后续决策使用。', result: '已接收 search_knowledge 的执行结果。', tool: 'search_knowledge', count: 3 },
    { ...base, id: 'f', sequence: 4, round: 2, stage: 'answer_ready', phase: 'answer', actor: 'MainAgent', title: '接收候选回答', why: '接收候选回答并准备审核。', result: '候选回答已就绪。' },
  ]
  const events = [{ type: 'meta', runId: 'react-demo', sessionId: 's-react' }, ...steps.map(step => ({ type: 'step', runId: 'react-demo', sessionId: 's-react', step })), { type: 'token', content: '可以先给自己安排一个短暂的休息，再把任务拆成今天能完成的一小步。' }, { type: 'done', runId: 'react-demo', messageId: 5, traceStatus: 'completed' }]
  await page.route('**/api/chat/stream', route => route.fulfill({ contentType: 'text/event-stream', body: events.map(event => `data: ${JSON.stringify(event)}\n\n`).join('') }))
  await page.goto('/student.html')
  await page.getByRole('button', { name: '最近有件事放不下' }).click()
  await page.getByRole('button', { name: /发送消息/ }).click()
  await expect(page.locator('.trace-step[data-phase="reason"]')).toContainText('Reason')
  await expect(page.locator('.trace-step[data-phase="act"]')).toContainText('RetrievalAgent')
  await expect(page.locator('.trace-step[data-phase="observe"]')).toContainText('search_knowledge')
  await expect(page.locator('.trace-step[data-phase="answer"]')).toContainText('第 2 轮')
  await expect(page.locator('.trace-step')).toHaveCount(4)
  await page.locator('.trace-step[data-phase="observe"]').scrollIntoViewIfNeeded()
  await expect(page.locator('.trace-step[data-phase="observe"]')).toBeInViewport()
  await page.locator('.trace-summary').scrollIntoViewIfNeeded()
  await page.locator('.answer-trace').screenshot({ path: `.deploy-private/react-trace-${testInfo.project.name}.png` })
})

test('offline retrieval shows dense sparse fusion and reranker candidate details', async ({ page }) => {
  await authenticate(page)
  await page.route('**/api/memories', route => route.fulfill({ json: [] }))
  await page.route('**/api/sessions?*', route => route.fulfill({ json: { items: [], total: 0 } }))
  const groups = ['dense', 'sparse', 'milvus_rrf', 'rrf', 'rerank', 'evidence'].map((channel, i) => ({ channel, queryIndex: 1, method: channel === 'rerank' ? 'semantic_local_cross_encoder' : 'rrf', total: 1, candidates: [{ rank: 1, chunkId: i + 1, source: `压力资料-${i + 1}`, content: `实际候选正文-${i + 1}`, score: 0.9 - i / 10 }] }))
  const steps = [
    { stage: 'recall', title: '多路召回与 Milvus RRF', retrievalDetails: groups.slice(0, 3) },
    { stage: 'fusion', title: '跨查询 RRF 融合', retrievalDetails: groups.slice(3, 4) },
    { stage: 'rerank', title: '参考资料排序', retrievalDetails: groups.slice(4, 5) },
    { stage: 'evidence_selection', title: '最终证据选择', retrievalDetails: groups.slice(5) },
  ].map((step, i) => ({ ...step, id: `d-${i}`, sequence: i + 1, phase: 'act', actor: 'RetrievalAgent', why: '实际执行结果', result: '已完成', status: 'completed', durationMs: 100, startedAt: new Date().toISOString() }))
  const events = [{ type: 'meta', runId: 'detail', sessionId: 'detail-session' }, ...steps.map(step => ({ type: 'step', runId: 'detail', sessionId: 'detail-session', step })), { type: 'token', content: '先试着放慢呼吸。' }, { type: 'done', runId: 'detail', sessionId: 'detail-session', traceStatus: 'completed' }]
  await page.route('**/api/chat/stream', route => route.fulfill({ contentType: 'text/event-stream', body: events.map(event => `data: ${JSON.stringify(event)}\n\n`).join('') }))
  await page.goto('/student.html')
  await page.getByRole('button', { name: '最近有件事放不下' }).click()
  await page.getByRole('button', { name: /发送消息/ }).click()
  await expect(page.locator('.retrieval-lane')).toHaveCount(6)
  await expect(page.getByText('实际候选正文-5', { exact: true })).toBeAttached()
  await expect(page.locator('.retrieval-details')).toContainText(['密集向量召回', '跨查询 RRF', '重排（Reranker）', '最终参考证据'])
  await expect(page.getByText('本地 Cross-Encoder', { exact: false })).toBeAttached()
})
