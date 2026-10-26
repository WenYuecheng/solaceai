import { mount, flushPromises } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import MailDeliveryPanel from '@/components/admin/MailDeliveryPanel.vue'
import { requestJson } from '@/core/api'
vi.mock('@/core/api', () => ({ requestJson: vi.fn() }))
vi.mock('element-plus', () => ({ ElMessage: { error: vi.fn(), success: vi.fn() } }))

describe('mail delivery controls', () => {
  beforeEach(() => vi.mocked(requestJson).mockReset())
  it('shows log-only mode and sends tests only to configured recipients', async () => {
    vi.mocked(requestJson).mockResolvedValue({ mode: 'log', ready: false, missing: [], issues: [], alertRecipients: ['teacher@qq.com'], message: '仅记录，未发送邮件' })
    const wrapper = mount(MailDeliveryPanel, { props: { alerts: [] } })
    await flushPromises()
    expect(wrapper.text()).toContain('仅记录，未发送邮件')
    expect(wrapper.text()).toContain('teacher@qq.com')
    expect(wrapper.find('input').exists()).toBe(false)
    vi.mocked(requestJson).mockResolvedValue({ status: 'LOG_ONLY', message: '仅记录，未发送邮件' })
    await wrapper.get('[data-test="mail-test"]').trigger('click')
    await flushPromises()
    expect(requestJson).toHaveBeenLastCalledWith('/api/admin/mail/test', { method: 'POST' })
    expect(wrapper.get('[role="status"]').text()).toContain('仅记录，未发送邮件')
  })
  it('offers retry only for failed email records and refreshes after sending', async () => {
    vi.mocked(requestJson).mockResolvedValue({ mode: 'smtp', ready: true, missing: [], issues: [], alertRecipients: [] })
    const wrapper = mount(MailDeliveryPanel, { props: { alerts: [
      { id: 1, status: 'FAILED', channel: 'email', reportId: 4, message: '', createdAt: '' },
      { id: 2, status: 'SUCCESS', channel: 'email', reportId: 5, message: '', createdAt: '' },
    ] } })
    await flushPromises()
    expect(wrapper.findAll('[data-test="mail-retry"]')).toHaveLength(1)
    vi.mocked(requestJson).mockResolvedValue({ status: 'SUCCESS', message: 'SMTP 服务器已接受' })
    await wrapper.get('[data-test="mail-retry"]').trigger('click')
    await flushPromises()
    expect(requestJson).toHaveBeenLastCalledWith('/api/admin/mail/alerts/1/retry', { method: 'POST' })
    expect(wrapper.emitted('refresh')).toHaveLength(1)
    expect(wrapper.findAll('[data-test="mail-retry"]')).toHaveLength(0)
  })
})

it('treats a failed HTTP 200 test delivery as an error', async () => {
  vi.mocked(requestJson).mockResolvedValue({ mode: 'smtp', ready: true, missing: [], issues: [], alertRecipients: ['teacher@qq.com'] })
  const wrapper = mount(MailDeliveryPanel, { props: { alerts: [] } })
  await flushPromises()
  vi.mocked(requestJson).mockResolvedValue({ status: 'FAILED', message: 'SMTP 投递失败' })
  await wrapper.get('[data-test="mail-test"]').trigger('click')
  await flushPromises()
  expect(wrapper.get('[role="alert"]').text()).toContain('SMTP 投递失败')
  expect(wrapper.find('[role="status"]').exists()).toBe(false)
})

it('retains the failed replacement receipt even if the parent cannot refresh', async () => {
  vi.mocked(requestJson).mockResolvedValue({ mode: 'smtp', ready: true, missing: [], issues: [], alertRecipients: [] })
  const wrapper = mount(MailDeliveryPanel, { props: { alerts: [
    { id: 1, status: 'FAILED', channel: 'email', reportId: 4, message: '', createdAt: '' },
  ] } })
  await flushPromises()
  vi.mocked(requestJson).mockResolvedValue({ id: 3, status: 'FAILED', channel: 'email', reportId: 4, message: '投递失败', createdAt: '' })
  await wrapper.get('[data-test="mail-retry"]').trigger('click')
  await flushPromises()
  expect(wrapper.get('[role="alert"]').text()).toContain('投递失败')
  expect(wrapper.text()).toContain('预警 #3')
  expect(wrapper.findAll('[data-test="mail-retry"]')).toHaveLength(1)
})

it('requires an explicit recovery action for expired interrupted deliveries', async () => {
  vi.mocked(requestJson).mockResolvedValue({ mode: 'smtp', ready: true, missing: [], issues: [], alertRecipients: [] })
  const wrapper = mount(MailDeliveryPanel, { props: { alerts: [
    { id: 7, status: 'RUNNING', channel: 'email', reportId: 4, message: JSON.stringify({ mailRetryLease: '2020-01-01T00:00:00+00:00' }), createdAt: '' },
  ] } })
  await flushPromises()
  expect(wrapper.text()).toContain('可能重复')
  vi.mocked(requestJson).mockResolvedValue({ status: 'SUCCESS', message: 'SMTP 已接受' })
  await wrapper.get('[data-test="mail-recover"]').trigger('click')
  await flushPromises()
  expect(requestJson).toHaveBeenLastCalledWith('/api/admin/mail/alerts/7/retry', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ recoverInterrupted: true }),
  })
})
