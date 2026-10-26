import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import AdminSidebar from '@/components/admin/AdminSidebar.vue'
import KnowledgeView from '@/components/admin/KnowledgeView.vue'

describe('admin controls', () => {
  it('keeps every compact navigation item accessible by name', () => {
    const wrapper = mount(AdminSidebar, {
      props: { section: 'overview', displayName: '李老师' },
      global: { stubs: { BrandMark: true, ThemeToggle: true } },
    })

    expect(wrapper.get('button[aria-label="知识库"]')).toBeTruthy()
  })

  it('reflects the real knowledge upload request state', () => {
    const wrapper = mount(KnowledgeView, {
      props: { knowledge: null, busy: true },
      global: {
        stubs: {
          'el-upload': { template: '<div><slot /><slot name="tip" /></div>' },
        },
      },
    })

    const upload = wrapper.get('.primary-button.upload-action')
    expect(upload.attributes('disabled')).toBeDefined()
    expect(upload.text()).toBe('正在上传…')
  })
})
