import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import KnowledgeView from '@/components/admin/KnowledgeView.vue'

vi.mock('element-plus', () => ({ ElMessage: { error: vi.fn(), warning: vi.fn() }, ElMessageBox: { confirm: vi.fn() } }))

describe('knowledge backend maintenance', () => {
  it('does not offer a Chroma snapshot action for Milvus', async () => {
    const wrapper = mount(KnowledgeView, {
      props: { knowledge: { backend: 'milvus_bge', databaseChunks: 94, vectorAvailable: true } },
      global: { stubs: { 'el-upload': true } },
    })
    const backup = wrapper.findAll('button.secondary-button').find(button => button.text() === '使用卷备份')!
    expect(backup.attributes('disabled')).toBeDefined()
    await backup.trigger('click')
    expect(wrapper.emitted('action')).toBeUndefined()
    expect(wrapper.text()).toContain('持久化卷')
  })
})
