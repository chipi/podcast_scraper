// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import InsightNodeView from './InsightNodeView.vue'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import { fetchInsightDetail } from '../../api/relationalApi'

vi.mock('../../api/relationalApi', () => ({ fetchInsightDetail: vi.fn() }))
const fetchDetail = vi.mocked(fetchInsightDetail)

const DETAIL = {
  subject: 'insight:1',
  text: 'Developing autonomy in-house is crucial for Rivian.',
  insight_type: 'claim',
  grounded: true,
  episode_id: 'ep-1',
  show_id: 'show1',
  quotes: [{ id: 'quote:q1', type: 'quote', text: 'You need a robust data architecture.', show_id: '', episode_id: '' }],
  topics: [{ id: 'topic:autonomy-strategy', type: 'topic', text: 'autonomy strategy', show_id: '', episode_id: '' }],
  entities: [{ id: 'org:rivian', type: 'org', text: 'Rivian', show_id: '', episode_id: '' }],
  error: null,
}

async function mountIt(id = 'g:insight:1') {
  const shell = useShellStore()
  shell.corpusPath = '/corpus'
  shell.healthStatus = 'ok'
  const w = mount(InsightNodeView, { props: { subjectIdOverride: id } })
  for (let i = 0; i < 8; i++) await w.vm.$nextTick()
  return { w, subject: useSubjectStore() }
}

describe('InsightNodeView', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    fetchDetail.mockResolvedValue(DETAIL as never)
  })

  it('strips the layer prefix and renders the insight text, quotes, topics, entities', async () => {
    const { w } = await mountIt('g:insight:1')
    // Prefixed graph id → bare corpus id for the endpoint.
    expect(fetchDetail).toHaveBeenCalledWith('/corpus', 'insight:1')
    expect(w.get('[data-testid="insight-node-text"]').text()).toContain('autonomy in-house')
    expect(w.get('[data-testid="insight-node-quotes"]').text()).toContain('robust data architecture')
    expect(w.get('[data-testid="insight-node-topics"]').text()).toContain('autonomy strategy')
    expect(w.get('[data-testid="insight-node-entities"]').text()).toContain('Rivian')
  })

  it('emits the resolved text up (for the rail header)', async () => {
    const { w } = await mountIt()
    expect(w.emitted('resolved')![0]).toEqual([DETAIL.text])
  })

  it('opens a topic / entity node view on chip click', async () => {
    const { w, subject } = await mountIt()
    await w.get('[data-testid="insight-node-topics"]').find('button').trigger('click')
    expect(subject.graphNodeCyId).toBe('topic:autonomy-strategy')
    await w.get('[data-testid="insight-node-entities"]').find('button').trigger('click')
    expect(subject.graphNodeCyId).toBe('org:rivian')
  })

  it('surfaces a not_found error without crashing', async () => {
    fetchDetail.mockResolvedValue({ ...DETAIL, error: 'not_found', text: '' } as never)
    const { w } = await mountIt()
    expect(w.find('[data-testid="insight-node-text"]').exists()).toBe(false)
  })
})
