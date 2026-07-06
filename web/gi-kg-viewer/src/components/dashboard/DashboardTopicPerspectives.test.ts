// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { fetchTopicPerspectiveLeaders } from '../../api/cilApi'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import DashboardTopicPerspectives from './DashboardTopicPerspectives.vue'

vi.mock('../../api/cilApi', () => ({ fetchTopicPerspectiveLeaders: vi.fn() }))
const fetchMock = vi.mocked(fetchTopicPerspectiveLeaders)

beforeEach(() => setActivePinia(createPinia()))

describe('DashboardTopicPerspectives (#1146)', () => {
  it('lists topics by speaker count and focuses one on click', async () => {
    fetchMock.mockResolvedValue({
      path: '/corpus',
      topics: [
        { topic_id: 'topic:ai', topic_label: 'AI development', speaker_count: 10, insight_count: 38 },
        { topic_id: 'topic:energy', topic_label: 'Energy crisis', speaker_count: 5, insight_count: 11 },
      ],
    } as never)
    useShellStore().corpusPath = '/corpus'
    const w = mount(DashboardTopicPerspectives)
    await flushPromises()
    const rows = w.findAll('[data-testid="dashboard-topic-perspective-row"]')
    expect(rows).toHaveLength(2)
    expect(rows[0].text()).toContain('AI development')
    expect(rows[0].text()).toContain('10 speakers')

    const spy = vi.spyOn(useSubjectStore(), 'focusTopic')
    await rows[0].trigger('click')
    expect(spy).toHaveBeenCalledWith('topic:ai')
  })

  it('shows an empty state when no topic has multiple speakers', async () => {
    fetchMock.mockResolvedValue({ path: '/corpus', topics: [] } as never)
    useShellStore().corpusPath = '/corpus'
    const w = mount(DashboardTopicPerspectives)
    await flushPromises()
    expect(w.text()).toContain('No multi-perspective topics')
  })
})
