// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

import { fetchTopicPerspectives } from '../../api/cilApi'
import NodeTopicPerspectives from './NodeTopicPerspectives.vue'

vi.mock('../../api/cilApi', () => ({
  fetchTopicPerspectives: vi.fn(),
}))

const fetchMock = vi.mocked(fetchTopicPerspectives)

function insight(text: string) {
  return { id: 'i-' + text, type: 'Insight', properties: { text } }
}

function mountFor(topicId: string) {
  return mount(NodeTopicPerspectives, { props: { corpusPath: '/corpus', topicId } })
}

describe('NodeTopicPerspectives (#1146)', () => {
  it('renders each speaker with their grounded insights', async () => {
    fetchMock.mockResolvedValue({
      path: '/corpus',
      topic_id: 'topic:ai',
      perspectives: [
        {
          person_id: 'person:jack',
          person_name: 'Jack Clark',
          insight_count: 2,
          episode_count: 1,
          insights: [insight('Take one'), insight('Take two')],
        },
        {
          person_id: 'person:amy',
          person_name: 'Amy Ng',
          insight_count: 1,
          episode_count: 1,
          insights: [insight('Amy take')],
        },
      ],
    } as never)
    const w = mountFor('topic:ai')
    await flushPromises()
    const cards = w.findAll('[data-testid="node-topic-perspective"]')
    expect(cards).toHaveLength(2)
    expect(cards[0].text()).toContain('Jack Clark')
    expect(cards[0].text()).toContain('2 insights')
    expect(cards[0].text()).toContain('Take one')
  })

  it('shows the empty state when a topic has no attributed perspectives', async () => {
    fetchMock.mockResolvedValue({
      path: '/corpus',
      topic_id: 'topic:empty',
      perspectives: [],
    } as never)
    const w = mountFor('topic:empty')
    await flushPromises()
    expect(w.text()).toContain('No attributed perspectives')
    expect(w.findAll('[data-testid="node-topic-perspective"]')).toHaveLength(0)
  })
})
