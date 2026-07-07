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

  it('caps insights at the preview count with a per-speaker expand toggle', async () => {
    const many = Array.from({ length: 6 }, (_, i) => insight(`Point ${i}`))
    fetchMock.mockResolvedValue({
      path: '/corpus',
      topic_id: 'topic:ai',
      perspectives: [
        {
          person_id: 'person:jack',
          person_name: 'Jack Clark',
          insight_count: 6,
          episode_count: 2,
          insights: many,
        },
      ],
    } as never)
    const w = mountFor('topic:ai')
    await flushPromises()
    // First 4 previewed; 5th/6th hidden until expanded.
    expect(w.text()).toContain('Point 0')
    expect(w.text()).toContain('Point 3')
    expect(w.text()).not.toContain('Point 4')
    const toggle = w.get('[data-testid="node-topic-perspective-toggle"]')
    expect(toggle.text()).toContain('+2 more')
    await toggle.trigger('click')
    expect(w.text()).toContain('Point 4')
    expect(w.text()).toContain('Point 5')
    expect(w.get('[data-testid="node-topic-perspective-toggle"]').text()).toContain('Show fewer')
  })

  it('shows a distinct error state when the fetch fails', async () => {
    fetchMock.mockRejectedValue(new Error('boom'))
    const w = mountFor('topic:ai')
    await flushPromises()
    expect(w.text()).toContain("Couldn't load perspectives")
    expect(w.findAll('[data-testid="node-topic-perspective"]')).toHaveLength(0)
  })
})
