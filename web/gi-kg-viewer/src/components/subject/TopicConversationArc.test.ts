// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('posthog-js', () => ({ default: { capture: vi.fn() } }))

const fetchTopicConversationArc = vi.fn()
const fetchTopicTimeline = vi.fn()
vi.mock('../../api/cilApi', () => ({
  fetchTopicConversationArc: (...a: unknown[]) => fetchTopicConversationArc(...a),
  fetchTopicTimeline: (...a: unknown[]) => fetchTopicTimeline(...a),
}))

import TopicConversationArc from './TopicConversationArc.vue'
import { useShellStore } from '../../stores/shell'

function arc() {
  return {
    path: '/c',
    topic_id: 'topic:ai',
    weeks: [
      { week: '2024-W03', volume: 3, negative: 1, neutral: 1, positive: 1, avg_compound: 0.1 },
      { week: '2024-W04', volume: 1, negative: 0, neutral: 0, positive: 1, avg_compound: 0.6 },
    ],
  }
}
function timeline() {
  return {
    path: '/c',
    topic_id: 'topic:ai',
    episodes: [
      {
        episode_id: 'ep1',
        publish_date: '2024-01-15', // ISO 2024-W03
        insights: [
          { id: 'i1', properties: { text: 'AI is a great breakthrough' }, sentiment: { compound: 0.8, label: 'positive' } },
          { id: 'i2', properties: { text: 'AI poses serious risks' }, sentiment: { compound: -0.6, label: 'negative' } },
        ],
      },
      {
        episode_id: 'ep2',
        publish_date: '2024-01-22', // ISO 2024-W04
        insights: [
          { id: 'i3', properties: { text: 'AI agents now do real work' }, sentiment: { compound: 0.5, label: 'positive' } },
        ],
      },
    ],
  }
}

async function mountArc() {
  fetchTopicConversationArc.mockResolvedValue(arc())
  fetchTopicTimeline.mockResolvedValue(timeline())
  const w = mount(TopicConversationArc, { props: { topicId: 'topic:ai' } })
  useShellStore().corpusPath = '/c'
  await w.setProps({ topicId: 'topic:ai ' })
  await w.setProps({ topicId: 'topic:ai' })
  await flushPromises()
  return w
}

describe('TopicConversationArc', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    fetchTopicConversationArc.mockReset()
    fetchTopicTimeline.mockReset()
  })

  it('renders a weekly bar per arc bucket + the tinted insight list', async () => {
    const w = await mountArc()
    expect(w.find('[data-testid="topic-conversation-arc"]').exists()).toBe(true)
    expect(w.find('[data-testid="tca-bar-2024-W03"]').exists()).toBe(true)
    expect(w.find('[data-testid="tca-bar-2024-W04"]').exists()).toBe(true)
    // all 3 insights listed by default (no week selected)
    expect(w.findAll('[data-testid="tca-insights"] > li').length).toBe(3)
  })

  it('clicking a week narrows the drill list to that week', async () => {
    const w = await mountArc()
    await w.get('[data-testid="tca-bar-2024-W03"]').trigger('click')
    // W03 has 2 insights (ep1); W04's single insight is filtered out.
    expect(w.find('[data-testid="tca-week-filter"]').exists()).toBe(true)
    expect(w.findAll('[data-testid="tca-insights"] > li').length).toBe(2)
  })
})
