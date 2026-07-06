import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'

import en from '../i18n/locales/en.json'
import * as api from '../services/api'
import type { Insight, TopicPerspectivesResponse } from '../services/types'
import TopicPerspectives from './TopicPerspectives.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function mountIt(id: string) {
  return mount(TopicPerspectives, { props: { id }, global: { plugins: [i18n] } })
}

function insight(id: string, text: string): Insight {
  return { id, text, grounded: true, insight_type: 'claim', confidence: null, position_hint: null, quotes: [] }
}

const RESP: TopicPerspectivesResponse = {
  topic_id: 'topic:ai',
  topic_label: 'ai',
  perspective_count: 2,
  perspectives: [
    {
      person_id: 'person:jack-clark',
      person_name: 'Jack Clark',
      insight_count: 5,
      episode_count: 2,
      insights: [
        insight('i1', 'Take one'),
        insight('i2', 'Take two'),
        insight('i3', 'Take three'),
        insight('i4', 'Take four'),
        insight('i5', 'Take five'),
      ],
    },
    {
      person_id: 'person:amy-ng',
      person_name: 'Amy Ng',
      insight_count: 1,
      episode_count: 1,
      insights: [insight('a1', 'Amy take')],
    },
  ],
}

afterEach(() => vi.restoreAllMocks())

describe('TopicPerspectives', () => {
  it('renders each speaker with a capped preview of their insights', async () => {
    vi.spyOn(api, 'getTopicPerspectives').mockResolvedValue(RESP)
    const w = mountIt('topic:ai')
    await flushPromises()
    expect(w.get('[data-testid="topic-perspectives"]').text()).toContain('2 perspectives')
    const cards = w.findAll('[data-testid="topic-perspective"]')
    expect(cards).toHaveLength(2)
    expect(cards[0].text()).toContain('Jack Clark')
    expect(cards[0].text()).toContain('5 insights')
    // preview caps at 3; the rest sit behind "show more"
    expect(cards[0].text()).toContain('Take three')
    expect(cards[0].text()).not.toContain('Take four')
    expect(cards[0].text()).toContain('Show 2 more')
  })

  it('expands a speaker on "show more"', async () => {
    vi.spyOn(api, 'getTopicPerspectives').mockResolvedValue(RESP)
    const w = mountIt('topic:ai')
    await flushPromises()
    await w.findAll('[data-testid="topic-perspective"]')[0].get('button.text-accent').trigger('click')
    expect(w.findAll('[data-testid="topic-perspective"]')[0].text()).toContain('Take four')
  })

  it('emits open with the person when a speaker name is clicked', async () => {
    vi.spyOn(api, 'getTopicPerspectives').mockResolvedValue(RESP)
    const w = mountIt('topic:ai')
    await flushPromises()
    await w.findAll('[data-testid="topic-perspective"]')[0].get('button').trigger('click')
    expect(w.emitted('open')![0]).toEqual([{ kind: 'person', id: 'person:jack-clark' }])
  })

  it('renders nothing when the topic has no perspectives', async () => {
    vi.spyOn(api, 'getTopicPerspectives').mockResolvedValue({
      ...RESP,
      perspective_count: 0,
      perspectives: [],
    })
    const w = mountIt('topic:empty')
    await flushPromises()
    expect(w.find('[data-testid="topic-perspectives"]').exists()).toBe(false)
  })
})
