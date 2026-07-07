import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'

import en from '../i18n/locales/en.json'
import * as api from '../services/api'
import type { TopicConversationArcResponse } from '../services/types'
import TopicConversationArc from './TopicConversationArc.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function mountIt(id: string) {
  return mount(TopicConversationArc, { props: { id }, global: { plugins: [i18n] } })
}

const RESP: TopicConversationArcResponse = {
  topic_id: 'topic:ai',
  weeks: [
    { week: '2024-W03', volume: 3, negative: 1, neutral: 1, positive: 1, avg_compound: 0.1 },
    { week: '2024-W04', volume: 1, negative: 0, neutral: 0, positive: 1, avg_compound: 0.6 },
  ],
}

afterEach(() => vi.restoreAllMocks())

describe('TopicConversationArc (consumer)', () => {
  it('renders a weekly bar per arc bucket', async () => {
    vi.spyOn(api, 'getTopicConversationArc').mockResolvedValue(RESP)
    const w = mountIt('topic:ai')
    await flushPromises()
    expect(w.find('[data-testid="topic-conversation-arc"]').exists()).toBe(true)
    expect(w.find('[data-testid="tca-bar-2024-W03"]').exists()).toBe(true)
    expect(w.find('[data-testid="tca-bar-2024-W04"]').exists()).toBe(true)
    expect(w.text()).toContain('4 insights') // total volume
  })

  it('renders nothing when the topic has no dated insights', async () => {
    vi.spyOn(api, 'getTopicConversationArc').mockResolvedValue({ topic_id: 'topic:x', weeks: [] })
    const w = mountIt('topic:x')
    await flushPromises()
    expect(w.find('[data-testid="topic-conversation-arc"]').exists()).toBe(false)
  })

  it('degrades gracefully on fetch error', async () => {
    vi.spyOn(api, 'getTopicConversationArc').mockRejectedValue(new Error('boom'))
    const w = mountIt('topic:ai')
    await flushPromises()
    expect(w.find('[data-testid="topic-conversation-arc"]').exists()).toBe(false)
  })
})
