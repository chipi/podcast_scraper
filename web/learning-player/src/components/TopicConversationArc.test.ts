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

describe('TopicConversationArc — a failed load must not look like a topic with no arc', () => {
  it('shows an error with a retry instead of vanishing', async () => {
    vi.spyOn(api, 'getTopicConversationArc').mockRejectedValue(new Error('down'))
    const w = mountIt('topic:ai')
    await flushPromises()

    expect(w.find('[data-testid="section-error"]').exists()).toBe(true)
    expect(w.find('[data-testid="section-retry"]').exists()).toBe(true)
    expect(w.find('[data-testid="topic-conversation-arc"]').exists()).toBe(false)
  })

  it('recovers on retry', async () => {
    const spy = vi.spyOn(api, 'getTopicConversationArc').mockRejectedValue(new Error('down'))
    const w = mountIt('topic:ai')
    await flushPromises()

    spy.mockResolvedValue(RESP)
    await w.find('[data-testid="section-retry"]').trigger('click')
    await flushPromises()

    expect(w.find('[data-testid="topic-conversation-arc"]').exists()).toBe(true)
    expect(w.find('[data-testid="section-error"]').exists()).toBe(false)
  })

  it('a topic with no dated insights still renders nothing at all', async () => {
    vi.spyOn(api, 'getTopicConversationArc').mockResolvedValue({ topic_id: 'topic:ai', weeks: [] })
    const w = mountIt('topic:ai')
    await flushPromises()
    expect(w.find('[data-testid="topic-conversation-arc"]').exists()).toBe(false)
    expect(w.find('[data-testid="section-error"]').exists()).toBe(false)
  })

  it('"My corpus" scope is a deliberate empty, not an error', async () => {
    // The arc has no per-user cut, so it renders nothing under `mine` — that must not be
    // dressed up as a failure the reader can retry.
    const spy = vi.spyOn(api, 'getTopicConversationArc').mockResolvedValue(RESP)
    const w = mount(TopicConversationArc, {
      props: { id: 'topic:ai', scope: 'mine' as const },
      global: { plugins: [i18n] },
    })
    await flushPromises()
    expect(spy).not.toHaveBeenCalled()
    expect(w.find('[data-testid="section-error"]').exists()).toBe(false)
    expect(w.find('[data-testid="topic-conversation-arc"]').exists()).toBe(false)
  })
})
