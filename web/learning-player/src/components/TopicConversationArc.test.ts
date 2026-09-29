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

  it('SAYS it is loading instead of appearing out of nowhere', async () => {
    /*
     * The defect the operator hit (2026-09-27): "conversation over time showed up later after I
     * clicked some buttons and was not there when I opened the page." It was the wait, not the
     * clicking — this rendered nothing at all until the fetch resolved, and the endpoint is a
     * corpus-wide uncached scan, so it popped in seconds later with no warning.
     *
     * Every other test in this file awaits `flushPromises()` FIRST, so all of them only ever
     * observed the resolved state. That is why an empty loading state shipped: the one moment the
     * user actually complained about was the one moment no test looked at.
     */
    let resolve!: (r: TopicConversationArcResponse) => void
    vi.spyOn(api, 'getTopicConversationArc').mockReturnValue(
      new Promise<TopicConversationArcResponse>((r) => {
        resolve = r
      }),
    )
    const w = mountIt('topic:ai')
    await Promise.resolve() // let the watcher fire, but do NOT settle the fetch

    expect(
      w.find('[data-testid="topic-arc-loading"]').exists(),
      'an in-flight arc must announce itself, or it reads as a section that appears at random',
    ).toBe(true)
    expect(w.text()).toContain(en.ec.conversationArc) // named while loading, not a bare box
    expect(w.find('[data-testid="topic-conversation-arc"]').exists()).toBe(false)

    resolve(RESP)
    await flushPromises()
    // ...and it hands over cleanly: the placeholder goes, the chart arrives, heading unchanged.
    expect(w.find('[data-testid="topic-arc-loading"]').exists()).toBe(false)
    expect(w.find('[data-testid="topic-conversation-arc"]').exists()).toBe(true)
  })

  it('bars grow to fill the box when a topic has only a few weeks', async () => {
    // Fixed 8px bars left-aligned in a full-width scroller made a two-week topic look like a chart
    // that had failed to load. `flex: 1 1 8px` with a cap lets sparse spread and keeps dense
    // scrolling; the cap is what stops three weeks becoming three slabs.
    vi.spyOn(api, 'getTopicConversationArc').mockResolvedValue(RESP)
    const w = mountIt('topic:ai')
    await flushPromises()
    const bar = w.get('[data-testid="tca-bar-2024-W03"]')
    // Vue expands the `flex` shorthand into longhands, so assert what is actually emitted rather
    // than the source spelling — matching on `flex: 1 1 8px` would fail on a correct element.
    const style = bar.attributes('style') ?? ''
    expect(style, 'bars must be able to grow').toContain('flex-grow: 1')
    expect(style, 'and to give space back').toContain('flex-shrink: 1')
    expect(style, 'from an 8px basis, so a dense topic still scrolls').toContain('flex-basis: 8px')
    expect(style, 'and must stay readable when there are many').toContain('min-width: 8px')
    expect(style, 'and must not become slabs when there are few').toContain('max-width: 28px')
    expect(bar.classes(), 'shrink-0 would defeat the growth').not.toContain('shrink-0')
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

describe('a failed section says WHAT failed (#2004 item 12)', () => {
  it('keeps its heading visible in the error state', async () => {
    vi.spyOn(api, 'getTopicConversationArc').mockRejectedValue(new Error('down'))
    const w = mountIt('topic:ai')
    await vi.waitFor(() => expect(w.find('[data-testid="topic-arc-error"]').exists()).toBe(true), {
      timeout: 3000,
    })
    expect(w.text()).toContain(en.ec.conversationArc)
    expect(w.text()).toContain(en.section.error)
  })
})
