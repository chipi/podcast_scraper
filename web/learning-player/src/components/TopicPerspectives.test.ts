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

  it('threads the corpus scope through to the API (#1149)', async () => {
    const spy = vi.spyOn(api, 'getTopicPerspectives').mockResolvedValue(RESP)
    mount(TopicPerspectives, { props: { id: 'topic:ai', scope: 'mine' }, global: { plugins: [i18n] } })
    await flushPromises()
    expect(spy).toHaveBeenCalledWith('topic:ai', 'mine')
  })

  it('refetches when the scope prop changes (#1149)', async () => {
    const spy = vi.spyOn(api, 'getTopicPerspectives').mockResolvedValue(RESP)
    const w = mount(TopicPerspectives, {
      props: { id: 'topic:ai', scope: 'all' },
      global: { plugins: [i18n] },
    })
    await flushPromises()
    expect(spy).toHaveBeenLastCalledWith('topic:ai', 'all')
    await w.setProps({ scope: 'mine' })
    await flushPromises()
    expect(spy).toHaveBeenLastCalledWith('topic:ai', 'mine')
  })
})

describe('TopicPerspectives — a failed load must not look like an empty topic', () => {
  afterEach(() => vi.restoreAllMocks())

  it('retries once before giving up, so a single blip does not blank the section', async () => {
    // `serve` is one process: a concurrent search is enough to time this request out. The old
    // code caught that into an empty array, and the section is v-if'd on being non-empty — so a
    // transient failure rendered as "nobody had a perspective on this topic", permanently.
    const spy = vi
      .spyOn(api, 'getTopicPerspectives')
      .mockRejectedValueOnce(new Error('timeout'))
      .mockResolvedValueOnce(RESP)

    const w = mountIt('topic:ai')
    await vi.waitFor(() => expect(spy).toHaveBeenCalledTimes(2), { timeout: 3000 })
    await flushPromises()

    expect(w.find('[data-testid="topic-perspectives"]').exists()).toBe(true)
    expect(w.find('[data-testid="section-error"]').exists()).toBe(false)
    expect(w.text()).toContain('Jack Clark')
  })

  it('surfaces an error with a retry when both attempts fail', async () => {
    vi.spyOn(api, 'getTopicPerspectives').mockRejectedValue(new Error('down'))
    const w = mountIt('topic:ai')
    await vi.waitFor(
      () => expect(w.find('[data-testid="section-error"]').exists()).toBe(true),
      { timeout: 3000 },
    )
    // The reader is told it failed rather than being shown a confident, silent nothing.
    expect(w.text()).toContain(en.section.error)
    expect(w.find('[data-testid="section-retry"]').exists()).toBe(true)
    expect(w.find('[data-testid="topic-perspectives"]').exists()).toBe(false)
  })

  it('the retry button recovers the section', async () => {
    const spy = vi.spyOn(api, 'getTopicPerspectives').mockRejectedValue(new Error('down'))
    const w = mountIt('topic:ai')
    await vi.waitFor(
      () => expect(w.find('[data-testid="section-retry"]').exists()).toBe(true),
      { timeout: 3000 },
    )

    spy.mockResolvedValue(RESP)
    await w.find('[data-testid="section-retry"]').trigger('click')
    await vi.waitFor(
      () => expect(w.find('[data-testid="topic-perspectives"]').exists()).toBe(true),
      { timeout: 3000 },
    )
    expect(w.find('[data-testid="section-error"]').exists()).toBe(false)
  })

  it('a genuinely empty topic still renders nothing at all', async () => {
    vi.spyOn(api, 'getTopicPerspectives').mockResolvedValue({
      ...RESP, perspective_count: 0, perspectives: [],
    })
    const w = mountIt('topic:quiet')
    await flushPromises()
    expect(w.find('[data-testid="topic-perspectives"]').exists()).toBe(false)
    expect(w.find('[data-testid="section-error"]').exists()).toBe(false)
  })
})
