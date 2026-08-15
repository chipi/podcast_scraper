/**
 * The band answers "what this show's about", so the thing under test is whether it can tell a
 * show's signature topic apart from corpus-wide wallpaper — the failure that motivated `lift`.
 */
import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { PodcastSignals } from '../services/types'
import PodcastSignalsBand from './PodcastSignalsBand.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

type Topic = PodcastSignals['top_topics'][number]

function topic(over: Partial<Topic> & Pick<Topic, 'topic_id' | 'label'>): Topic {
  return {
    episode_count: 4,
    velocity: null,
    corpus_episode_count: null,
    corpus_episode_total: 36,
    lift: null,
    ...over,
  }
}

function signals(over: Partial<PodcastSignals> = {}): PodcastSignals {
  return {
    feed_id: 'p01',
    episode_count: 4,
    top_topics: [],
    key_people: [],
    recurring_guests: [],
    dominant_themes: [],
    trending_topics: [],
    ...over,
  }
}

async function mountBand(s: PodcastSignals) {
  vi.spyOn(api, 'getPodcastSignals').mockResolvedValue(s)
  const w = mount(PodcastSignalsBand, {
    props: { feedId: s.feed_id },
    global: { plugins: [i18n] },
  })
  await flushPromises()
  return w
}

const labels = (w: Awaited<ReturnType<typeof mountBand>>, testId: string) =>
  w.findAll(`[data-testid="${testId}"]`).map((el) => el.text().trim())

afterEach(() => vi.restoreAllMocks())

describe('PodcastSignalsBand — distinctiveness', () => {
  it('promotes the topic the show is unusually focused on, not the most-covered one', async () => {
    // Both are in 4/4 episodes, so coverage cannot separate them — only lift can. This mirrors
    // the real validation corpus, where every show covers "expert interviews" in every episode.
    const w = await mountBand(
      signals({
        top_topics: [
          topic({ topic_id: 't:ei', label: 'expert interviews', corpus_episode_count: 36, lift: 1 }),
          topic({ topic_id: 't:ll', label: 'lifelong learning', corpus_episode_count: 36, lift: 1 }),
          topic({ topic_id: 't:rm', label: 'risk management', corpus_episode_count: 28, lift: 1.29 }),
          topic({ topic_id: 't:es', label: 'endurance sport', corpus_episode_count: 4, lift: 9 }),
        ],
      }),
    )

    expect(labels(w, 'ps-distinctive-topic')).toEqual(['endurance sport'])
    // 1.29 is above the base rate but below the bar — it stays context, it is not a claim.
    expect(labels(w, 'ps-topic')).toEqual([
      'expert interviews',
      'lifelong learning',
      'risk management',
    ])
    expect(w.get('[data-testid="ps-topics-heading"]').text()).toBe(en.podcast.sigAlsoCovers)
  })

  it('orders several distinctive topics strongest-first and explains the multiplier', async () => {
    const w = await mountBand(
      signals({
        top_topics: [
          topic({ topic_id: 't:a', label: 'mild', corpus_episode_count: 20, lift: 1.8 }),
          topic({ topic_id: 't:b', label: 'strong', corpus_episode_count: 4, lift: 9 }),
        ],
      }),
    )

    expect(labels(w, 'ps-distinctive-topic')).toEqual(['strong', 'mild'])
    const [strong, mild] = w.findAll('[data-testid="ps-distinctive-topic"]')
    // "9", not "9.0" — and the fractional one keeps its digit.
    expect(strong.attributes('title')).toContain('9×')
    expect(mild.attributes('title')).toContain('1.8×')
  })

  it('never promotes a one-off mention, however high its lift', async () => {
    // 1 of 4 episodes against a rare corpus topic scores 4.5×, but one passing mention says
    // nothing about the show — it must not be presented as the show's identity.
    const w = await mountBand(
      signals({
        top_topics: [
          topic({ topic_id: 't:one', label: 'one-off', episode_count: 1, corpus_episode_count: 2, lift: 4.5 }),
        ],
      }),
    )

    expect(labels(w, 'ps-distinctive-topic')).toEqual([])
    expect(labels(w, 'ps-topic')).toEqual(['one-off'])
  })

  it('falls back to the coverage claim when no lift is known', async () => {
    // No corpus base rate (the co-occurrence envelope is absent) ⇒ lift null everywhere. The band
    // must degrade to what it said before, not guess at distinctiveness.
    const w = await mountBand(
      signals({
        top_topics: [
          topic({ topic_id: 't:a', label: 'alpha' }),
          topic({ topic_id: 't:b', label: 'beta' }),
        ],
      }),
    )

    expect(w.find('[data-testid="ps-distinctive-heading"]').exists()).toBe(false)
    expect(labels(w, 'ps-topic')).toEqual(['alpha', 'beta'])
    expect(w.get('[data-testid="ps-topics-heading"]').text()).toBe(en.podcast.sigCoverageAll)
  })
})

describe('PodcastSignalsBand — a failed load must not delete the band', () => {
  it('keeps the band with a retry when signals fail', async () => {
    // The band used to catch into null and hide, so an API failure removed a whole titled
    // section of the show page — indistinguishable from a show with nothing to say (#1591).
    vi.spyOn(api, 'getPodcastSignals').mockRejectedValue(new Error('down'))
    const w = mount(PodcastSignalsBand, { props: { feedId: 'p01' }, global: { plugins: [i18n] } })
    await flushPromises()

    expect(w.find('[data-testid="podcast-signals-error"]').exists()).toBe(true)
    expect(w.find('[data-testid="section-retry"]').exists()).toBe(true)
    expect(w.text()).toContain(en.podcast.about)
  })

  it('recovers on retry', async () => {
    const spy = vi.spyOn(api, 'getPodcastSignals').mockRejectedValue(new Error('down'))
    const w = mount(PodcastSignalsBand, { props: { feedId: 'p01' }, global: { plugins: [i18n] } })
    await flushPromises()

    spy.mockResolvedValue(
      signals({ top_topics: [topic({ topic_id: 't:a', label: 'alpha' })] }),
    )
    await w.find('[data-testid="section-retry"]').trigger('click')
    await flushPromises()

    expect(w.find('[data-testid="podcast-signals"]').exists()).toBe(true)
    expect(w.find('[data-testid="podcast-signals-error"]').exists()).toBe(false)
  })

  it('a show with genuinely no signals still renders nothing', async () => {
    vi.spyOn(api, 'getPodcastSignals').mockResolvedValue(signals())
    const w = mount(PodcastSignalsBand, { props: { feedId: 'p01' }, global: { plugins: [i18n] } })
    await flushPromises()
    expect(w.find('[data-testid="podcast-signals"]').exists()).toBe(false)
    expect(w.find('[data-testid="podcast-signals-error"]').exists()).toBe(false)
  })
})
