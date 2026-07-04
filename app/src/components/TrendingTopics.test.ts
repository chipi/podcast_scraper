import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import { useAuthStore } from '../stores/auth'
import { useInterestsStore } from '../stores/interests'
import type { CorpusEnrichmentSignals } from '../services/types'
import TrendingTopics from './TrendingTopics.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountIt = (setup?: () => void) => {
  setActivePinia(createPinia()) // fresh pinia per mount; no user → signed out unless setup says so
  setup?.()
  return mount(TrendingTopics, { global: { plugins: [i18n] } })
}

const VELOCITY: CorpusEnrichmentSignals['temporal_velocity'] = {
  window_months: ['2026-01', '2026-02', '2026-03'],
  topics: [
    { topic_id: 'topic:ai', topic_label: 'ai', velocity_last_over_6mo: 2, total: 10, monthly_counts: { '2026-01': 1, '2026-02': 3, '2026-03': 6 } },
    { topic_id: 'topic:policy', topic_label: 'foreign policy', velocity_last_over_6mo: 4, total: 3, monthly_counts: { '2026-03': 3 } },
    { topic_id: 'topic:steady', topic_label: 'steady', velocity_last_over_6mo: 1, total: 20, monthly_counts: { '2026-02': 10 } }, // not rising
    { topic_id: 'topic:noise', topic_label: 'noise', velocity_last_over_6mo: 5, total: 1, monthly_counts: {} }, // below floor
  ],
}
const withVelocity = (tv = VELOCITY) =>
  vi.spyOn(api, 'getCorpusEnrichment').mockResolvedValue({ temporal_velocity: tv })

afterEach(() => vi.restoreAllMocks())

describe('TrendingTopics container', () => {
  it('defaults to the Pills view with rising topics sorted by velocity', async () => {
    withVelocity()
    const w = mountIt()
    await flushPromises()
    const chips = w.findAll('[data-testid="trend-chip"]')
    // policy (4x) before ai (2x); steady + noise excluded.
    expect(chips).toHaveLength(2)
    expect(chips[0].text()).toContain('foreign policy')
    expect(chips[0].text()).toContain('4×')
  })

  it('emits open with the topic id from a pill', async () => {
    withVelocity()
    const w = mountIt()
    await flushPromises()
    // The chip is a container; its first button opens the topic (the second, when present, follows).
    await w.findAll('[data-testid="trend-chip"]')[0].get('button').trigger('click')
    expect(w.emitted('open')![0]).toEqual(['topic:policy'])
  })

  it('signed out: no follow buttons on the pills (#12)', async () => {
    withVelocity()
    const w = mountIt()
    await flushPromises()
    expect(w.findAll('[data-testid="trend-chip-follow"]')).toHaveLength(0)
  })

  it('signed in: a follow button adds the trending topic to interests (#12)', async () => {
    withVelocity()
    let interests!: ReturnType<typeof useInterestsStore>
    const w = mountIt(() => {
      useAuthStore().user = { user_id: 'u_1', email: 'd@l', name: 'Dev' }
      interests = useInterestsStore()
      interests.loaded = true // ensureLoaded() becomes a no-op (no API call)
      vi.spyOn(interests, 'toggle').mockResolvedValue()
    })
    await flushPromises()
    const followBtns = w.findAll('[data-testid="trend-chip-follow"]')
    expect(followBtns).toHaveLength(2) // one per rising topic
    await followBtns[0].trigger('click')
    expect(interests.toggle).toHaveBeenCalledWith('topic:policy')
  })

  it('signed in: a followed topic shows the following state (#12)', async () => {
    withVelocity()
    const w = mountIt(() => {
      useAuthStore().user = { user_id: 'u_1', email: 'd@l', name: 'Dev' }
      const interests = useInterestsStore()
      interests.ids = ['topic:policy']
      interests.loaded = true
    })
    await flushPromises()
    const first = w.findAll('[data-testid="trend-chip-follow"]')[0]
    expect(first.attributes('aria-pressed')).toBe('true')
    expect(first.text()).toBe('✓')
  })

  it('switches to the Sparklines view (rows with mini series)', async () => {
    withVelocity()
    const w = mountIt()
    await flushPromises()
    await w.get('[data-testid="trend-view-sparks"]').trigger('click')
    expect(w.findAll('[data-testid="trend-spark-row"]')).toHaveLength(2)
  })

  it('switches to the Over-time (stream) view with one band per rising topic', async () => {
    withVelocity()
    const w = mountIt()
    await flushPromises()
    await w.get('[data-testid="trend-view-stream"]').trigger('click')
    expect(w.find('[data-testid="trend-stream"]').exists()).toBe(true)
    expect(w.findAll('[data-testid="trend-stream-band"]')).toHaveLength(2)
    // A legend chip opens the topic card.
    await w.findAll('[data-testid="trend-stream-legend"]')[0].trigger('click')
    expect(w.emitted('open')).toBeTruthy()
  })

  it('switches to the Momentum view with one point per rising topic', async () => {
    withVelocity()
    const w = mountIt()
    await flushPromises()
    await w.get('[data-testid="trend-view-momentum"]').trigger('click')
    expect(w.find('[data-testid="trend-momentum"]').exists()).toBe(true)
    const pts = w.findAll('[data-testid="trend-momentum-point"]')
    expect(pts).toHaveLength(2)
    await pts[0].trigger('click')
    expect(w.emitted('open')![0]).toEqual(['topic:policy'])
  })

  it('renders nothing when no topic is rising', async () => {
    withVelocity({
      window_months: ['2026-01'],
      topics: [{ topic_id: 'topic:flat', topic_label: 'flat', velocity_last_over_6mo: 0.9, total: 50, monthly_counts: { '2026-01': 5 } }],
    })
    const w = mountIt()
    await flushPromises()
    expect(w.find('[data-testid="home-trending"]').exists()).toBe(false)
  })

  it('renders nothing when the velocity enricher is absent', async () => {
    vi.spyOn(api, 'getCorpusEnrichment').mockResolvedValue({})
    const w = mountIt()
    await flushPromises()
    expect(w.find('[data-testid="home-trending"]').exists()).toBe(false)
  })
})
