import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import { useAuthStore } from '../stores/auth'
import { useInterestsStore } from '../stores/interests'
import type { TrendingTopicsResponse } from '../services/types'
import TrendingTopics from './TrendingTopics.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const mountIt = (setup?: () => void) => {
  setActivePinia(createPinia()) // fresh pinia per mount; no user → signed out unless setup says so
  setup?.()
  return mount(TrendingTopics, { global: { plugins: [i18n] } })
}

// The endpoint now does the rising filter / velocity-desc sort / top-N trim server-side (covered by
// the backend tests), so the client renders these rows as-is. `topic:policy` (4×) leads `topic:ai`
// (2×); the below-bar / sparse topics never reach the client.
const TRENDING: TrendingTopicsResponse = {
  has_velocity_data: true,
  window_months: ['2026-01', '2026-02', '2026-03'],
  topics: [
    { topic_id: 'topic:policy', topic_label: 'foreign policy', velocity_last_over_6mo: 4, total: 3, monthly_counts: { '2026-03': 3 } },
    { topic_id: 'topic:ai', topic_label: 'ai', velocity_last_over_6mo: 2, total: 10, monthly_counts: { '2026-01': 1, '2026-02': 3, '2026-03': 6 } },
  ],
  theme_clusters: [],
}
const withVelocity = (res = TRENDING) =>
  vi.spyOn(api, 'getTrendingTopics').mockResolvedValue(res)

// The rail defers its fetch until it scrolls into view via IntersectionObserver. jsdom/happy-dom
// never scrolls, so stub the observer to report "visible" immediately — these tests exercise the
// loaded state, not the lazy-load gate.
beforeEach(() => {
  vi.stubGlobal(
    'IntersectionObserver',
    class {
      cb: IntersectionObserverCallback
      constructor(cb: IntersectionObserverCallback) {
        this.cb = cb
      }
      observe(el: Element): void {
        this.cb(
          [{ isIntersecting: true, target: el } as IntersectionObserverEntry],
          this as unknown as IntersectionObserver,
        )
      }
      unobserve(): void {}
      disconnect(): void {}
      takeRecords(): IntersectionObserverEntry[] {
        return []
      }
    },
  )
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('TrendingTopics container', () => {
  it('defaults to the Sparklines view with rising topics sorted by velocity', async () => {
    withVelocity()
    const w = mountIt()
    await flushPromises()
    const rows = w.findAll('[data-testid="trend-spark-row"]')
    // policy (4x) before ai (2x); steady + noise excluded. No theme clusters here → velocity order.
    expect(rows).toHaveLength(2)
    expect(rows[0].text()).toContain('foreign policy')
    expect(rows[0].text()).toContain('4×')
  })

  it('emits open with the topic id from a sparkline row', async () => {
    withVelocity()
    const w = mountIt()
    await flushPromises()
    await w.findAll('[data-testid="trend-spark-row"]')[0].trigger('click')
    expect(w.emitted('open')![0]).toEqual(['topic:policy'])
  })

  it('signed out: no follow buttons on the rows (#12)', async () => {
    withVelocity()
    const w = mountIt()
    await flushPromises()
    expect(w.findAll('[data-testid="trend-spark-follow"]')).toHaveLength(0)
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
    const followBtns = w.findAll('[data-testid="trend-spark-follow"]')
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
    const first = w.findAll('[data-testid="trend-spark-follow"]')[0]
    expect(first.attributes('aria-pressed')).toBe('true')
    expect(first.text()).toBe('✓')
  })

  it('offers exactly one view — the A/B switcher is gone (#1589)', async () => {
    // This section shipped with a four-way view switcher whose own comment admitted it existed so
    // "the operator can flip between to decide what to keep". That decision was never made, so an
    // internal experiment reached users. Sparklines won; the control and the other three views are
    // deleted. A reintroduced switcher fails here.
    withVelocity()
    const w = mountIt()
    await flushPromises()
    expect(w.find('[role="tablist"]').exists()).toBe(false)
    expect(w.findAll('[data-testid="trend-spark-row"]')).toHaveLength(2)
  })

  it('stays on screen and says so when nothing clears the bar', async () => {
    // Changed deliberately: this used to hide. Hiding made the metric unobservable — the rail has
    // never once appeared on the validation corpus (nothing reaches 1.5x), so the fact that it
    // disagrees with the momentum rail beside it (0.86x vs 1.78x on the same topic) could not be
    // seen. A measured quiet is a result worth showing while both measures are being evaluated.
    // Enricher ran (has_velocity_data), but nothing cleared the bar server-side → topics empty.
    withVelocity({ has_velocity_data: true, window_months: ['2026-01'], topics: [], theme_clusters: [] })
    const w = mountIt()
    await flushPromises()
    expect(w.find('[data-testid="home-trending"]').exists()).toBe(true)
    expect(w.find('[data-testid="home-trending-quiet"]').exists()).toBe(true)
    // ...and it must not imply there is something to look at.
    expect(w.find('[data-testid="trending-spark-chip"]').exists()).toBe(false)
  })

  it('renders nothing when the velocity enricher is absent', async () => {
    // No enricher: has_velocity_data false, no rows → the whole section stays off.
    withVelocity({ has_velocity_data: false, window_months: [], topics: [], theme_clusters: [] })
    const w = mountIt()
    await flushPromises()
    expect(w.find('[data-testid="home-trending"]').exists()).toBe(false)
  })
})
