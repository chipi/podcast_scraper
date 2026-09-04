import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createRouter, createWebHistory } from 'vue-router'
import en from '../i18n/locales/en.json'
import * as api from '../services/api'
import type { RecapResponse } from '../services/types'
import ListeningRecap from './ListeningRecap.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'home', component: { template: '<div/>' } },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
  ],
})

function recap(over: Partial<RecapResponse> = {}): RecapResponse {
  return {
    window: 'week',
    from_day: '2026-08-28',
    to_day: '2026-09-03',
    listening_seconds: 8_640,
    by_day: {
      '2026-08-28': 0,
      '2026-08-29': 0,
      '2026-08-30': 1_200,
      '2026-08-31': 0,
      '2026-09-01': 3_600,
      '2026-09-02': 0,
      '2026-09-03': 3_840,
    },
    episodes_started: 9,
    distinct_episodes: 6,
    top_episodes: [],
    episodes_finished: 4,
    topics: [
      { token: 'topic:indexing', label: 'Index investing', episodes: 5, delta: 2, is_new: false },
    ],
    people: [{ token: 'person:cho', label: 'Daniel Cho', episodes: 2, delta: 0, is_new: false }],
    top_by_strength: [],
    best_line: null,
    days_recorded: 3,
    days_in_window: 7,
    coverage_from: '2026-08-30',
    first_listened_at: 1788000000,
    ...over,
  }
}

const mountRecap = () => mount(ListeningRecap, { global: { plugins: [i18n, router] } })

beforeEach(() => setActivePinia(createPinia()))
afterEach(() => vi.restoreAllMocks())

describe('ListeningRecap', () => {
  it('shows time actually listened, not a lifetime position sum', async () => {
    vi.spyOn(api, 'getRecap').mockResolvedValue(recap())
    const w = mountRecap()
    await flushPromises()
    expect(w.text()).toContain('2.4h') // 8640s
    expect(w.text()).toContain('Listened')
    expect(w.text()).toContain('6') // distinct episodes
    expect(w.text()).toContain('4') // finished
  })

  it('ALWAYS states its coverage while the window is only partly recorded', async () => {
    // Recording started with Phase 0, so a total can be real and still cover three days of seven.
    // Showing it silently would repeat, in a new form, the untruth this panel replaced.
    vi.spyOn(api, 'getRecap').mockResolvedValue(recap())
    const w = mountRecap()
    await flushPromises()
    expect(w.text()).toContain('Recorded 3 of 7 days')
  })

  it('drops the caveat once the window is fully recorded', async () => {
    vi.spyOn(api, 'getRecap').mockResolvedValue(recap({ days_recorded: 7 }))
    const w = mountRecap()
    await flushPromises()
    expect(w.text()).not.toContain('Recorded')
  })

  it('renders one bar per day in the window, so a chart has no holes', async () => {
    vi.spyOn(api, 'getRecap').mockResolvedValue(recap())
    const w = mountRecap()
    await flushPromises()
    expect(w.findAll('[role="img"] > div')).toHaveLength(7)
  })

  it('lists what kept coming up, topics and people together', async () => {
    vi.spyOn(api, 'getRecap').mockResolvedValue(recap())
    const w = mountRecap()
    await flushPromises()
    expect(w.text()).toContain('Index investing')
    expect(w.text()).toContain('Daniel Cho')
  })

  it('links a saved line to the MOMENT it came from', async () => {
    vi.spyOn(api, 'getRecap').mockResolvedValue(
      recap({
        best_line: {
          quote_text: 'the stable anchor is the timestamp',
          episode_slug: 'p06-721',
          start_ms: 42_000,
          created_at: 1788429600,
        },
      }),
    )
    const w = mountRecap()
    await flushPromises()
    expect(w.text()).toContain('the stable anchor is the timestamp')
    // Opening at the beginning would lose the reason the line was worth showing.
    expect(w.find('a[href*="/episode/p06-721"]').attributes('href')).toContain('t=42')
  })

  it('renders NOTHING when there is nothing recorded — an empty recap is worse than none', async () => {
    vi.spyOn(api, 'getRecap').mockResolvedValue(
      recap({ days_recorded: 0, episodes_started: 0, listening_seconds: 0 }),
    )
    const w = mountRecap()
    await flushPromises()
    expect(w.find('section').exists()).toBe(false)
  })

  it('renders nothing rather than an error when the request fails', async () => {
    // It sits on Profile; a recap must never break the page it is a panel on.
    vi.spyOn(api, 'getRecap').mockResolvedValue(null)
    const w = mountRecap()
    await flushPromises()
    expect(w.find('section').exists()).toBe(false)
  })

  it('refetches when the window changes', async () => {
    const spy = vi.spyOn(api, 'getRecap').mockResolvedValue(recap())
    const w = mountRecap()
    await flushPromises()
    await w.findAll('button')[1].trigger('click')
    await flushPromises()
    expect(spy).toHaveBeenLastCalledWith('month')
  })
})

/**
 * Trend arrows (#1923). A list that reads the same every week is a list nobody looks at twice —
 * the movement is the point, and it is what the recorded exposure log buys.
 */
describe('what changed', () => {
  it('shows the direction and size of a move', async () => {
    vi.spyOn(api, 'getRecap').mockResolvedValue(recap())
    const w = mountRecap()
    await flushPromises()
    expect(w.text()).toContain('↑2')
  })

  it('says "new" rather than a number for something that was absent', async () => {
    // A different fact, not a bigger number.
    vi.spyOn(api, 'getRecap').mockResolvedValue(
      recap({
        topics: [
          { token: 'topic:systems', label: 'Systems', episodes: 3, delta: 3, is_new: true },
        ],
      }),
    )
    const w = mountRecap()
    await flushPromises()
    expect(w.text()).toContain('new')
    expect(w.text()).not.toContain('↑3')
  })

  it('shows a fall too — a recap that only ever goes up is flattery', async () => {
    vi.spyOn(api, 'getRecap').mockResolvedValue(
      recap({
        topics: [{ token: 'topic:bonds', label: 'Bonds', episodes: 1, delta: -3, is_new: false }],
      }),
    )
    const w = mountRecap()
    await flushPromises()
    expect(w.text()).toContain('↓3')
  })

  it('renders NO marker when nothing moved — an "unchanged" arrow is noise on every chip', async () => {
    vi.spyOn(api, 'getRecap').mockResolvedValue(
      recap({
        topics: [{ token: 'topic:x', label: 'Steady', episodes: 4, delta: 0, is_new: false }],
        people: [],
      }),
    )
    const w = mountRecap()
    await flushPromises()
    expect(w.text()).not.toContain('↑')
    expect(w.text()).not.toContain('↓')
  })
})
