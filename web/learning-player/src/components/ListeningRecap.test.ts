import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createRouter, createWebHistory } from 'vue-router'
import en from '../i18n/locales/en.json'
import * as api from '../services/api'
import type { RecapResponse, RecapSubject } from '../services/types'
import ListeningRecap from './ListeningRecap.vue'
import { useAuthStore } from '../stores/auth'
import { bumpIdentityEpoch } from '../services/identity'
import type { Me } from '../services/types'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'home', component: { template: '<div/>' } },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
  ],
})

function subj(
  id: string,
  kind: RecapSubject['kind'],
  label: string,
  over: Partial<RecapSubject> = {},
): RecapSubject {
  return { id, kind, label, episodes: 0, moments: 0, learning_days: 0, delta: 0, is_new: false, ...over }
}

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
    subjects: [
      subj('topic:indexing', 'topic', 'Index investing', { episodes: 5, delta: 2 }),
      subj('person:cho', 'person', 'Daniel Cho', { episodes: 2 }),
    ],
    ...over,
  }
}

const mountRecap = () => mount(ListeningRecap, { global: { plugins: [i18n, router] } })

beforeEach(() => setActivePinia(createPinia()))
afterEach(() => vi.restoreAllMocks())

describe('ListeningRecap', () => {
  it("never shows the previous account's recap after an account switch (operator 2026-10-07)", async () => {
    // A brand-new Apple account showed the Google account's half hour, six episodes and saved line:
    // Profile is kept alive, the recap loaded once on mount and nothing reloaded it.
    const auth = useAuthStore()
    auth.user = { user_id: 'u_a', email: 'a@x.com', name: 'A' } as unknown as Me
    const spy = vi.spyOn(api, 'getRecap').mockResolvedValue(recap())
    const w = mountRecap()
    await flushPromises()
    expect(w.text()).toContain('2.4h')

    spy.mockResolvedValue(recap({ listening_seconds: 0, days_recorded: 0, episodes_started: 0 }))
    bumpIdentityEpoch() // what App.vue does on every identity change
    auth.user = { user_id: 'u_b', email: 'b@x.com', name: 'B' } as unknown as Me
    await flushPromises()
    expect(spy).toHaveBeenCalledTimes(2)
    expect(w.text()).not.toContain('2.4h')
  })

  it("drops a recap that resolves after the account changed", async () => {
    const auth = useAuthStore()
    auth.user = { user_id: 'u_a', email: 'a@x.com', name: 'A' } as unknown as Me
    let resolveA: (r: RecapResponse) => void = () => {}
    const spy = vi.spyOn(api, 'getRecap').mockImplementationOnce(
      () => new Promise<RecapResponse>((res) => { resolveA = res }),
    )
    spy.mockResolvedValue(recap({ listening_seconds: 0, days_recorded: 0, episodes_started: 0 }))
    const w = mountRecap()
    bumpIdentityEpoch()
    auth.user = { user_id: 'u_b', email: 'b@x.com', name: 'B' } as unknown as Me
    await flushPromises()
    resolveA(recap()) // A's late answer
    await flushPromises()
    expect(w.text()).not.toContain('2.4h')
  })

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

  it('lists what the window was about in one row, each pill naming its kind', async () => {
    vi.spyOn(api, 'getRecap').mockResolvedValue(
      recap({
        subjects: [
          subj('topic:indexing', 'topic', 'Index investing', { episodes: 5, moments: 2 }),
          subj('thc:risk', 'storyline', 'Managing risk', { learning_days: 3 }),
        ],
      }),
    )
    const w = mountRecap()
    await flushPromises()
    const pills = w.findAll('[data-testid="recap-subject"]')
    expect(pills.map((p) => p.find('[data-testid="kind-word"]').text().trim())).toEqual(['Topic', 'Storyline'])
    // The pill shows the total; its spoken name says what the total is made of.
    expect(pills[0].text()).toContain('Index investing7')
    expect(pills[0].attributes('aria-label')).toBe('Topic Index investing: 5 episodes, 2 moments')
    expect(pills[1].attributes('aria-label')).toBe('Storyline Managing risk: 3 learning days')
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
    // The shared jump control, worded like every other one (operator 2026-10-05).
    expect(w.get('[data-testid="play-from"]').text()).toBe('▶ Play from 0:42')
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
        subjects: [subj('topic:systems', 'topic', 'Systems', { episodes: 3, delta: 3, is_new: true })],
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
        subjects: [subj('topic:bonds', 'topic', 'Bonds', { episodes: 1, delta: -3 })],
      }),
    )
    const w = mountRecap()
    await flushPromises()
    expect(w.text()).toContain('↓3')
  })

  it('renders NO marker when nothing moved — an "unchanged" arrow is noise on every chip', async () => {
    vi.spyOn(api, 'getRecap').mockResolvedValue(
      recap({
        subjects: [subj('topic:x', 'topic', 'Steady', { episodes: 4 })],
      }),
    )
    const w = mountRecap()
    await flushPromises()
    expect(w.text()).not.toContain('↑')
    expect(w.text()).not.toContain('↓')
  })
})

/** Moments and learning beside the listening (operator 2026-10-12). */
describe('short listens and learning', () => {
  it('counts moments and learning days in their own tiles, and says minutes under an hour', async () => {
    vi.spyOn(api, 'getRecap').mockResolvedValue(
      recap({ listening_seconds: 170, moments_heard: 9, moments_seconds: 600, learning_days: 2, distinct_shows: 3 }),
    )
    const w = mountRecap()
    await flushPromises()
    expect(w.get('[data-testid="recap-listened"]').text()).toContain('3 min')
    expect(w.get('[data-testid="recap-moments"]').text()).toContain('9')
    expect(w.get('[data-testid="recap-moments"]').text()).toContain('10 min')
    expect(w.get('[data-testid="recap-learning"]').text()).toContain('2')
    expect(w.get('[data-testid="recap-episodes"]').text()).toContain('3 shows · 4 finished')
  })

  it('stacks a day of moments on the day chart, so it is not a blank bar', async () => {
    const r = recap({ moments_heard: 1, moments_by_day: { '2026-08-28': 300 } })
    vi.spyOn(api, 'getRecap').mockResolvedValue(r)
    const w = mountRecap()
    await flushPromises()
    const first = w.findAll('[data-testid="recap-day"]')[0]
    expect(first.attributes('style')).toContain('opacity: 1')
    expect(first.find('.bg-accent').exists()).toBe(true)
  })

  it('says how many moments led to the full episode, only once there were moments', async () => {
    vi.spyOn(api, 'getRecap').mockResolvedValue(recap({ moment_episodes: 5, moment_to_full: 2 }))
    const w = mountRecap()
    await flushPromises()
    expect(w.get('[data-testid="recap-moment-to-full"]').text()).toBe(
      '2 of 5 episodes you heard as moments, you went on to play in full',
    )
  })

  it('shows the running streaks from the all-time stats, and nothing for one not running', async () => {
    vi.spyOn(api, 'getRecap').mockResolvedValue(recap())
    const w = mount(ListeningRecap, {
      props: { stats: { day_streak: 4, learning_streak: 0 } as never },
      global: { plugins: [i18n, router] },
    })
    await flushPromises()
    expect(w.get('[data-testid="recap-streaks"]').text()).toBe('Day streak: 4 days')
  })

  it('shows for a week of only short listens', async () => {
    vi.spyOn(api, 'getRecap').mockResolvedValue(
      recap({ days_recorded: 0, episodes_started: 0, listening_seconds: 0, moments_heard: 3 }),
    )
    const w = mountRecap()
    await flushPromises()
    expect(w.find('[data-testid="stats-recap"]').exists()).toBe(true)
  })
})
