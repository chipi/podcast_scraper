import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createRouter, createWebHistory } from 'vue-router'
import en from '../i18n/locales/en.json'
import * as api from '../services/api'
import type { RecapResponse } from '../services/types'
import { useAuthStore } from '../stores/auth'
import RecapPrompt from './RecapPrompt.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'home', component: { template: '<div/>' } },
    { path: '/profile', name: 'profile', component: { template: '<div/>' } },
  ],
})

function recap(over: Partial<RecapResponse> = {}): RecapResponse {
  return {
    window: 'week',
    from_day: '2026-08-28',
    to_day: '2026-09-03',
    listening_seconds: 8_640,
    by_day: {},
    episodes_started: 9,
    distinct_episodes: 6,
    top_episodes: [],
    episodes_finished: 4,
    topics: [{ token: 'topic:indexing', label: 'Index investing', episodes: 5, delta: 2, is_new: false }],
    people: [],
    top_by_strength: [],
    best_line: null,
    days_recorded: 3,
    days_in_window: 7,
    coverage_from: '2026-08-30',
    first_listened_at: 1788000000,
    ...over,
  }
}

function signIn(): void {
  useAuthStore().user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
}
const mountPrompt = () => mount(RecapPrompt, { global: { plugins: [i18n, router] } })

beforeEach(() => setActivePinia(createPinia()))
afterEach(() => vi.restoreAllMocks())

describe('RecapPrompt', () => {
  it('says the least it can and points at Profile', async () => {
    // Profile is the permanent home for the recap; this is the reminder that it exists.
    signIn()
    vi.spyOn(api, 'getRecap').mockResolvedValue(recap())
    const w = mountPrompt()
    await flushPromises()
    expect(w.text()).toContain('2.4h')
    expect(w.text()).toContain('6 episodes')
    expect(w.text()).toContain('Index investing')
    // Deep-links to the Stats tab (operator 2026-09-14) — the prompt IS "see my stats".
    expect(w.find('a').attributes('href')).toBe('/profile?tab=stats')
  })

  it('renders nothing when nothing was listened to', async () => {
    // A row saying "0h · 0 episodes" takes space to tell you what you already knew.
    signIn()
    vi.spyOn(api, 'getRecap').mockResolvedValue(recap({ listening_seconds: 0 }))
    const w = mountPrompt()
    await flushPromises()
    expect(w.find('a').exists()).toBe(false)
  })

  it('does not even ask when signed out', async () => {
    const spy = vi.spyOn(api, 'getRecap')
    const w = mountPrompt()
    await flushPromises()
    expect(spy).not.toHaveBeenCalled()
    expect(w.find('a').exists()).toBe(false)
  })

  it('stays silent when the request fails — Home must not break for a prompt', async () => {
    signIn()
    vi.spyOn(api, 'getRecap').mockResolvedValue(null)
    const w = mountPrompt()
    await flushPromises()
    expect(w.find('a').exists()).toBe(false)
  })

  it('drops the theme clause rather than the row when there is no theme', async () => {
    signIn()
    vi.spyOn(api, 'getRecap').mockResolvedValue(recap({ topics: [] }))
    const w = mountPrompt()
    await flushPromises()
    expect(w.find('a').exists()).toBe(true)
    expect(w.text()).toContain('2.4h')
    // No topics → no second line at all, rather than an empty one.
    expect(w.find('[data-testid="recap-prompt-topics"]').exists()).toBe(false)
  })

  /**
   * Three topics with movement (operator 2026-09-20).
   *
   * The endpoint already returned five topics carrying `delta` / `is_new`; the card rendered one
   * label and no movement, so the rest was fetched on every Home load and dropped.
   */
  it('shows the top three topics, each with how it moved', async () => {
    signIn()
    vi.spyOn(api, 'getRecap').mockResolvedValue(
      recap({
        topics: [
          { token: 'topic:a', label: 'Index investing', episodes: 5, delta: 2, is_new: false },
          { token: 'topic:b', label: 'Sleep science', episodes: 3, delta: 0, is_new: false },
          { token: 'topic:c', label: 'Fusion power', episodes: 2, delta: 0, is_new: true },
          { token: 'topic:d', label: 'Should render too', episodes: 1, delta: -1, is_new: false },
          { token: 'topic:e', label: 'Should not render', episodes: 1, delta: -1, is_new: false },
        ],
      }),
    )
    const w = mountPrompt()
    await flushPromises()
    const line = w.find('[data-testid="recap-prompt-topics"]')
    expect(line.exists()).toBe(true)

    // The strongest topic stays in the summary line; the row carries the ones AFTER it, so it
    // never repeats what was just said one line up.
    expect(w.text()).toContain('Index investing')
    expect(line.text()).not.toContain('Index investing')
    expect(line.text()).toContain('Sleep science')
    expect(line.text()).toContain('Fusion power')
    expect(line.text()).toContain('Should render too')

    // Capped: headline + three pills, so the fifth topic never renders.
    expect(line.text()).not.toContain('Should not render')
    expect(line.text()).toContain('new')
    expect(line.text()).toContain('↓1')
    // A delta of zero renders no arrow: "unchanged" on most entries is noise.
    expect(line.text()).not.toContain('↑0')
    expect(line.text()).not.toContain('↓0')
  })

  it('renders fewer than three without padding, and keeps the card a single link', async () => {
    signIn()
    vi.spyOn(api, 'getRecap').mockResolvedValue(
      recap({
        topics: [
          { token: 'topic:a', label: 'Headline one', episodes: 9, delta: 1, is_new: false },
          { token: 'topic:b', label: 'Only one', episodes: 5, delta: -3, is_new: false },
        ],
      }),
    )
    const w = mountPrompt()
    await flushPromises()
    const line = w.find('[data-testid="recap-prompt-topics"]')
    expect(line.text()).toContain('Only one')
    expect(line.text()).toContain('↓3')
    // Nested anchors are invalid and break assistive tech: the topics must stay text.
    expect(w.findAll('a')).toHaveLength(1)
  })
})
