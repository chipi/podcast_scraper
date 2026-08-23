import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import { useAuthStore } from '../stores/auth'
import { useUserPreferencesStore } from '../stores/userPreferences'
import type { YourWeekResponse } from '../services/types'
import YourWeek from './YourWeek.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
    // The first-run state links here (#1591). Without the route, RouterLink throws during setup
    // and takes the whole block down — which is how this surfaced.
    { path: '/catalog', name: 'catalog', component: { template: '<div/>' } },
  ],
})

const EMPTY: YourWeekResponse = { sections: [], period_label: '', generated_at: '' }
const RESP: YourWeekResponse = {
  sections: [
    {
      kind: 'revisit',
      items: [
        {
          episode_slug: 'ep-a',
          episode_title: 'Episode A',
          // source='user' capture → carries the id that advances its spaced ladder (#35).
          highlight_id: 'h-a',
          deep_link: '/episode/ep-a?t=10&revisit=h-a',
          quote: 'A memorable line.',
          t_ms: 10000,
          image_url: 'https://img.example/ep-a.jpg',
          graph_refs: [{ id: 'topic:x', kind: 'topic', label: 'Topic X' }],
        },
      ],
    },
    {
      kind: 'new_in_follows',
      items: [
        {
          episode_slug: 'ep-b',
          episode_title: 'Episode B',
          deep_link: '/episode/ep-b',
          graph_refs: [{ id: 'topic:y', kind: 'topic', label: 'Topic Y' }],
        },
      ],
    },
    {
      kind: 'trending_in_your_corpus',
      items: [
        {
          episode_slug: 'ep-c',
          episode_title: 'Episode C',
          deep_link: '/topic/z?scope=mine',
          graph_refs: [{ id: 'topic:z', kind: 'topic', label: 'Topic Z' }],
        },
      ],
    },
  ],
  period_label: 'Aug 1 – 7',
  generated_at: '2026-08-07T00:00:00Z',
}

function mountIt(opts: { signedIn?: boolean; resp?: YourWeekResponse; layout?: 'full' | 'compact' } = {}) {
  setActivePinia(createPinia())
  const prefs = useUserPreferencesStore()
  vi.spyOn(prefs, 'hydrate').mockResolvedValue()
  vi.spyOn(prefs, 'get').mockReturnValue(opts.layout)
  const setSpy = vi.spyOn(prefs, 'set').mockResolvedValue()
  if (opts.signedIn) {
    useAuthStore().user = { user_id: 'u_1', email: 'd@l', name: 'Dev' }
  }
  vi.spyOn(api, 'getYourWeek').mockResolvedValue(opts.resp ?? EMPTY)
  const wrapper = mount(YourWeek, { global: { plugins: [i18n, router] } })
  return { wrapper, setSpy }
}

afterEach(() => vi.restoreAllMocks())

describe('YourWeek section', () => {
  it('is hidden when signed out (never calls the API)', async () => {
    const spy = vi.spyOn(api, 'getYourWeek')
    const { wrapper } = mountIt({ signedIn: false, resp: RESP })
    await flushPromises()
    expect(wrapper.find('section').exists()).toBe(false)
    expect(spy).not.toHaveBeenCalled()
  })

  it('renders a first-run state when signed in with nothing due (#1591)', async () => {
    // Reverses the old self-hiding contract, deliberately. Hiding meant a brand-new user — the
    // person most in need of learning a weekly digest exists — got no hint of it at all, and an
    // API outage was indistinguishable from a quiet week.
    const { wrapper } = mountIt({ signedIn: true, resp: EMPTY })
    await flushPromises()
    expect(wrapper.find('section').exists()).toBe(true)
    expect(wrapper.find('[data-testid="yourweek-firstrun"]').exists()).toBe(true)
    // Three rows, one per digest section, each saying what will appear there.
    expect(wrapper.findAll('[data-testid="yourweek-firstrun"] li')).toHaveLength(3)
    // No compact/full toggle: there is nothing to expand yet.
    expect(wrapper.find('[data-testid="yourweek-toggle"]').exists()).toBe(false)
  })

  it('the follows row is actionable; the other two are not (#1591)', async () => {
    // new_in_follows is USER-empty — blank because you follow nothing, which you can fix — so it
    // links. revisit and trending are SYSTEM-empty: they fill as you listen, nothing to click.
    const { wrapper } = mountIt({ signedIn: true, resp: EMPTY })
    await flushPromises()
    const rows = wrapper.findAll('[data-testid="yourweek-firstrun"] li')
    expect(rows[0].find('a').exists()).toBe(true)
    expect(rows[1].find('a').exists()).toBe(false)
    expect(rows[2].find('a').exists()).toBe(false)
  })

  it('stays hidden when signed out', async () => {
    // Unchanged: the digest is per-user, so there is nothing to teach an anonymous visitor here.
    const { wrapper } = mountIt({ signedIn: false, resp: EMPTY })
    await flushPromises()
    expect(wrapper.find('section').exists()).toBe(false)
  })

  it('renders the compact rail by default with content', async () => {
    const { wrapper } = mountIt({ signedIn: true, resp: RESP })
    await flushPromises()
    expect(wrapper.text()).toContain(en.home.yourWeek)
    expect(wrapper.text()).toContain('Episode A')
    expect(wrapper.text()).toContain('A memorable line.')
    expect(wrapper.text()).toContain(en.home.yourWeekShowMore)
    // section labels appear only in the full layout
    expect(wrapper.text()).not.toContain(en.home.yourWeekSection.new_in_follows)
  })

  it('uses the item artwork as the card backdrop when present', async () => {
    const { wrapper } = mountIt({ signedIn: true, resp: RESP })
    await flushPromises()
    const art = wrapper.find('img')
    expect(art.exists()).toBe(true)
    expect(art.attributes('src')).toBe('https://img.example/ep-a.jpg')
  })

  it('respects a saved full layout and shows per-section labels', async () => {
    const { wrapper } = mountIt({ signedIn: true, resp: RESP, layout: 'full' })
    await flushPromises()
    expect(wrapper.text()).toContain(en.home.yourWeekSection.revisit)
    expect(wrapper.text()).toContain(en.home.yourWeekSection.new_in_follows)
    expect(wrapper.text()).toContain(en.home.yourWeekSection.trending_in_your_corpus)
    // trending cards carry a (route-backfilled) episode title — never blank
    expect(wrapper.text()).toContain('Episode C')
    expect(wrapper.text()).toContain(en.home.yourWeekShowLess)
  })

  it('loads once auth resolves AFTER mount (guards the async-auth race)', async () => {
    setActivePinia(createPinia())
    const prefs = useUserPreferencesStore()
    vi.spyOn(prefs, 'hydrate').mockResolvedValue()
    vi.spyOn(prefs, 'get').mockReturnValue(undefined)
    vi.spyOn(prefs, 'set').mockResolvedValue()
    const auth = useAuthStore() // signed OUT at mount time
    const spy = vi.spyOn(api, 'getYourWeek').mockResolvedValue(RESP)
    const wrapper = mount(YourWeek, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(wrapper.find('section').exists()).toBe(false) // hidden; no fetch while signed out
    expect(spy).not.toHaveBeenCalled()

    auth.user = { user_id: 'u_1', email: 'd@l', name: 'Dev' } // auth hydrates after mount
    await flushPromises()
    expect(spy).toHaveBeenCalled() // the watcher (re)loads on the transition
    await flushPromises()
    expect(wrapper.find('[data-testid="your-week"]').exists()).toBe(true)
  })

  it('toggles layout inline and persists the preference', async () => {
    const { wrapper, setSpy } = mountIt({ signedIn: true, resp: RESP })
    await flushPromises()
    await wrapper.get('[data-testid="yourweek-toggle"]').trigger('click')
    expect(setSpy).toHaveBeenCalledWith('lp.yourweek.layout', 'full')
    expect(wrapper.text()).toContain(en.home.yourWeekSection.new_in_follows)
  })
})

describe('Your Week links advance the spaced ladder (#35)', () => {
  // Consuming revisit through Your Week used to advance nothing — the only advance path in the
  // product was the inbox's dismiss button — so the same cards came back every week. The card now
  // carries ?revisit=<id> and the PLAYER marks on arrival, the same mechanism the digest email and
  // the inbox jump link use.

  it("a user's own capture links with ?revisit", async () => {
    const { wrapper } = mountIt({ signedIn: true, resp: RESP })
    await flushPromises()
    const href =
      wrapper.findAll('a').find((a) => (a.attributes('href') ?? '').includes('/episode/ep-a'))
        ?.attributes('href') ?? ''
    expect(href).toContain('revisit=h-a')
    expect(href).toContain('t=10')
  })

  it('an item with no highlight_id links without one', async () => {
    // Auto-picks and the follows/trending rows have no ladder behind them; a marker there would
    // record a review against a highlight that does not exist.
    const { wrapper } = mountIt({ signedIn: true, resp: RESP })
    await flushPromises()
    const href =
      wrapper.findAll('a').find((a) => (a.attributes('href') ?? '').includes('/episode/ep-b'))
        ?.attributes('href') ?? ''
    expect(href).not.toContain('revisit')
  })
})
