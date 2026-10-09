import { flushPromises, mount } from '@vue/test-utils'
import { keptAlive } from '../test/keptAlive'
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
    // "Find shows →" points at the Shows index, not the episode catalogue (#2013).
    { path: '/browse', name: 'browse', component: { template: '<div/>' } },
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


function mountIt(
  opts: { signedIn?: boolean; resp?: YourWeekResponse; layout?: 'full' | 'compact'; fail?: boolean } = {},
) {
  setActivePinia(createPinia())
  const prefs = useUserPreferencesStore()
  vi.spyOn(prefs, 'hydrate').mockResolvedValue()
  vi.spyOn(prefs, 'get').mockReturnValue(opts.layout)
  const setSpy = vi.spyOn(prefs, 'set').mockResolvedValue()
  if (opts.signedIn) {
    useAuthStore().user = { user_id: 'u_1', email: 'd@l', name: 'Dev' }
  }
  if (opts.fail) vi.spyOn(api, 'getYourWeek').mockRejectedValue(new Error('boom'))
  else vi.spyOn(api, 'getYourWeek').mockResolvedValue(opts.resp ?? EMPTY)
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

  it('renders nothing when signed in with nothing to review (operator 2026-10-08)', async () => {
    // Reverses #1591's teach-instead-of-hide, at the operator's call: a week in review with nothing
    // in it was an empty heading and a promise. Skip the section until there is something.
    const { wrapper } = mountIt({ signedIn: true, resp: EMPTY })
    await flushPromises()
    expect(wrapper.find('section').exists()).toBe(false)
  })

  it('still shows when the load fails, so an outage is not mistaken for a quiet week', async () => {
    const { wrapper } = mountIt({ signedIn: true, fail: true })
    await flushPromises()
    expect(wrapper.find('section').exists()).toBe(true)
    expect(wrapper.find('[data-testid="yourweek-toggle"]').exists()).toBe(false)
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
    expect(wrapper.text()).toContain('Episode B')
    expect(wrapper.text()).toContain(en.home.yourWeekShowMore)
    // section labels appear only in the full layout
    expect(wrapper.text()).not.toContain(en.home.yourWeekSection.new_in_follows)
  })

  it('shows an episode ONLY ONCE even when the digest returns it in two sections (#2072 follow-up)', async () => {
    // The native bug (2026-09-15): the same episode appeared twice in Your Week. The digest can
    // surface one episode in more than one section; compact flattens them, so it rendered twice.
    const RESP_DUP: YourWeekResponse = {
      sections: [
        {
          kind: 'new_in_follows',
          items: [
            { episode_slug: 'ep-dup', episode_title: 'Dup', deep_link: '/episode/ep-dup', graph_refs: [] },
          ],
        },
        {
          kind: 'trending_in_your_corpus',
          items: [
            { episode_slug: 'ep-dup', episode_title: 'Dup', deep_link: '/episode/ep-dup', graph_refs: [] },
            { episode_slug: 'ep-other', episode_title: 'Other', deep_link: '/episode/ep-other', graph_refs: [] },
          ],
        },
      ],
      period_label: 'x',
      generated_at: 'y',
    }
    const { wrapper } = mountIt({ signedIn: true, resp: RESP_DUP }) // compact by default
    await flushPromises()
    // Two UNIQUE episodes → two cards, not three.
    expect(wrapper.findAll('li')).toHaveLength(2)
    const dupLinks = wrapper
      .findAll('a')
      .filter((a) => (a.attributes('href') ?? '').includes('/episode/ep-dup'))
    expect(dupLinks).toHaveLength(1)
  })

  it('uses the item artwork as the card backdrop when present', async () => {
    const withArt: YourWeekResponse = {
      ...RESP,
      sections: RESP.sections.map((s) =>
        s.kind === 'new_in_follows'
          ? { ...s, items: s.items.map((i) => ({ ...i, image_url: 'https://img.example/ep-b.jpg' })) }
          : s,
      ),
    }
    const { wrapper } = mountIt({ signedIn: true, resp: withArt })
    await flushPromises()
    const art = wrapper.find('img')
    expect(art.exists()).toBe(true)
    expect(art.attributes('src')).toBe('https://img.example/ep-b.jpg')
  })

  it('does NOT show the revisit section, in either layout (operator 2026-09-30)', async () => {
    // "What's new" is new episodes. Due highlights live in Home's own "Highlights to revisit"
    // section (RevisitRail), with its reviewed / stop / unsave actions; showing them here too put
    // the same highlights on Home twice. The server still sends the section — the email uses it.
    for (const layout of ['compact', 'full'] as const) {
      const { wrapper } = mountIt({ signedIn: true, resp: RESP, layout })
      await flushPromises()
      expect(wrapper.text()).not.toContain('Episode A')
      expect(wrapper.text()).not.toContain('A memorable line.')
      expect(wrapper.text()).not.toContain(en.home.revisitTitle) // the old rail's label
      expect(wrapper.findAll('a').some((a) => (a.attributes('href') ?? '').includes('revisit='))).toBe(false)
      wrapper.unmount()
    }
  })

  it('a digest holding ONLY revisit renders nothing, not an empty rail', async () => {
    const onlyRevisit: YourWeekResponse = { ...RESP, sections: [RESP.sections[0]] }
    const { wrapper } = mountIt({ signedIn: true, resp: onlyRevisit })
    await flushPromises()
    expect(wrapper.find('section').exists()).toBe(false)
  })

  it('respects a saved full layout and shows per-section labels', async () => {
    const { wrapper } = mountIt({ signedIn: true, resp: RESP, layout: 'full' })
    await flushPromises()
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

describe('a return to Home re-reads the week (2026-10-09)', () => {
  it.each([false, true])('re-asks the server when the kept-alive Home comes back (late=%s)', async (late) => {
    setActivePinia(createPinia())
    const prefs = useUserPreferencesStore()
    vi.spyOn(prefs, 'hydrate').mockResolvedValue()
    vi.spyOn(prefs, 'get').mockReturnValue(undefined)
    useAuthStore().user = { user_id: 'u_1', email: 'd@l', name: 'Dev' }
    const spy = vi.spyOn(api, 'getYourWeek').mockResolvedValue(RESP)
    const tab = await keptAlive(YourWeek, { plugins: [i18n, router], late })
    expect(spy).toHaveBeenCalledTimes(1)
    await tab.leave()
    await tab.back()
    expect(spy).toHaveBeenCalledTimes(2)
  })
})
