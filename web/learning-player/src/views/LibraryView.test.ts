import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeDetail, EpisodeSummary } from '../services/types'
import { useSavedQueriesStore } from '../stores/savedQueries'
// Defaults to "nothing cached", so the offline test below reaches the unavailable branch rather
// than sitting on a real device-storage read that never resolves under happy-dom.
const readCached = vi.fn(async (_k: string): Promise<unknown> => null)
vi.mock('../services/contentCache', () => ({
  isArrayCache: (v: unknown) => Array.isArray(v),
  hasArrayFields:
    (...f: string[]) =>
    (v: unknown) =>
      typeof v === 'object' &&
      v !== null &&
      !Array.isArray(v) &&
      f.every((k) => Array.isArray((v as Record<string, unknown>)[k])),
  readCached: (k: string) => readCached(k),
  writeCached: async () => {},
  CACHE_KEYS: [],
  setCacheNamespace: () => {},
  cacheNamespace: () => 'anon',
  clearCached: async () => {},
  ANON_NAMESPACE: 'anon',
}))

import { useLibraryStore } from '../stores/library'
import { useAuthStore } from '../stores/auth'
import LibraryView from './LibraryView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'catalog', component: { template: '<div/>' } },
    { path: '/library', name: 'library', component: LibraryView },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
    { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
    { path: '/search', name: 'search', component: { template: '<div/>' } },
  ],
})

function summary(over: Partial<EpisodeSummary> = {}): EpisodeSummary {
  return {
    slug: 'fav-1', title: 'Saved Episode', feed_id: 'f', podcast_title: 'Show',
    publish_date: '2024-03-10', duration_seconds: 1800, episode_image_url: null,
    feed_image_url: null, artwork_url: null, status: 'ready', summary_preview: 'A recap.',
    summary_text: null, summary_bullets: [], topics: [], has_transcript: true,
    has_summary: false, has_gi: false, has_kg: false, has_bridge: false, ...over,
  }
}

function detail(over: Partial<EpisodeDetail> = {}): EpisodeDetail {
  return {
    slug: 'recent-1', title: 'Recently Played', feed_id: 'f', podcast_title: 'Show',
    publish_date: '2024-03-09', duration_seconds: 2400, episode_image_url: null,
    feed_image_url: null, artwork_url: null, summary_title: null, summary_bullets: [],
    summary_text: 'Heard recently.', has_transcript: true, has_summary: false,
    has_gi: false, has_kg: false, has_bridge: false, ...over,
  }
}

/**
 * Library inside a `<KeepAlive>`, which is how `App.vue` renders it — and how every test here
 * mounts it (#2024).
 *
 * `onActivated` only fires under KeepAlive, and this view uses it for two things nothing else
 * does: refreshing the Revisit badge on every return, and retrying the follows list when it has
 * nothing good to show. A plain mount runs neither, so those paths were invisible.
 */
function mountKeptAlive() {
  return mount(
    { components: { LibraryView }, template: '<KeepAlive><LibraryView /></KeepAlive>' },
    { global: { plugins: [i18n, router] } },
  )
}

/** Library under KeepAlive with a switch, so a test can leave and come back. */
function mountReturnable() {
  const wrapper = mount(
    {
      components: { LibraryView },
      data: () => ({ here: true }),
      template: '<KeepAlive><LibraryView v-if="here" /></KeepAlive>',
    },
    { global: { plugins: [i18n, router] } },
  )
  return {
    wrapper,
    leave: async () => {
      await wrapper.setData({ here: false })
      await flushPromises()
    },
    comeBack: async () => {
      await wrapper.setData({ here: true })
      await flushPromises()
    },
  }
}

function tabButton(w: ReturnType<typeof mount>, label: string) {
  return w.findAll('button').find((b) => b.text() === label)!
}

beforeEach(() => {
  setActivePinia(createPinia())
  // QueueView (embedded) hydrates the queue; EpisodeCards embed FavoriteButton.
  vi.spyOn(api, 'getQueue').mockResolvedValue([])
  vi.spyOn(api, 'putQueue').mockResolvedValue()
  vi.spyOn(api, 'getFavorites').mockResolvedValue({ episodes: [] })
  // Shows tab loads the public catalogue to join artwork onto follows.
  vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
  vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
  vi.spyOn(api, 'getEpisode').mockResolvedValue(detail())
  // HighlightsView (embedded) hydrates the capture store on mount.
  vi.spyOn(api, 'getHighlights').mockResolvedValue([])
  vi.spyOn(api, 'getNotes').mockResolvedValue([])
  // CollectionsView (embedded) loads on mount.
  vi.spyOn(api, 'getCollections').mockResolvedValue([])
  // ResurfacingInbox (embedded) loads on mount.
  vi.spyOn(api, 'getResurfacing').mockResolvedValue({ items: [], paused: false })
})
afterEach(() => vi.restoreAllMocks())

describe('LibraryView', () => {
  it('renders all tabs with their labels', async () => {
    const w = mountKeptAlive()
    await flushPromises()
    const labels = w.findAll('button').map((b) => b.text())
    // Following · Saved · Revisit. Highlights + Collections are SECTIONS inside Saved; Queue + Recent
    // moved to the player surface (#1838), so neither is a Library tab any more.
    expect(labels).toContain('Following') // was "Shows" — now covers shows + topics/people/storylines
    expect(labels).toContain('Saved')
    expect(labels).toContain('Boards') // the collections tab, renamed (CO.7)
    expect(labels).toContain('Revisit')
    expect(labels).not.toContain('Queue')
    expect(labels).not.toContain('Recent')
    expect(labels).not.toContain('Knowledge')
    expect(labels).not.toContain('Highlights') // a section header (h2) in Saved, not a tab
  })

  it('Saved shows the captured highlights (folded-in section) with an export link', async () => {
    vi.spyOn(api, 'getHighlights').mockResolvedValue([
      {
        id: 'h1', episode_slug: 'fav-1', kind: 'span', start_ms: 65_000, end_ms: 68_000,
        char_start: 0, char_end: 5, segment_ids: ['s1'], quote_text: 'a captured line',
        speaker: null, source_insight_id: null, color: null, created_at: 1, anchor_status: null,
      },
    ])
    vi.spyOn(api, 'getEpisode').mockResolvedValue(detail({ slug: 'fav-1', title: 'Saved Episode' }))
    const w = mountKeptAlive()
    await flushPromises()
    // Highlights lives in the default Saved tab now — no tab switch needed.
    expect(w.text()).toContain('a captured line')
    expect(w.text()).toContain('1:05') // 65_000ms jump link
    const exportLink = w.findAll('a').find((a) => (a.attributes('href') ?? '').includes('export.md'))
    expect(exportLink).toBeTruthy()
  })

  it('Saved lists favorited episodes via EpisodeCard', async () => {
    vi.spyOn(api, 'getFavorites').mockResolvedValue({
      episodes: [summary({ slug: 'a', title: 'Alpha Saved' })],
    })
    const w = mountKeptAlive()
    await flushPromises()
    expect(w.text()).toContain('Alpha Saved')
    expect(w.findAll('a').map((a) => a.attributes('href'))).toContain('/episode/a')
  })

  it('caps a long Saved episodes list and expands it in place (#2042)', async () => {
    const episodes = Array.from({ length: 9 }, (_, i) =>
      summary({ slug: `e${i}`, title: `Saved Episode ${i}` }),
    )
    vi.spyOn(api, 'getFavorites').mockResolvedValue({ episodes })
    const w = mountKeptAlive()
    await flushPromises()
    // Capped to the top 6 (SECTION_CAP) with a Show-all toggle.
    expect(w.findAll('[data-testid="episode-card"]')).toHaveLength(6)
    const toggle = w.find('[data-testid="show-all-toggle"]')
    expect(toggle.exists()).toBe(true)
    await toggle.trigger('click')
    expect(w.findAll('[data-testid="episode-card"]')).toHaveLength(9)
  })

  it('the Saved search narrows every section and lifts the caps (#2042)', async () => {
    const episodes = Array.from({ length: 9 }, (_, i) =>
      summary({ slug: `e${i}`, title: i === 0 ? 'Unique Sleep Talk' : `Filler ${i}` }),
    )
    vi.spyOn(api, 'getFavorites').mockResolvedValue({ episodes })
    const w = mountKeptAlive()
    await flushPromises()
    await w.find('[data-testid="saved-search"]').setValue('sleep')
    await flushPromises()
    const cards = w.findAll('[data-testid="episode-card"]')
    expect(cards).toHaveLength(1)
    expect(w.text()).toContain('Unique Sleep Talk')
    // With a match present the "nothing matches" note stays hidden.
    expect(w.find('[data-testid="saved-no-match"]').exists()).toBe(false)
  })

  it('a Saved search with no matches shows the no-match note (#2042)', async () => {
    vi.spyOn(api, 'getFavorites').mockResolvedValue({
      episodes: [summary({ slug: 'a', title: 'Alpha Saved' })],
    })
    const w = mountKeptAlive()
    await flushPromises()
    await w.find('[data-testid="saved-search"]').setValue('zzzzz-nothing')
    await flushPromises()
    expect(w.find('[data-testid="saved-no-match"]').exists()).toBe(true)
  })

  it('an empty Saved shows ONE empty state, not a lone Highlights heading', async () => {
    // Highlights used to be the only unconditional section here, so an empty account met a single
    // "Highlights" heading standing in for a tab that actually holds three things — episodes,
    // insights AND highlights — and read as redundant. It is conditional like its siblings now, and
    // the tab speaks for itself once when it has nothing at all.
    const w = mountKeptAlive()
    await flushPromises()
    const headings = w.findAll('h2').map((h) => h.text())
    expect(headings).not.toContain('Highlights')
    expect(headings).not.toContain('Collections') // its own tab now, not a Saved section
    expect(w.text()).toContain('Episodes you favourite, insights you keep, and moments you mark')
    // The empty state is a dead end without an action to take.
    const cta = w.findAll('a').find((a) => a.text().includes('Find something to listen to'))
    expect(cta).toBeTruthy()
  })

  // Insights are NOT favorites (RFC-121 / #1593) — they save via the highlights path and render in
  // the Highlights section, not a favorites Insights section. Covered by the highlights tests.

  // #1261-8: Saved-searches section in the Saved tab
  it('Saved tab renders a "Searches" section for each saved query with a re-run link and remove button', async () => {
    const savedQueries = useSavedQueriesStore()
    await savedQueries.save('AI regulation', 'all', 1_000)
    await savedQueries.save('sleep science', 'mine', 2_000)
    const w = mountKeptAlive()
    await flushPromises()
    const section = w.get('[data-testid="saved-searches-section"]')
    // Header + both entries render.
    expect(section.text()).toContain('Searches')
    expect(section.text()).toContain('AI regulation')
    expect(section.text()).toContain('sleep science')
    // Re-run links point at the Search route with q + scope query params.
    const links = section.findAll('a')
    expect(links.some((a) => a.attributes('href') === '/search?q=sleep+science&scope=mine')).toBe(
      true,
    )
    expect(links.some((a) => a.attributes('href') === '/search?q=AI+regulation&scope=all')).toBe(
      true,
    )
  })

  it('the saved-search × button removes just that entry (leaves others in place)', async () => {
    const savedQueries = useSavedQueriesStore()
    await savedQueries.save('AI regulation', 'all')
    await savedQueries.save('sleep science', 'mine')
    const w = mountKeptAlive()
    await flushPromises()
    const removeBtn = w
      .get('[data-testid="saved-searches-section"]')
      .findAll('button')
      .find((b) => (b.attributes('aria-label') ?? '').includes('AI regulation'))!
    await removeBtn.trigger('click')
    await flushPromises()
    expect(savedQueries.list.map((it) => it.q)).toEqual(['sleep science'])
  })

  // Recent (playback history) + Queue moved to the player-surface QueuePanel (#1838); their coverage
  // lives in QueuePanel.test.ts. Library no longer has those tabs.

  /**
   * Offline, the capture load rejected, `count` stayed 0, and this tab rendered "Episodes you
   * favourite, insights you keep, and moments you mark all live here" — telling a user with
   * highlights that they had kept nothing. Emptiness is a claim about the ACCOUNT; that one was a
   * claim about the network.
   */
  it('does not claim the account is empty when the library is merely unknown', async () => {
    vi.spyOn(api, 'getHighlights').mockRejectedValue(new Error('offline'))
    vi.spyOn(api, 'getNotes').mockRejectedValue(new Error('offline'))
    const w = mountKeptAlive()
    await flushPromises()
    await tabButton(w, 'Saved').trigger('click')
    await flushPromises()

    expect(w.find('[data-testid="saved-unavailable"]').exists(), 'nothing said it was unknown').toBe(
      true,
    )
    expect(w.text()).not.toContain('Episodes you favourite, insights you keep')
  })

  it('still shows the empty state for an account that genuinely has nothing', async () => {
    // The distinction has to cut both ways or it is just a different lie.
    const w = mountKeptAlive()
    await flushPromises()
    await tabButton(w, 'Saved').trigger('click')
    await flushPromises()

    expect(w.find('[data-testid="saved-unavailable"]').exists()).toBe(false)
    expect(w.text()).toContain('Episodes you favourite, insights you keep')
  })

  it('caps the followed-shows grid and a search narrows it (#2042)', async () => {
    // The followed-shows grid gates on auth (useFollowedShows), unlike the favourites path.
    useAuthStore().user = { user_id: 'u1', email: 'u@x.com', name: 'U' } as never
    // 9 followed shows → capped to 6 with a Show-all toggle; the Following search narrows the grid.
    vi.spyOn(api, 'getLibrary').mockResolvedValue(
      Array.from({ length: 9 }, (_, i) => ({
        feed_id: `f${i}`,
        feed_url: null,
        title: i === 0 ? 'Unique Science Weekly' : `Filler Show ${i}`,
        added_at: null,
      })),
    )
    const w = mountKeptAlive()
    await flushPromises()
    await tabButton(w, 'Following').trigger('click')
    await flushPromises()

    const grid = () => w.find('[data-testid="library-shows-grid"]')
    expect(grid().findAll('li')).toHaveLength(6)
    await w.find('[data-testid="show-all-toggle"]').trigger('click')
    expect(grid().findAll('li')).toHaveLength(9)

    // Only the active tab mounts its SavedFilterBar, so the shared `saved-search` testid is
    // unambiguous here (Following tab active).
    await w.find('[data-testid="saved-search"]').setValue('science')
    await flushPromises()
    expect(grid().findAll('li')).toHaveLength(1)
    expect(w.text()).toContain('Unique Science Weekly')
  })

  /**
   * Every per-account store here already tracked `stale` and none of them said so. Home got a
   * notice; Library — the tab you open to find things you kept — showed a cached list as though it
   * were current (#1909).
   */
  describe('staleness (#1909)', () => {
    it('says the lists are the ones it last loaded', async () => {
      // Shape matters per key: favourites stores `{episodes, insights}`, the others store arrays.
      // A wrong shape here makes the LibraryView render throw, because no store validates what it
      // reads back — worth knowing, and not what this test is about.
      readCached.mockImplementation(async (k: string) =>
        k === 'favorites' ? { episodes: [] } : [],
      )
      vi.spyOn(api, 'getFavorites').mockRejectedValue(new Error('offline'))
      const w = mountKeptAlive()
      await flushPromises()
      await flushPromises()
      expect(w.find('[data-testid="stale-notice"]').exists(), 'nothing said it was stale').toBe(true)
    })

    it('watches every list, not just favourites', async () => {
      // Otherwise a stale captures/library/collections copy is presented as current, and the notice
      // is only honest about one of the four things this tab shows.
      readCached.mockImplementation(async (k: string) =>
        k === 'captures' ? { highlights: [], notes: [] } : null,
      )
      vi.spyOn(api, 'getHighlights').mockRejectedValue(new Error('offline'))
      vi.spyOn(api, 'getNotes').mockRejectedValue(new Error('offline'))
      const w = mountKeptAlive()
      await flushPromises()
      await flushPromises()
      expect(w.find('[data-testid="stale-notice"]').exists()).toBe(true)
    })

    it('says nothing when everything is fresh', async () => {
      const w = mountKeptAlive()
      await flushPromises()
      expect(w.find('[data-testid="stale-notice"]').exists()).toBe(false)
    })

    it('the retry goes back to the network for lists already loaded from cache', async () => {
      // Shape matters per key: favourites stores `{episodes, insights}`, the others store arrays.
      // A wrong shape here makes the LibraryView render throw, because no store validates what it
      // reads back — worth knowing, and not what this test is about.
      readCached.mockImplementation(async (k: string) =>
        k === 'favorites' ? { episodes: [] } : [],
      )
      const spy = vi.spyOn(api, 'getFavorites').mockRejectedValue(new Error('offline'))
      const w = mountKeptAlive()
      await flushPromises()
      await flushPromises()
      const before = spy.mock.calls.length

      // The retry reloads ALL four stores, so every one of them has to be able to succeed — the
      // notice is about the tab, not about favourites.
      spy.mockResolvedValue({ episodes: [] })
      vi.spyOn(api, 'getLibrary').mockResolvedValue([])
      await w.get('[data-testid="stale-retry"]').trigger('click')
      await flushPromises()
      expect(spy.mock.calls.length, 'the retry did not refetch').toBeGreaterThan(before)
      expect(w.find('[data-testid="stale-notice"]').exists(), 'the notice stayed up').toBe(false)
    })
  })
})

/**
 * What `onActivated` owns here, and nothing covered until this file mounted under KeepAlive
 * (#2024). Both of these exist because `onMounted` fires ONCE for a kept-alive tab.
 */
describe('returning to Library (#2024)', () => {
  it('refreshes the Revisit badge, so it cannot disagree with the tab', async () => {
    // A badge loaded only in onMounted goes stale the moment you review anything, and Revisit is
    // one tap away — the nav would keep claiming items that are no longer due.
    const spy = vi.spyOn(api, 'getResurfacing').mockResolvedValue({ items: [], paused: false })
    const { leave, comeBack } = mountReturnable()
    await flushPromises()
    const first = spy.mock.calls.length
    expect(first).toBeGreaterThan(0)

    await leave()
    await comeBack()
    expect(spy.mock.calls.length, 'the badge was left stale on return').toBeGreaterThan(first)
  })

  it('retries the follows list when it had nothing good to show', async () => {
    // A failed library fetch left this tab saying "you're not following any shows yet" plus six
    // suggestions for the rest of the session, with no way back short of a full reload.
    // The follows section resolves through `useFollowedShows`, which fetches the CATALOGUE —
    // `getLibrary` is a different call and asserting on it proved nothing.
    const spy = vi.spyOn(api, 'getPodcasts').mockRejectedValue(new Error('offline'))
    const { leave, comeBack } = mountReturnable()
    await flushPromises()
    const first = spy.mock.calls.length
    expect(first, 'the first load never ran').toBeGreaterThan(0)

    await leave()
    await comeBack()
    expect(spy.mock.calls.length, 'a failed follows list never retried').toBeGreaterThan(first)
  })

  it('does NOT refetch follows on every visit once it has them', async () => {
    // The other half of the rule: a healthy tab must not re-request on each return.
    // "Healthy" means BOTH halves: the catalogue AND the library store. The retry is gated on
    // `phase === 'error' || !libraryStore.loaded`, so leaving `getLibrary` to fail makes the tab
    // legitimately un-healthy and it retries — correctly. Mocking only the catalogue tested
    // nothing.
    const spy = vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    // `App.vue` loads the library store at sign-in, not this view — so in a test that mounts only
    // LibraryView, `loaded` is permanently false and the tab is legitimately un-healthy forever.
    // Standing it up is what makes "healthy" mean anything here.
    const lib = useLibraryStore()
    lib.items = []
    lib.loaded = true
    const { leave, comeBack } = mountReturnable()
    await flushPromises()
    const first = spy.mock.calls.length
    expect(first, 'the first load never ran').toBeGreaterThan(0)

    await leave()
    await comeBack()
    expect(spy.mock.calls.length, 'a healthy tab refetched its follows').toBe(first)
  })
})
