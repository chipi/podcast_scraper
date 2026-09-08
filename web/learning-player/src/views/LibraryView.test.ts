import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeDetail, EpisodeSummary, FavoriteInsight } from '../services/types'
import { useSavedQueriesStore } from '../stores/savedQueries'
// Defaults to "nothing cached", so the offline test below reaches the unavailable branch rather
// than sitting on a real device-storage read that never resolves under happy-dom.
const readCached = vi.fn(async (_k: string): Promise<unknown> => null)
vi.mock('../services/contentCache', () => ({
  readCached: (k: string) => readCached(k),
  writeCached: async () => {},
  CACHE_KEYS: [],
  setCacheNamespace: () => {},
  cacheNamespace: () => 'anon',
  clearCached: async () => {},
  ANON_NAMESPACE: 'anon',
}))

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

function insight(over: Partial<FavoriteInsight> = {}): FavoriteInsight {
  return {
    ref: 'ins-1', text: 'A grounded saved insight.', episode_slug: 'fav-1',
    podcast_title: 'Show', start_ms: 65_000, ...over,
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
  vi.spyOn(api, 'getFavorites').mockResolvedValue({ episodes: [], insights: [] })
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
    const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    const labels = w.findAll('button').map((b) => b.text())
    // Following · Saved · Revisit. Highlights + Collections are SECTIONS inside Saved; Queue + Recent
    // moved to the player surface (#1838), so neither is a Library tab any more.
    expect(labels).toContain('Following') // was "Shows" — now covers shows + topics/people/storylines
    expect(labels).toContain('Saved')
    expect(labels).toContain('Collections') // first-class tab now (RFC-119)
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
    const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
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
      insights: [],
    })
    const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.text()).toContain('Alpha Saved')
    expect(w.findAll('a').map((a) => a.attributes('href'))).toContain('/episode/a')
  })

  it('an empty Saved shows ONE empty state, not a lone Highlights heading', async () => {
    // Highlights used to be the only unconditional section here, so an empty account met a single
    // "Highlights" heading standing in for a tab that actually holds three things — episodes,
    // insights AND highlights — and read as redundant. It is conditional like its siblings now, and
    // the tab speaks for itself once when it has nothing at all.
    const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    const headings = w.findAll('h2').map((h) => h.text())
    expect(headings).not.toContain('Highlights')
    expect(headings).not.toContain('Collections') // its own tab now, not a Saved section
    expect(w.text()).toContain('Episodes you favourite, insights you keep, and moments you mark')
    // The empty state is a dead end without an action to take.
    const cta = w.findAll('a').find((a) => a.text().includes('Find something to listen to'))
    expect(cta).toBeTruthy()
  })

  it('Saved shows saved insights in the Insights section (no separate tab) with a ?t= jump', async () => {
    vi.spyOn(api, 'getFavorites').mockResolvedValue({ episodes: [], insights: [insight()] })
    const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    // Saved is the default tab — insights render here under the "Insights" section, no tab switch.
    expect(w.text()).toContain('Insights') // section heading
    expect(w.text()).toContain('A grounded saved insight.')
    expect(w.text()).toContain('1:05') // 65_000ms → 1:05
    const link = w.findAll('a').find((a) => (a.attributes('href') ?? '').includes('/episode/fav-1'))!
    expect(link.attributes('href')).toContain('t=65')
  })

  it('Saved shows Episodes + Insights as separate sections when both are present', async () => {
    vi.spyOn(api, 'getFavorites').mockResolvedValue({
      episodes: [summary({ slug: 'a', title: 'Alpha Saved' })],
      insights: [insight()],
    })
    const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    const headings = w.findAll('h2').map((h) => h.text())
    expect(headings).toContain('Episodes')
    expect(headings).toContain('Insights')
    expect(w.text()).toContain('Alpha Saved')
    expect(w.text()).toContain('A grounded saved insight.')
  })

  // #1261-8: Saved-searches section in the Saved tab
  it('Saved tab renders a "Searches" section for each saved query with a re-run link and remove button', async () => {
    const savedQueries = useSavedQueriesStore()
    await savedQueries.save('AI regulation', 'all', 1_000)
    await savedQueries.save('sleep science', 'mine', 2_000)
    const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
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
    const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
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
    const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
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
    const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    await tabButton(w, 'Saved').trigger('click')
    await flushPromises()

    expect(w.find('[data-testid="saved-unavailable"]').exists()).toBe(false)
    expect(w.text()).toContain('Episodes you favourite, insights you keep')
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
        k === 'favorites' ? { episodes: [], insights: [] } : [],
      )
      vi.spyOn(api, 'getFavorites').mockRejectedValue(new Error('offline'))
      const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
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
      const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
      await flushPromises()
      await flushPromises()
      expect(w.find('[data-testid="stale-notice"]').exists()).toBe(true)
    })

    it('says nothing when everything is fresh', async () => {
      const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
      await flushPromises()
      expect(w.find('[data-testid="stale-notice"]').exists()).toBe(false)
    })

    it('the retry goes back to the network for lists already loaded from cache', async () => {
      // Shape matters per key: favourites stores `{episodes, insights}`, the others store arrays.
      // A wrong shape here makes the LibraryView render throw, because no store validates what it
      // reads back — worth knowing, and not what this test is about.
      readCached.mockImplementation(async (k: string) =>
        k === 'favorites' ? { episodes: [], insights: [] } : [],
      )
      const spy = vi.spyOn(api, 'getFavorites').mockRejectedValue(new Error('offline'))
      const w = mount(LibraryView, { global: { plugins: [i18n, router] } })
      await flushPromises()
      await flushPromises()
      const before = spy.mock.calls.length

      // The retry reloads ALL four stores, so every one of them has to be able to succeed — the
      // notice is about the tab, not about favourites.
      spy.mockResolvedValue({ episodes: [], insights: [] })
      vi.spyOn(api, 'getLibrary').mockResolvedValue([])
      await w.get('[data-testid="stale-retry"]').trigger('click')
      await flushPromises()
      expect(spy.mock.calls.length, 'the retry did not refetch').toBeGreaterThan(before)
      expect(w.find('[data-testid="stale-notice"]').exists(), 'the notice stayed up').toBe(false)
    })
  })
})
