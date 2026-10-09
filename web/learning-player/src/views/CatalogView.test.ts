import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeSummary } from '../services/types'
const readCached = vi.fn(async (_k: string): Promise<unknown> => null)
const writeCached = vi.fn(async (_k: string, _v: unknown): Promise<void> => {})
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
  writeCached: (k: string, v: unknown) => writeCached(k, v),
}))

import CatalogView from './CatalogView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'catalog', component: CatalogView },
    { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
  ],
})

function ep(slug: string, title: string): EpisodeSummary {
  return {
    slug, title, feed_id: 'f', podcast_title: 'Show', publish_date: '2024-01-01',
    duration_seconds: 1800, episode_image_url: null, feed_image_url: null, artwork_url: null,
    status: 'ready', summary_preview: 'recap', topics: [], has_transcript: true,
    has_summary: true, has_gi: false, has_kg: false, has_bridge: false,
    // Required by EpisodeSummary. Absent here for a long time — test files are excluded
    // from tsconfig.app.json, so nothing type-checks fixtures against the real shape.
    summary_text: null, summary_bullets: [],
  }
}

beforeEach(() => setActivePinia(createPinia()))
afterEach(() => vi.restoreAllMocks())

function mountView() {
  return mount(CatalogView, { global: { plugins: [i18n, router] } })
}

beforeEach(() => {
  readCached.mockReset().mockResolvedValue(null)
  writeCached.mockReset().mockResolvedValue(undefined)
})

describe('CatalogView', () => {
  it('renders episode cards from the API', async () => {
    vi.spyOn(api, 'listEpisodes').mockResolvedValue({
      items: [ep('a-1', 'First'), ep('a-2', 'Second')],
      page: 1, page_size: 20, total: 2, has_more: false,
    })
    const w = mountView()
    await flushPromises()
    expect(w.text()).toContain('First')
    expect(w.text()).toContain('Second')
    expect(w.text()).not.toContain('Load more')
  })

  it('reveals matches a page at a time while a search is active, instead of dumping every page', async () => {
    // Operator 2026-10-05: a control loads every page (a search is only right over the whole list)
    // but rendered all of it with no Load more. Now 20 matches, then 20 more per press.
    const many = Array.from({ length: 45 }, (_, i) => ep(`m-${i}`, `Match ${i}`))
    vi.spyOn(api, 'listEpisodes').mockResolvedValue({ items: many, page: 1, page_size: 45, total: 45, has_more: false })
    const w = mountView()
    await flushPromises()
    await w.get('input').setValue('Match')
    await flushPromises()
    const loadMore = () => w.findAll('button').find((b) => b.text() === 'Load more')
    expect(w.text()).toContain('Match 19')
    expect(w.text()).not.toContain('Match 20')
    await loadMore()!.trigger('click')
    expect(w.text()).toContain('Match 39')
    expect(w.text()).not.toContain('Match 40')
    await loadMore()!.trigger('click')
    expect(w.text()).toContain('Match 44')
    expect(loadMore()).toBeUndefined()
  })

  it('shows Load more and appends the next page', async () => {
    const spy = vi.spyOn(api, 'listEpisodes')
    spy.mockResolvedValueOnce({ items: [ep('a-1', 'First')], page: 1, page_size: 20, total: 2, has_more: true })
    spy.mockResolvedValueOnce({ items: [ep('a-2', 'Second')], page: 2, page_size: 20, total: 2, has_more: false })
    const w = mountView()
    await flushPromises()
    expect(w.text()).toContain('Load more')
    await w.findAll('button').find((b) => b.text() === 'Load more')!.trigger('click')
    await flushPromises()
    expect(w.text()).toContain('Second')
    expect(spy).toHaveBeenCalledTimes(2)
  })

  it('shows the empty state when there are no episodes', async () => {
    vi.spyOn(api, 'listEpisodes').mockResolvedValue({ items: [], page: 1, page_size: 20, total: 0, has_more: false })
    const w = mountView()
    await flushPromises()
    expect(w.text()).toContain('No episodes yet.')
  })

  it('shows a graceful error with a retry when the API fails (F1.4)', async () => {
    vi.spyOn(api, 'listEpisodes').mockRejectedValue(new Error('boom'))
    const w = mountView()
    await flushPromises()
    expect(w.find('[data-testid="section-retry"]').exists()).toBe(true)
  })

  /**
   * Browse offline was a bare red "Couldn't load episodes." on an otherwise empty page — no retry,
   * no content — on a device that had rendered that exact list minutes earlier (#1909).
   */
  it('falls back to the episodes it last loaded instead of a red sentence', async () => {
    readCached.mockResolvedValue([ep('cached-1', 'From Last Time')])
    vi.spyOn(api, 'listEpisodes').mockRejectedValue(new Error('offline'))
    const w = mountView()
    // Two flushes: the rejected fetch, then the cache read it falls back to.
    await flushPromises()
    await flushPromises()

    expect(w.text(), 'the cached list did not render').toContain('From Last Time')
    expect(w.find('[data-testid="catalog-stale"]').exists(), 'nothing said it was stale').toBe(true)
    expect(w.text()).not.toContain('Couldn’t load episodes.')
  })

  it('snapshots the first page so there is something to fall back TO', async () => {
    vi.spyOn(api, 'listEpisodes').mockResolvedValue({
      items: [ep('a-1', 'Fresh')],
      page: 1,
      page_size: 20,
      total: 1,
      has_more: false,
    })
    const w = mountView()
    await flushPromises()
    expect(writeCached).toHaveBeenCalledWith('browse.episodes', [
      expect.objectContaining({ slug: 'a-1' }),
    ])
    expect(w.find('[data-testid="catalog-stale"]').exists(), 'fresh data read as stale').toBe(false)
  })

  it('a failed LATER page is the end of the list, not an error over it', async () => {
    // The rows already fetched are still correct; only the continuation failed.
    const spy = vi.spyOn(api, 'listEpisodes')
    spy.mockResolvedValueOnce({
      items: [ep('a-1', 'Page One')],
      page: 1,
      page_size: 20,
      total: 40,
      has_more: true,
    })
    const w = mountView()
    await flushPromises()
    spy.mockRejectedValueOnce(new Error('offline'))
    const loadMore = w.findAll('button').find((b) => b.text().includes('Load more'))
    expect(loadMore, 'no Load more button to click').toBeTruthy()
    await loadMore!.trigger('click')
    await flushPromises()
    await flushPromises()
    expect(w.text(), 'the page already loaded was replaced').toContain('Page One')
    expect(readCached, 'a later page reached for the first-page snapshot').not.toHaveBeenCalled()
  })

  it('still shows the failure (retry) when there is nothing cached', async () => {
    readCached.mockResolvedValue(null)
    vi.spyOn(api, 'listEpisodes').mockRejectedValue(new Error('offline'))
    const w = mountView()
    await flushPromises()
    await flushPromises()
    expect(w.find('[data-testid="section-retry"]').exists()).toBe(true)
  })

})

// "Which episodes" is its own row, so it combines with the state filter (operator 2026-10-09):
// Shows I follow + Unplayed is the list What's new and Discover 2 open.
describe('CatalogView — From (which episodes) × State', () => {
  function epOf(slug: string, feed: string): EpisodeSummary {
    return { ...ep(slug, slug.toUpperCase()), feed_id: feed }
  }
  const items = [epOf('a1', 'followed'), epOf('a2', 'followed'), epOf('b1', 'other'), epOf('b2', 'other')]

  async function mountAt(query: Record<string, string>) {
    // One pinia for the test AND the component, pinned — the shared default let them drift apart
    // in a full-file run and the component read a signed-out store.
    const pinia = createPinia()
    setActivePinia(pinia)
    const { useAuthStore } = await import('../stores/auth')
    useAuthStore(pinia).user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    vi.spyOn(api, 'listEpisodes').mockResolvedValue({ items, page: 1, page_size: 20, total: 4, has_more: false })
    vi.spyOn(api, 'getPodcastsPage').mockResolvedValue({ items: [], total: 0 } as never)
    vi.spyOn(api, 'getLibrary').mockResolvedValue([{ feed_id: 'followed', feed_url: null, title: 'F', added_at: 1 }])
    vi.spyOn(api, 'getCompleted').mockResolvedValue(['a2'])
    vi.spyOn(api, 'getWorldEpisodeSlugs').mockResolvedValue(['b2'])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([
      { slug: 'b1', position_seconds: 100 } as never,
      { slug: 'a2', position_seconds: 100 } as never,
    ])
    // "a2 is finished" — through its saved position, which `isPlayed` reads alongside the
    // hand-marked set (usePlayed); the precondition, not something this test exercises.
    const { recordPosition } = await import('../services/playbackPositions')
    recordPosition('a2', 1800, true, true)
    await router.push({ path: '/', query })
    const w = mount(CatalogView, { global: { plugins: [i18n, router, pinia] } })
    await flushPromises()
    await flushPromises()
    return w
  }
  const titles = (w: Awaited<ReturnType<typeof mountAt>>) =>
    w.findAll('[data-testid="episode-card"]').map((c) => c.text()).join(' ')

  it('opens on ?from=following&state=unplayed — your shows, minus what you finished', async () => {
    const w = await mountAt({ from: 'following', state: 'unplayed' })
    const text = titles(w)
    expect(text).toContain('A1')
    expect(text).not.toContain('A2') // finished
    expect(text).not.toContain('B1') // not a show you follow
    expect(w.get('[data-testid="catalog-from-following"]').attributes('aria-checked')).toBe('true')
  })

  it('opened already filtered, it pulls in every page — a match on page two is not lost', async () => {
    const w = await mountAt({ from: 'following' })
    // mountAt's listEpisodes answers page one; make the catalogue two pages and remount.
    w.unmount()
    vi.mocked(api.listEpisodes).mockImplementation(async (p = {}) =>
      p.page === 2
        ? { items: [epOf('late', 'followed')], page: 2, page_size: 20, total: 5, has_more: false }
        : { items, page: 1, page_size: 20, total: 5, has_more: true },
    )
    const pinia = createPinia()
    setActivePinia(pinia)
    const { useAuthStore } = await import('../stores/auth')
    useAuthStore(pinia).user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    const again = mount(CatalogView, { global: { plugins: [i18n, router, pinia] } })
    await flushPromises()
    await flushPromises()
    expect(titles(again)).toContain('LATE')
  })

  it('Mine is the server\'s set (ADR-162), not a guess on the client', async () => {
    const w = await mountAt({ from: 'mine' })
    expect(api.getWorldEpisodeSlugs).toHaveBeenCalled()
    const text = titles(w)
    expect(text).toContain('B2')
    expect(text).not.toContain('A1')
  })

  it('In progress is started and not finished', async () => {
    const w = await mountAt({ state: 'inprogress' })
    const text = titles(w)
    expect(text).toContain('B1')
    expect(text).not.toContain('A2') // has a position but is finished
    expect(text).not.toContain('A1')
  })

  it('signed out there is no From row (no follows, no "mine")', async () => {
    vi.spyOn(api, 'listEpisodes').mockResolvedValue({ items, page: 1, page_size: 20, total: 4, has_more: false })
    vi.spyOn(api, 'getPodcastsPage').mockResolvedValue({ items: [], total: 0 } as never)
    await router.push({ path: '/' })
    const w = mountView()
    await flushPromises()
    expect(w.find('[data-testid="catalog-from"]').exists()).toBe(false)
  })
})
