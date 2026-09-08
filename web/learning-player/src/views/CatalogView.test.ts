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

  it('shows an error message when the API fails', async () => {
    vi.spyOn(api, 'listEpisodes').mockRejectedValue(new Error('boom'))
    const w = mountView()
    await flushPromises()
    expect(w.text()).toContain('Couldn’t load episodes.')
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

  it('still says it failed when there is nothing cached', async () => {
    readCached.mockResolvedValue(null)
    vi.spyOn(api, 'listEpisodes').mockRejectedValue(new Error('offline'))
    const w = mountView()
    await flushPromises()
    await flushPromises()
    expect(w.text()).toContain("Couldn’t load episodes.")
  })

})
