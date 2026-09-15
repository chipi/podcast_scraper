import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { Podcast } from '../services/types'
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

import ShowBrowseView from './ShowBrowseView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const stub = { template: '<div/>' }

function makeRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/', name: 'home', component: stub },
      { path: '/browse/shows', name: 'browse-shows', component: ShowBrowseView },
      { path: '/podcast/:feedId', name: 'podcast', component: stub, props: true },
    ],
  })
}

function show(feed_id: string, title: string): Podcast {
  return { feed_id, title, artwork_url: null, image_url: null, description: null, episode_count: 3 }
}

async function mountView(props: Record<string, unknown> = {}) {
  setActivePinia(createPinia())
  const router = makeRouter()
  await router.push({ name: 'browse-shows' })
  await router.isReady()
  const w = mount(ShowBrowseView, { props, global: { plugins: [i18n, router, createPinia()] } })
  await flushPromises()
  return w
}

afterEach(() => vi.restoreAllMocks())

describe('ShowBrowseView', () => {
  it('lists all shows alphabetically, each linking to its podcast page', async () => {
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([show('f-z', 'Zebra Cast'), show('f-a', 'Acme Show')])
    const w = await mountView()
    expect(w.find('[data-testid="show-browse-grid"]').exists()).toBe(true)
    const links = w.findAll('a[href^="/podcast/"]')
    expect(links.length).toBe(2)
    // Alphabetical: Acme before Zebra.
    expect(links[0].attributes('href')).toBe('/podcast/f-a')
  })

  it('filters by name and sorts Z–A (shared four-way sort)', async () => {
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([
      { feed_id: 'f-a', title: 'Acme Show', artwork_url: null, image_url: null, description: null, episode_count: 2 },
      { feed_id: 'f-z', title: 'Zebra Cast', artwork_url: null, image_url: null, description: null, episode_count: 40 },
    ])
    const w = await mountView()
    // Sort Z–A → Zebra leads Acme. Sort is a ToolbarMenu (Newest/Oldest/A–Z/Z–A): open, pick.
    await w.get('[data-testid="show-browse-sort"]').trigger('click')
    await w.get('[data-testid="show-browse-sort-opt-za"]').trigger('click')
    expect(w.findAll('a[href^="/podcast/"]')[0].attributes('href')).toBe('/podcast/f-z')
    // Filter narrows to matches only.
    await w.get('[data-testid="show-browse-search"]').setValue('acme')
    const links = w.findAll('a[href^="/podcast/"]')
    expect(links.length).toBe(1)
    expect(links[0].attributes('href')).toBe('/podcast/f-a')
  })

  it('filters by category when the catalogue carries any (BS.1)', async () => {
    const withCat = (feed_id: string, title: string, category: string | null): Podcast => ({
      ...show(feed_id, title),
      category,
    })
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([
      withCat('f-b', 'Biz Cast', 'Business'),
      withCat('f-t', 'Tech Cast', 'Technology'),
      withCat('f-n', 'No Cat', null),
    ])
    const w = await mountView()
    // Category is a ToolbarMenu now: open it, then read/pick the option buttons.
    await w.get('[data-testid="show-browse-category"]').trigger('click')
    const opts = w
      .findAll('[data-testid^="show-browse-category-opt-"]')
      .map((o) => o.text().replace('✓', '').trim()) // the active option renders a ✓ tick
    // Distinct categories only (the null one is excluded), plus the "All" reset.
    expect(opts).toEqual(['All', 'Business', 'Technology'])
    await w.get('[data-testid="show-browse-category-opt-Business"]').trigger('click')
    expect(w.text()).toContain('Biz Cast')
    expect(w.text()).not.toContain('Tech Cast')
    expect(w.text()).not.toContain('No Cat')
  })

  it('shows no category picker when no show carries a category', async () => {
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([show('f-a', 'Acme Show')])
    const w = await mountView()
    expect(w.find('[data-testid="show-browse-category"]').exists()).toBe(false)
  })

  it('hides heading + back-to-Home when embedded', async () => {
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    const w = await mountView({ embedded: true })
    expect(w.find('[data-testid="browse-back-home"]').exists()).toBe(false)
    expect(w.find('h1').exists()).toBe(false)
  })

  /**
   * This tab already REPORTED its failure rather than pretending the corpus was empty — better than
   * Topics and People managed — but it still had nothing to show. The list you last loaded beats
   * "couldn't load" (#1909).
   */
  describe('offline (#1909)', () => {
    beforeEach(() => {
      readCached.mockReset().mockResolvedValue(null)
      writeCached.mockReset().mockResolvedValue(undefined)
    })

    it('shows the shows it last loaded', async () => {
      readCached.mockResolvedValue([show('f-c', 'Cached Cast')])
      vi.spyOn(api, 'getPodcasts').mockRejectedValue(new Error('offline'))
      const w = await mountView()
      await flushPromises()
      await flushPromises()
      expect(w.text(), 'the cached shows are gone').toContain('Cached Cast')
      expect(w.find('[data-testid="browse-stale-shows"]').exists()).toBe(true)
    })

    it('still reports failure when there is nothing cached', async () => {
      vi.spyOn(api, 'getPodcasts').mockRejectedValue(new Error('offline'))
      const w = await mountView()
      await flushPromises()
      await flushPromises()
      expect(w.find('[data-testid="browse-stale-shows"]').exists()).toBe(false)
      expect(w.text()).not.toContain('Cached Cast')
    })

    it('snapshots a successful load', async () => {
      vi.spyOn(api, 'getPodcasts').mockResolvedValue([show('f-a', 'Acme')])
      await mountView()
      await flushPromises()
      expect(writeCached.mock.calls.map((c) => c[0])).toContain('browse.shows')
    })
  })

  it('toggles between the grid and the list view (BS.2)', async () => {
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([show('f-a', 'Acme Show')])
    const w = await mountView()
    expect(w.find('[data-testid="show-browse-grid"]').exists()).toBe(true)
    expect(w.find('[data-testid="show-browse-list"]').exists()).toBe(false)

    // View is a ToolbarMenu circle now: open it, pick "list".
    await w.get('[data-testid="show-view"]').trigger('click')
    await w.get('[data-testid="show-view-opt-list"]').trigger('click')
    expect(w.find('[data-testid="show-browse-list"]').exists()).toBe(true)
    expect(w.find('[data-testid="show-browse-grid"]').exists()).toBe(false)
  })
})
