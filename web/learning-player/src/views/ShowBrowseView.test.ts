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

  it('filters by name and sorts by episode count', async () => {
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([
      { feed_id: 'f-a', title: 'Acme Show', artwork_url: null, image_url: null, description: null, episode_count: 2 },
      { feed_id: 'f-z', title: 'Zebra Cast', artwork_url: null, image_url: null, description: null, episode_count: 40 },
    ])
    const w = await mountView()
    // Sort by most episodes → Zebra (40) leads Acme (2).
    await w.get('[data-testid="show-browse-sort"]').setValue('episodes')
    expect(w.findAll('a[href^="/podcast/"]')[0].attributes('href')).toBe('/podcast/f-z')
    // Filter narrows to matches only.
    await w.get('[data-testid="show-browse-search"]').setValue('acme')
    const links = w.findAll('a[href^="/podcast/"]')
    expect(links.length).toBe(1)
    expect(links[0].attributes('href')).toBe('/podcast/f-a')
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
})
