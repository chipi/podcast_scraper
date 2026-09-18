import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { Collection } from '../services/types'
import CollectionsTeaser from './CollectionsTeaser.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [{ path: '/library', name: 'library', component: { template: '<div/>' } }],
})

const board = (over: Partial<Collection> = {}): Collection => ({
  id: 'c1', name: 'A board', created_at: 100, updated_at: 100, count: 2,
  cover_url: 'http://x/cover.png', position: null, ...over,
})

function mountTeaser() {
  return mount(CollectionsTeaser, { global: { plugins: [i18n, router] } })
}

describe('CollectionsTeaser (Home)', () => {
  beforeEach(() => setActivePinia(createPinia()))
  afterEach(() => vi.restoreAllMocks())

  it('shows the most recently CHANGED boards first, capped at four', async () => {
    // On Home the question is "what am I working on", and the board you added to yesterday
    // answers it — not the alphabetical first, and not the manual board order, which is the
    // Boards tab's own affordance.
    vi.spyOn(api, 'getCollections').mockResolvedValue([
      board({ id: 'old', name: 'Oldest', updated_at: 1 }),
      board({ id: 'newest', name: 'Newest', updated_at: 900 }),
      board({ id: 'mid', name: 'Middle', updated_at: 500 }),
      board({ id: 'x4', name: 'Fourth', updated_at: 400 }),
      board({ id: 'x5', name: 'Fifth', updated_at: 300 }),
    ])
    const w = mountTeaser()
    await flushPromises()

    const tiles = w.findAll('[data-testid="home-collection-tile"]')
    expect(tiles).toHaveLength(4)
    expect(tiles[0].text()).toContain('Newest')
    expect(w.text(), 'the oldest board was shown despite five existing').not.toContain('Oldest')
  })

  it('falls back to created_at when a board has never been changed', async () => {
    vi.spyOn(api, 'getCollections').mockResolvedValue([
      board({ id: 'never', name: 'Never touched', created_at: 900, updated_at: undefined }),
      board({ id: 'older', name: 'Older', created_at: 1, updated_at: 5 }),
    ])
    const w = mountTeaser()
    await flushPromises()
    expect(w.findAll('[data-testid="home-collection-tile"]')[0].text()).toContain('Never touched')
  })

  it('renders a flat tile rather than a broken image when a board has no cover', async () => {
    // `cover_url` is derived from the first member, so a brand-new board has none. A broken-image
    // glyph or an icon pretending to be artwork would both be worse than an honest empty square.
    vi.spyOn(api, 'getCollections').mockResolvedValue([
      board({ id: 'bare', name: 'Bare board', cover_url: null, count: 0 }),
    ])
    const w = mountTeaser()
    await flushPromises()
    expect(w.find('[data-testid="home-collection-tile"] img').exists()).toBe(false)
    expect(w.text()).toContain('Bare board')
  })

  it('deep-links each tile to its own board, not just to the tab', async () => {
    // The Boards list is an accordion; landing on it collapsed makes the tile feel inert.
    vi.spyOn(api, 'getCollections').mockResolvedValue([board({ id: 'c7', name: 'Seven' })])
    const w = mountTeaser()
    await flushPromises()
    const href = w.find('[data-testid="home-collection-tile"]').attributes('href')
    expect(href).toContain('tab=collections')
    expect(href).toContain('board=c7')
  })

  it('renders nothing at all when the user has no boards', async () => {
    vi.spyOn(api, 'getCollections').mockResolvedValue([])
    const w = mountTeaser()
    await flushPromises()
    expect(w.find('[data-testid="home-collections-teaser"]').exists()).toBe(false)
  })
})
