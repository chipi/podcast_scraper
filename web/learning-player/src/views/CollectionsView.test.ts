import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { Collection, CollectionDetail } from '../services/types'
import { useAuthStore } from '../stores/auth'
import CollectionsView from './CollectionsView.vue'

// Explicit, per-test cache. Without this, an earlier test's `writeCached` leaks into a later one:
// the store's cache FALLBACK then satisfies a test that is asserting the no-cache error path, and
// the failure looks like a product bug. Same isolation `favorites.test.ts` uses.
let cached: Record<string, unknown> = {}
vi.mock('../services/contentCache', () => ({
  isArrayCache: (v: unknown) => Array.isArray(v),
  hasArrayFields:
    (...f: string[]) =>
    (v: unknown) =>
      typeof v === 'object' &&
      v !== null &&
      !Array.isArray(v) &&
      f.every((k) => Array.isArray((v as Record<string, unknown>)[k])),
  readCached: async (k: string) => cached[k] ?? null,
  writeCached: async (k: string, v: unknown) => void (cached[k] = v),
  clearCached: async () => void (cached = {}),
  setCacheNamespace: () => {},
  CACHE_KEYS: ['library', 'favorites', 'queue', 'collections'],
}))

// jsdom does not implement `<dialog>`: without these, mounting the confirm throws.
if (!('showModal' in HTMLDialogElement.prototype)) {
  Object.assign(HTMLDialogElement.prototype, {
    showModal(this: HTMLDialogElement) { this.open = true },
    show(this: HTMLDialogElement) { this.open = true },
    close(this: HTMLDialogElement) { this.open = false; this.dispatchEvent(new Event('close')) },
  })
}

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: { template: '<div/>' } },
    { path: '/player/:slug', name: 'player', component: { template: '<div/>' } },
  ],
})

function col(over: Partial<Collection> = {}): Collection {
  return { id: 'col_1', name: 'AI takes', created_at: 1, count: 2, ...over }
}

const mountView = () => {
  setActivePinia(createPinia())
  return mount(CollectionsView, { global: { plugins: [i18n, router, createPinia()] } })
}

// File-level: every test starts with an empty cache, so no test inherits another's writes.
beforeEach(() => {
  cached = {}
})
afterEach(() => vi.restoreAllMocks())

describe('CollectionsView', () => {
  beforeEach(() => {
    vi.spyOn(api, 'getCollections').mockResolvedValue([col()])
  })

  it('lists collections with their counts', async () => {
    const w = mountView()
    await flushPromises()
    expect(w.text()).toContain('AI takes')
    expect(w.text()).toContain('2 items')
  })

  it('grid view (CO.3) shows cover tiles; a tile opens the board back in the list', async () => {
    vi.spyOn(api, 'getCollections').mockResolvedValue([
      col({ id: 'col_1', name: 'AI takes', cover_url: 'https://art/ep-1.jpg' }),
      col({ id: 'col_2', name: 'No cover', cover_url: null }),
    ])
    const getDetail = vi
      .spyOn(api, 'getCollection')
      .mockResolvedValue({ collection: col(), items: [] })
    const w = mountView()
    await flushPromises()

    await w.get('[data-testid="boards-view-grid"]').trigger('click')
    const tiles = w.findAll('[data-testid="board-tile"]')
    expect(tiles).toHaveLength(2)
    // Cover renders only where the collection has one; the other shows the placeholder.
    const covers = w.findAll('[data-testid="board-cover"]')
    expect(covers).toHaveLength(1)
    expect(covers[0].attributes('src')).toBe('https://art/ep-1.jpg')
    // The accordion is hidden in grid view.
    expect(w.find('[data-testid="collection-open"]').exists()).toBe(false)

    // Tapping a tile switches back to list and opens that board.
    await tiles[0].trigger('click')
    await flushPromises()
    expect(getDetail).toHaveBeenCalledWith('col_1')
    expect(w.find('[data-testid="collection-open"]').exists()).toBe(true)
  })

  it('creates a collection and prepends it', async () => {
    const create = vi
      .spyOn(api, 'createCollection')
      .mockResolvedValue(col({ id: 'col_2', name: 'ML', count: 0 }))
    const w = mountView()
    await flushPromises()
    await w.find('input[type="text"]').setValue('ML')
    await w.find('form').trigger('submit')
    await flushPromises()
    expect(create).toHaveBeenCalledWith('ML')
    expect(w.text()).toContain('ML')
  })

  it('opens a collection and renders its mixed items, and removes one', async () => {
    const detail: CollectionDetail = {
      collection: col(),
      items: [
        { kind: 'highlight', ref: 'h1', title: 'a line', deep_link: '/player/ep' },
        { kind: 'episode', ref: 'ep-x', title: 'An episode', deep_link: '/episode/ep-x' },
        { kind: 'link', ref: 'https://ex.com/p', title: 'A post', deep_link: 'https://ex.com/p' },
      ],
    }
    vi.spyOn(api, 'getCollection').mockResolvedValue(detail)
    const remove = vi.spyOn(api, 'removeFromCollection').mockResolvedValue(col({ count: 2 }))
    const w = mountView()
    await flushPromises()
    await w.findAll('button').find((b) => b.text().includes('AI takes'))!.trigger('click')
    await flushPromises()
    expect(w.text()).toContain('a line')
    expect(w.text()).toContain('An episode')
    // Link opens externally; in-app items use RouterLink.
    expect(w.find('a[href="https://ex.com/p"]').exists()).toBe(true)
    // Remove the first item → (kind, ref) identity.
    await w.findAll('[data-testid="collection-item-remove"]')[0].trigger('click')
    expect(remove).toHaveBeenCalledWith('col_1', 'highlight', 'h1')
  })

  it('hydrates episode titles, play-all queues them, and add-link pins a URL', async () => {
    const detail: CollectionDetail = {
      collection: col(),
      items: [
        { kind: 'episode', ref: 'ep-1' },
        { kind: 'episode', ref: 'ep-2' },
      ],
    }
    vi.spyOn(api, 'getCollection').mockResolvedValue(detail)
    vi.spyOn(api, 'getEpisode').mockResolvedValue({
      slug: 'ep-1', title: 'Ep One', podcast_title: 'Show',
    } as never)
    vi.spyOn(api, 'getQueue').mockResolvedValue([])
    vi.spyOn(api, 'putQueue').mockResolvedValue()
    const add = vi.spyOn(api, 'addToCollection').mockResolvedValue(col({ count: 3 }))
    const w = mountView()
    const auth = useAuthStore() // play-all is sign-in gated
    auth.user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    auth.loaded = true
    await flushPromises()
    await w.findAll('button').find((b) => b.text().includes('AI takes'))!.trigger('click')
    await flushPromises()
    expect(w.text()).toContain('Ep One') // episode title hydrated from the slug

    const push = vi.spyOn(router, 'push')
    await w.get('[data-testid="collection-play-all"]').trigger('click')
    await flushPromises()
    expect(push).toHaveBeenCalledWith({ name: 'player', params: { slug: 'ep-1' } })

    const linkForm = w.findAll('form').find((f) => f.find('[data-testid="collection-add-link"]').exists())!
    await linkForm.find('[data-testid="collection-add-link"]').setValue('https://ex.com/a')
    await linkForm.trigger('submit')
    expect(add).toHaveBeenCalledWith('col_1', { kind: 'link', ref: 'https://ex.com/a' })
  })

  it('deletes a collection, once confirmed', async () => {
    // This test used to tap ✕ and assert the delete. That is no longer what the ✕ does (#1594):
    // it opens a confirmation, and the delete happens on accept. Updated rather than deleted,
    // because the thing it covers — the list re-renders empty afterwards — is still true and
    // still worth asserting.
    const del = vi.spyOn(api, 'deleteCollection').mockResolvedValue([])
    const w = mountView()
    await flushPromises()
    await w.find('[aria-label="Delete collection"]').trigger('click')
    await w.get('[data-testid="confirm-accept"]').trigger('click')
    await flushPromises()
    expect(del).toHaveBeenCalledWith('col_1')
    expect(w.text()).toContain('No collections yet')
  })
})

describe('a failed load is not an empty library (#2004 item 13)', () => {
  it('shows a retryable error instead of the "no collections yet" empty state', async () => {
    // The screen a user saw after creating a collection elsewhere: `getCollections().catch(() => [])`
    // plus a `loaded` latch turned any failure — including the 401 the API layer used to
    // manufacture — into "you have no collections yet".
    vi.spyOn(api, 'getCollections').mockRejectedValue(new api.ApiError(401, 'nope'))
    const w = mountView()
    await flushPromises()
    expect(w.find('[data-testid="collections-load-error"]').exists()).toBe(true)
    expect(w.text()).not.toContain(en.collections.empty)
  })

  it('still shows the real empty state when the account genuinely has none', async () => {
    // The other half: the fix must not turn "you have none" into an error.
    vi.spyOn(api, 'getCollections').mockResolvedValue([])
    const w = mountView()
    await flushPromises()
    expect(w.find('[data-testid="collections-load-error"]').exists()).toBe(false)
    expect(w.text()).toContain(en.collections.empty)
  })

  it('recovers on retry', async () => {
    const spy = vi.spyOn(api, 'getCollections').mockRejectedValue(new api.ApiError(500, 'boom'))
    const w = mountView()
    await flushPromises()
    spy.mockResolvedValue([col()])
    await w.get('[data-testid="section-retry"]').trigger('click')
    await flushPromises()
    expect(w.find('[data-testid="collections-load-error"]').exists()).toBe(false)
    expect(w.text()).toContain('AI takes')
  })
})

describe('a cached copy beats a false empty state (#2013)', () => {
  it('renders the CACHED collections when the read fails', async () => {
    // The point of the store: a failed read must not render "you have no collections yet". That is
    // the lie that made created collections look lost.
    cached = { collections: { items: [col({ name: 'From cache' })] } }
    vi.spyOn(api, 'getCollections').mockRejectedValue(new api.ApiError(503, 'gateway'))
    const w = mountView()
    await flushPromises()
    expect(w.text()).toContain('From cache')
    expect(w.find('[data-testid="collections-load-error"]').exists()).toBe(false)
    expect(w.text()).not.toContain(en.collections.empty)
  })

  it('errors only when there is no answer AND no cache', async () => {
    cached = {}
    vi.spyOn(api, 'getCollections').mockRejectedValue(new api.ApiError(503, 'gateway'))
    const w = mountView()
    await flushPromises()
    expect(w.find('[data-testid="collections-load-error"]').exists()).toBe(true)
  })

  describe('deleting a collection is confirmed (#1594)', () => {
    it('the ✕ does NOT delete — it asks first', async () => {
      // The whole point. Before this, one tap on a control the size of a fingernail destroyed a
      // board and everything in it, with no dialog and no undo.
      const del = vi.spyOn(api, 'deleteCollection').mockResolvedValue([])
      vi.spyOn(api, 'getCollections').mockResolvedValue([col()])
      const w = mountView()
      useAuthStore().user = { id: 'u1' } as never
      await flushPromises()

      await w.get('[data-testid="collection-delete"]').trigger('click')
      await flushPromises()
      expect(del, 'tapping ✕ deleted immediately — the confirm is not wired').not.toHaveBeenCalled()
      expect(w.find('[data-testid="collection-delete-confirm"]').exists()).toBe(true)
    })

    it('confirming deletes exactly the collection that was asked about', async () => {
      const del = vi.spyOn(api, 'deleteCollection').mockResolvedValue([])
      vi.spyOn(api, 'getCollections').mockResolvedValue([col({ id: 'col_a' }), col({ id: 'col_b', name: 'Other' })])
      const w = mountView()
      useAuthStore().user = { id: 'u1' } as never
      await flushPromises()

      // The SECOND row, so a bug that always deletes the first is caught rather than passing.
      await w.findAll('[data-testid="collection-delete"]')[1].trigger('click')
      await w.get('[data-testid="confirm-accept"]').trigger('click')
      await flushPromises()
      expect(del).toHaveBeenCalledWith('col_b')
    })

    it('cancelling deletes nothing and forgets the pending id', async () => {
      const del = vi.spyOn(api, 'deleteCollection').mockResolvedValue([])
      vi.spyOn(api, 'getCollections').mockResolvedValue([col()])
      const w = mountView()
      useAuthStore().user = { id: 'u1' } as never
      await flushPromises()

      await w.get('[data-testid="collection-delete"]').trigger('click')
      await w.get('[data-testid="confirm-cancel"]').trigger('click')
      await flushPromises()
      expect(del).not.toHaveBeenCalled()

      // And the pending id must be cleared, or a later confirm — opened for a DIFFERENT row —
      // would delete the one abandoned here.
      await w.get('[data-testid="confirm-accept"]').trigger('click')
      await flushPromises()
      expect(del, 'a cancelled delete was still pending and fired later').not.toHaveBeenCalled()
    })
  })

describe('a board renders its items, not their refs', () => {
  it('a show renders its title and artwork, never the content hash', async () => {
    // The reported bug: a show's `ref` is a content hash, the server returns no title for it (its
    // own docstring says the client hydrates), and the row rendered `title ?? ref` — so the board
    // showed `sha256:68377a5abb…` where a person expects the show.
    vi.spyOn(api, 'getCollections').mockResolvedValue([col()])
    vi.spyOn(api, 'getCollection').mockResolvedValue({
      collection: col(),
      items: [{ kind: 'show', ref: 'sha256:68377a5abbfeba1c8' }],
    } as CollectionDetail)
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([
      {
        feed_id: 'sha256:68377a5abbfeba1c8',
        title: 'The Pragmatic Engineer',
        artwork_url: '/art/pe.jpg',
        image_url: null,
        description: null,
        episode_count: 42,
      },
    ] as never)
    const w = mountView()
    await flushPromises()
    await w.findAll('button').find((b) => b.text().includes('AI takes'))!.trigger('click')
    await flushPromises()

    expect(w.text()).toContain('The Pragmatic Engineer')
    expect(w.text(), 'the raw content hash reached the screen').not.toContain('sha256:')
    expect(w.get('[data-testid="collection-item"] img').attributes('src')).toBe('/art/pe.jpg')
  })

  it('an unresolvable item says so instead of showing a hash', async () => {
    // Falling back to the ref is what produced the bug. A failed lookup is a different fact from
    // "here is the item", and a hash communicates neither.
    vi.spyOn(api, 'getCollections').mockResolvedValue([col()])
    vi.spyOn(api, 'getCollection').mockResolvedValue({
      collection: col(),
      items: [{ kind: 'show', ref: 'sha256:deadbeefdeadbeef' }],
    } as CollectionDetail)
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([]) // the feed is gone
    const w = mountView()
    await flushPromises()
    await w.findAll('button').find((b) => b.text().includes('AI takes'))!.trigger('click')
    await flushPromises()

    const row = w.get('[data-testid="collection-item-title"]').text()
    expect(row).toContain("Couldn't load")
    expect(row, 'the full hash is not a label').not.toContain('deadbeefdeadbeef')
  })

  it('one unresolvable item does not blank the rest of the board', async () => {
    // Per-item best-effort: the whole point of resolving these in parallel with individual catches.
    vi.spyOn(api, 'getCollections').mockResolvedValue([col()])
    vi.spyOn(api, 'getCollection').mockResolvedValue({
      collection: col(),
      items: [
        { kind: 'episode', ref: 'ep-good' },
        { kind: 'episode', ref: 'ep-bad' },
      ],
    } as CollectionDetail)
    vi.spyOn(api, 'getEpisode').mockImplementation(async (slug: string) => {
      if (slug === 'ep-bad') throw new Error('gone')
      return { slug, title: 'Good Episode', podcast_title: 'Show' } as never
    })
    const w = mountView()
    await flushPromises()
    await w.findAll('button').find((b) => b.text().includes('AI takes'))!.trigger('click')
    await flushPromises()

    expect(w.text()).toContain('Good Episode')
    expect(w.findAll('[data-testid="collection-item"]')).toHaveLength(2)
  })
})

describe('collections open as an accordion (#2004 follow-up)', () => {
  async function openList() {
    vi.spyOn(api, 'getCollections').mockResolvedValue([
      col({ id: 'col_a', name: 'Tech', count: 2 }),
      col({ id: 'col_b', name: 'Investments', count: 1 }),
    ])
    // Non-empty: the items list only renders when there is something in it, so an empty board
    // would make these assertions pass or fail for the wrong reason.
    vi.spyOn(api, 'getCollection').mockImplementation(
      async (id: string) =>
        ({
          collection: col({ id, name: id === 'col_a' ? 'Tech' : 'Investments' }),
          items: [{ kind: 'topic', ref: 'topic:ai', title: 'ai' }],
        }) as never,
    )
    const w = mountView()
    await flushPromises()
    return w
  }

  it('tapping the open board CLOSES it', async () => {
    // It used to open a panel with a "Back" link and no way to collapse in place.
    const w = await openList()
    const rows = () => w.findAll('[data-testid="collection-open"]')
    await rows()[0].trigger('click')
    await flushPromises()
    expect(w.find('[data-testid="collection-items"]').exists()).toBe(true)

    await rows()[0].trigger('click')
    await flushPromises()
    expect(w.find('[data-testid="collection-items"]').exists(), 'it did not close').toBe(false)
  })

  it('opening another board moves the expansion — only ONE is open', async () => {
    // The old panel meant reaching a second board required closing the first.
    const w = await openList()
    const rows = () => w.findAll('[data-testid="collection-open"]')
    await rows()[0].trigger('click')
    await flushPromises()
    expect(rows()[0].attributes('aria-expanded')).toBe('true')

    await rows()[1].trigger('click')
    await flushPromises()
    expect(rows()[0].attributes('aria-expanded'), 'two boards open at once').toBe('false')
    expect(rows()[1].attributes('aria-expanded')).toBe('true')
    expect(w.findAll('[data-testid="collection-items"]')).toHaveLength(1)
  })

  it('the board renders INSIDE its own row, not in a panel above the list', async () => {
    // The duplication that made this confusing: the open board appeared twice, once as a panel and
    // once as a row below it.
    const w = await openList()
    await w.findAll('[data-testid="collection-open"]')[0].trigger('click')
    await flushPromises()
    const items = w.get('[data-testid="collection-items"]').element
    const row = w.findAll('[data-testid="collection-open"]')[0].element.closest('li')
    expect(row?.contains(items), 'the board is not inside its row').toBe(true)
    expect(w.text().match(/Tech/g)?.length, 'the board name appears twice').toBe(1)
  })
})
})
