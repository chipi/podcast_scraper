import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Same reason the queue store's tests mock it: otherwise this writes to real device storage and
// one test's cached favourites bleed into the next.
const cached: Record<string, unknown> = {}
vi.mock('../services/contentCache', () => ({
  readCached: async (k: string) => cached[k] ?? null,
  writeCached: async () => {},
  clearCached: async () => {},
  setCacheNamespace: () => {},
  CACHE_KEYS: ['library', 'favorites', 'queue'],
}))
import * as api from '../services/api'
import { ApiError } from '../services/api'
import * as outbox from '../services/outbox'
import type { EpisodeSummary } from '../services/types'
import { useFavoritesStore } from './favorites'

function episode(slug: string): EpisodeSummary {
  return {
    slug,
    title: `Episode ${slug}`,
    feed_id: 'p05',
    podcast_title: 'A Show',
    publish_date: null,
    duration_seconds: null,
    episode_image_url: null,
    feed_image_url: null,
    artwork_url: null,
    status: 'ready',
    summary_preview: null,
    summary_text: null,
  } as EpisodeSummary
}

beforeEach(() => {
  setActivePinia(createPinia())
  vi.spyOn(api, 'getFavorites').mockResolvedValue({ episodes: [], insights: [] })
})
afterEach(() => {
  vi.restoreAllMocks()
  for (const k of Object.keys(cached)) delete cached[k]
})

describe('favorites store', () => {
  it('takes the server response as authoritative on a successful toggle', async () => {
    vi.spyOn(api, 'addFavorite').mockResolvedValue({ episodes: [episode('a')], insights: [] })
    const f = useFavoritesStore()
    await f.toggle({ kind: 'episode', ref: 'a' })
    expect(f.has('episode', 'a')).toBe(true)
    expect(f.episodes).toHaveLength(1)
  })

  it('falls back to the cached copy and marks it stale (#1909)', async () => {
    cached.favorites = { episodes: [episode('a')], insights: [] }
    vi.spyOn(api, 'getFavorites').mockRejectedValue(new Error('offline'))
    const f = useFavoritesStore()
    await f.load()
    expect(f.has('episode', 'a')).toBe(true)
    expect(f.stale).toBe(true)
  })
})

/**
 * The offline seam. Before this the write went to the outbox and local state was left alone, so
 * the heart sat unchanged and the control read as dead — the app had recorded the intent and told
 * the user nothing (#1925 review).
 */
describe('favorites offline (#1910)', () => {
  it('flips the heart and queues the write when the request never lands', async () => {
    vi.spyOn(api, 'addFavorite').mockRejectedValue(new TypeError('Failed to fetch'))
    const enqueue = vi.spyOn(outbox, 'enqueue').mockImplementation(() => {})
    const f = useFavoritesStore()
    await f.toggle({ kind: 'episode', ref: 'a' })

    expect(f.has('episode', 'a')).toBe(true)
    expect(enqueue).toHaveBeenCalledWith({ op: 'favorite.add', kind: 'episode', ref: 'a' })
    // No EpisodeSummary exists offline, so the LIST waits for the server rather than showing a
    // card with a blank title.
    expect(f.episodes).toHaveLength(0)
  })

  it('removes from the list too — an unfavourite we can represent exactly', async () => {
    vi.spyOn(api, 'getFavorites').mockResolvedValue({ episodes: [episode('a')], insights: [] })
    vi.spyOn(api, 'removeFavorite').mockRejectedValue(new TypeError('Failed to fetch'))
    const enqueue = vi.spyOn(outbox, 'enqueue').mockImplementation(() => {})
    const f = useFavoritesStore()
    await f.load()

    await f.toggle({ kind: 'episode', ref: 'a' })
    expect(f.has('episode', 'a')).toBe(false)
    expect(f.episodes).toHaveLength(0)
    expect(enqueue).toHaveBeenCalledWith({ op: 'favorite.remove', kind: 'episode', ref: 'a' })
  })

  it('does NOT queue a server refusal — replaying it would just fail again', async () => {
    vi.spyOn(api, 'addFavorite').mockRejectedValue(new ApiError(404, 'gone'))
    const enqueue = vi.spyOn(outbox, 'enqueue').mockImplementation(() => {})
    const f = useFavoritesStore()
    await f.toggle({ kind: 'episode', ref: 'a' })

    expect(enqueue).not.toHaveBeenCalled()
    // ...and the heart must NOT flip: the server answered, and the answer was no.
    expect(f.has('episode', 'a')).toBe(false)
  })

  it('drops the pending flip once the server answers', async () => {
    vi.spyOn(api, 'addFavorite').mockRejectedValue(new TypeError('Failed to fetch'))
    vi.spyOn(outbox, 'enqueue').mockImplementation(() => {})
    const f = useFavoritesStore()
    await f.toggle({ kind: 'episode', ref: 'a' })
    expect(f.has('episode', 'a')).toBe(true)

    // The reconnect path flushes the outbox and THEN reloads, so this read already includes it.
    vi.spyOn(api, 'getFavorites').mockResolvedValue({ episodes: [episode('a')], insights: [] })
    await f.load()
    expect(f.pendingFlips).toEqual({})
    expect(f.has('episode', 'a')).toBe(true)
  })
})
