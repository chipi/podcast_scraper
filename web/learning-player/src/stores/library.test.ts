import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../services/api'
import { useLibraryStore } from './library'
import type { LibraryItem } from '../services/types'

/**
 * The store had NO tests, while every sibling per-user store has them — and it is the one holding
 * the optimistic toggle, the cache fallback and the identity reset.
 *
 * What that cost: Library rendered "You're not following any shows yet" plus six shows to follow,
 * to a user who WAS following shows. Reported from prod. The load path could not tell "you follow
 * nothing" from "we could not ask", and nothing here would have noticed.
 */

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

const enqueued: unknown[] = []
vi.mock('../services/outbox', async (orig) => {
  const actual = await orig<typeof import('../services/outbox')>()
  return { ...actual, enqueue: (op: unknown) => void enqueued.push(op) }
})

function item(feedId: string): LibraryItem {
  return { feed_id: feedId, feed_url: null, title: feedId, added_at: null }
}

beforeEach(() => {
  setActivePinia(createPinia())
  cached = {}
  enqueued.length = 0
  vi.restoreAllMocks()
})

describe('library store — reading follows', () => {
  it('a fresh load populates and caches', async () => {
    vi.spyOn(api, 'getLibrary').mockResolvedValue([item('f1')])
    const s = useLibraryStore()
    await s.load()
    expect(s.items).toHaveLength(1)
    expect(s.loaded).toBe(true)
    expect(s.stale).toBe(false)
    expect(s.has('f1')).toBe(true)
  })

  it('falls back to the cache and says it is stale', async () => {
    cached.library = [item('f1')]
    vi.spyOn(api, 'getLibrary').mockRejectedValue(new Error('offline'))
    const s = useLibraryStore()
    await s.load()
    expect(s.items).toHaveLength(1)
    expect(s.loaded).toBe(true)
    expect(s.stale).toBe(true)
  })

  it('with NO answer and NO cache, `loaded` stays false', async () => {
    // The bug, in one assertion. `loaded` is the only thing separating "you follow nothing" from
    // "we could not ask" — Library reads it to choose between the follows grid and an empty state
    // that offers six shows to follow. Latching it here turns a failed fetch into a confident,
    // wrong claim about the user's own data.
    vi.spyOn(api, 'getLibrary').mockRejectedValue(new Error('offline'))
    const s = useLibraryStore()
    await s.load()
    expect(s.loaded, 'a failed load claimed to be an authoritative empty list').toBe(false)
    expect(s.items).toEqual([])
  })

  it('a 401 never caches an empty library', async () => {
    // `getLibrary` used to map 401 → [], which the store took as fresh truth and wrote to the
    // per-user cache. An expired session did not just render "you follow nothing" — it PERSISTED
    // that answer, so the offline fallback kept repeating it.
    vi.spyOn(api, 'getLibrary').mockRejectedValue(new api.ApiError(401, 'unauthorized'))
    const s = useLibraryStore()
    await s.load()
    expect(s.loaded).toBe(false)
    expect(cached.library, 'an auth failure was written to the cache as real data').toBeUndefined()
  })

  it('ensureLoaded retries after a failure instead of latching', async () => {
    const spy = vi.spyOn(api, 'getLibrary').mockRejectedValueOnce(new Error('offline'))
    const s = useLibraryStore()
    await s.ensureLoaded()
    expect(s.loaded).toBe(false)

    spy.mockResolvedValueOnce([item('f1')])
    await s.ensureLoaded()
    expect(s.loaded).toBe(true)
    expect(s.has('f1')).toBe(true)
  })
})

describe('library store — toggling a follow', () => {
  it('flips immediately, then takes the server list', async () => {
    vi.spyOn(api, 'followShow').mockResolvedValue([item('f1')])
    const s = useLibraryStore()
    const pending = s.toggle('f1')
    expect(s.has('f1'), 'the button did not respond before the request settled').toBe(true)
    await pending
    expect(s.items).toEqual([item('f1')])
  })

  it('reverts when the server REFUSES', async () => {
    vi.spyOn(api, 'followShow').mockRejectedValue(new api.ApiError(422, 'nope'))
    const s = useLibraryStore()
    await s.toggle('f1')
    expect(s.has('f1'), 'the UI claimed a subscription the server rejected').toBe(false)
    expect(enqueued, 'a refused write was queued for replay').toHaveLength(0)
  })

  it('keeps the flip and QUEUES it when the request never landed', async () => {
    // A 502 is not the server saying no. Reverting on any error destroyed a follow over a blip.
    vi.spyOn(api, 'followShow').mockRejectedValue(new api.ApiError(502, 'bad gateway'))
    const s = useLibraryStore()
    await s.toggle('f1')
    expect(s.has('f1'), 'a transient failure discarded the follow').toBe(true)
    expect(enqueued).toHaveLength(1)
  })

  it('a dead session is queued, not discarded', async () => {
    // Signing in repairs it; throwing the intent away loses what the user asked for.
    vi.spyOn(api, 'followShow').mockRejectedValue(new api.ApiError(401, 'unauthorized'))
    const s = useLibraryStore()
    await s.toggle('f1')
    expect(s.has('f1')).toBe(true)
    expect(enqueued).toHaveLength(1)
  })
})

describe('library store — identity', () => {
  it('$reset clears one account before the next reads it', () => {
    const s = useLibraryStore()
    s.items = [item('f1')]
    s.loaded = true
    s.$reset()
    expect(s.items).toEqual([])
    expect(s.loaded).toBe(false)
  })
})
