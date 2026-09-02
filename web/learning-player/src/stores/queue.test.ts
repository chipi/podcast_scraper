import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The cache writes to real device storage otherwise, and one test's cached queue then bleeds into
// the next — which is exactly how this mock came to be needed.
// Seeded by the stale tests; null everywhere else.
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
import { useQueueStore } from './queue'

// A fake server, not bare resolved values: the store now takes the RESPONSE as truth, so a mock
// that returns nothing meaningful would make every assertion vacuous. These mirror the real route
// semantics (see tests/integration/server/test_app_user_state_routes.py).
let server: string[] = []

beforeEach(() => {
  setActivePinia(createPinia())
  server = []
  vi.spyOn(api, 'putQueue').mockImplementation(async (items) => {
    server = [...items]
  })
  vi.spyOn(api, 'addQueueItem').mockImplementation(async (slug, after) => {
    if (after == null && server.includes(slug)) return [...server]
    server = server.filter((x) => x !== slug)
    if (after == null) server.push(slug)
    else server.splice(server.indexOf(after) + 1, 0, slug)
    return [...server]
  })
  vi.spyOn(api, 'removeQueueItem').mockImplementation(async (slug) => {
    server = server.filter((x) => x !== slug)
    return [...server]
  })
  // Mutations ensureLoaded() first (guards against a late load clobbering an optimistic add).
  vi.spyOn(api, 'getQueue').mockImplementation(async () => [...server])
})
afterEach(() => vi.restoreAllMocks())

describe('queue store', () => {
  it('load() pulls items from the API', async () => {
    vi.spyOn(api, 'getQueue').mockResolvedValue(['a', 'b'])
    const q = useQueueStore()
    await q.load()
    expect(q.items).toEqual(['a', 'b'])
    expect(q.count).toBe(2)
  })

  it('add() appends once and persists', async () => {
    const q = useQueueStore()
    await q.add('a')
    await q.add('a') // idempotent
    await q.add('b')
    expect(q.items).toEqual(['a', 'b'])
    // ITEM-level, not a whole-list PUT (#1925): that is what makes an offline add replay-safe.
    expect(api.addQueueItem).toHaveBeenCalledWith('a')
    expect(api.putQueue).not.toHaveBeenCalled()
  })

  it('toggle() adds then removes', async () => {
    const q = useQueueStore()
    await q.toggle('a')
    expect(q.has('a')).toBe(true)
    await q.toggle('a')
    expect(q.has('a')).toBe(false)
  })

  it('playNext() inserts right after the current slug', async () => {
    server = ['a', 'b', 'c']
    const q = useQueueStore()
    q.loaded = true // pre-loaded; ensureLoaded is a no-op so the seeded items survive
    q.items = ['a', 'b', 'c']
    await q.playNext('z', 'a')
    expect(q.items).toEqual(['a', 'z', 'b', 'c'])
    expect(api.addQueueItem).toHaveBeenCalledWith('z', 'a')
  })

  it('move() reorders within bounds', async () => {
    const q = useQueueStore()
    q.loaded = true
    q.items = ['a', 'b', 'c']
    await q.move('c', -1)
    expect(q.items).toEqual(['a', 'c', 'b'])
    await q.move('a', -1) // no-op at the top
    expect(q.items).toEqual(['a', 'c', 'b'])
  })

  it('add() during an in-flight load is not clobbered by the late load (RFC-099 race)', async () => {
    // The initial load()'s GET resolves LATE. A mutation must wait for it (ensureLoaded) and
    // append to the loaded state — otherwise the late load overwrites items and drops the add
    // ("queue empty" after add). Regression for the queue-persist race.
    let resolveGet: (v: string[]) => void = () => {}
    vi.spyOn(api, 'getQueue').mockReturnValue(
      new Promise<string[]>((r) => {
        resolveGet = r
      }),
    )
    server = ['a']
    const q = useQueueStore()
    const addP = q.add('x') // triggers load(); blocks on ensureLoaded()
    resolveGet(['a']) // the server already had ['a']
    await addP
    expect(q.items).toEqual(['a', 'x']) // appended to loaded state, not clobbered to []
    expect(api.addQueueItem).toHaveBeenCalledWith('x')
  })

  it('nextAfter() returns the auto-advance target', () => {
    const q = useQueueStore()
    q.items = ['a', 'b', 'c']
    expect(q.nextAfter('a')).toBe('b')
    expect(q.nextAfter('c')).toBeNull()
    expect(q.nextAfter('zzz')).toBeNull()
  })

  // #1906 — offline honesty. A rejected PUT used to leave the optimistic mutation in place, so
  // the app showed a queued episode the server never received; it vanished at the next launch.

  it('move() restores the original order when the write fails', async () => {
    const q = useQueueStore()
    q.loaded = true
    q.items = ['a', 'b', 'c']
    vi.spyOn(api, 'putQueue').mockRejectedValue(new TypeError('Failed to fetch'))
    await expect(q.move('c', -1)).resolves.toBe(false)
    expect(q.items).toEqual(['a', 'b', 'c'])
  })

  /**
   * The contract CHANGED with item-level operations (#1925), and this is the change:
   *
   * A whole-list write had to revert on failure — the app would otherwise show a queued episode
   * the server never received, gone at the next launch. An item-level write does not, because it
   * goes to the outbox and is replayed. Reverting would now be the dishonest option: it would
   * discard a tap the app has in fact recorded.
   */
  it('keeps the optimistic order and queues the write when the request never lands', async () => {
    const q = useQueueStore()
    q.loaded = true
    q.items = ['a', 'b', 'c']
    vi.spyOn(api, 'addQueueItem').mockRejectedValue(new TypeError('Failed to fetch'))
    const enqueue = vi.spyOn(outbox, 'enqueue').mockImplementation(() => {})

    await expect(q.playNext('c', 'a')).resolves.toBe(true)
    expect(q.items).toEqual(['a', 'c', 'b'])
    expect(enqueue).toHaveBeenCalledWith({ op: 'queue.add', slug: 'c', after: 'a' })
  })

  it('reverts and reports failure when the SERVER refuses — that is an answer', async () => {
    const q = useQueueStore()
    q.loaded = true
    q.items = ['a', 'b']
    vi.spyOn(api, 'removeQueueItem').mockRejectedValue(new ApiError(404, 'gone'))
    const enqueue = vi.spyOn(outbox, 'enqueue').mockImplementation(() => {})

    await expect(q.remove('a')).resolves.toBe(false)
    expect(q.items).toEqual(['a', 'b'])
    expect(enqueue).not.toHaveBeenCalled()
  })

  it('a successful write reports success', async () => {
    const q = useQueueStore()
    await expect(q.add('a')).resolves.toBe(true)
    await expect(q.toggle('a')).resolves.toBe(true)
  })

  // #1906 — a cold offline start. Before this, load() rejected and the rejection travelled
  // through every mutation into `void store.toggle()` call sites.

  it('never throws into a void call site, even with no loaded queue', async () => {
    vi.spyOn(api, 'getQueue').mockRejectedValue(new TypeError('Failed to fetch'))
    const q = useQueueStore()
    await expect(q.add('a')).resolves.toBeDefined()
    await expect(q.toggle('a')).resolves.toBeDefined()
    await expect(q.remove('a')).resolves.toBeDefined()
    await expect(q.move('a', 1)).resolves.toBe(false)
  })

  it('never PUTs from an unknown baseline — that would delete the server queue', async () => {
    // `move` sends the WHOLE list. Writing from an empty local array because the GET failed
    // would wipe whatever the user actually had. (add/remove are item-level and exempt.)
    vi.spyOn(api, 'getQueue').mockRejectedValue(new TypeError('Failed to fetch'))
    const q = useQueueStore()
    await q.move('a', 1)
    expect(api.putQueue).not.toHaveBeenCalled()
  })

  it('load() reports success and failure rather than rejecting', async () => {
    const q = useQueueStore()
    await expect(q.load()).resolves.toBe(true)
    vi.spyOn(api, 'getQueue').mockRejectedValue(new TypeError('Failed to fetch'))
    q.loaded = false
    await expect(q.load()).resolves.toBe(false)
  })
})

/**
 * A queue restored from cache is READABLE, and — since #1925 — writable at the ITEM level.
 *
 * `move` still is not: it sends the whole list, so reordering from a baseline we never revalidated
 * would delete whatever the server actually holds. Add and remove are idempotent one-slug
 * operations that replay safely, so refusing them was costing the user a working control for no
 * safety it bought. That distinction is load-bearing for data loss, not UI polish.
 */
describe('stale queue guard (#1909)', () => {
  beforeEach(() => {
    cached.queue = ['a', 'b']
  })
  afterEach(() => {
    delete cached.queue
  })

  it('falls back to the cached copy and marks it stale', async () => {
    vi.spyOn(api, 'getQueue').mockRejectedValue(new Error('offline'))
    const q = useQueueStore()
    expect(await q.load()).toBe(false)
    expect(q.items).toEqual(['a', 'b'])
    expect(q.loaded).toBe(true)
    expect(q.stale).toBe(true)
  })

  it('refuses REORDERING from a stale baseline, and never PUTs', async () => {
    vi.spyOn(api, 'getQueue').mockRejectedValue(new Error('offline'))
    const q = useQueueStore()
    await q.load()

    expect(await q.move('a', 1)).toBe(false)
    expect(api.putQueue).not.toHaveBeenCalled()
    // ...and the visible list is exactly what it was, not a half-applied optimistic edit.
    expect(q.items).toEqual(['a', 'b'])
  })

  it('still accepts add and remove — they are item-level and replay safely', async () => {
    vi.spyOn(api, 'getQueue').mockRejectedValue(new Error('offline'))
    vi.spyOn(api, 'addQueueItem').mockRejectedValue(new TypeError('Failed to fetch'))
    vi.spyOn(api, 'removeQueueItem').mockRejectedValue(new TypeError('Failed to fetch'))
    const enqueue = vi.spyOn(outbox, 'enqueue').mockImplementation(() => {})
    const q = useQueueStore()
    await q.load()

    expect(await q.add('c')).toBe(true)
    expect(q.items).toEqual(['a', 'b', 'c'])
    expect(await q.remove('a')).toBe(true)
    expect(q.items).toEqual(['b', 'c'])
    expect(enqueue).toHaveBeenCalledTimes(2)
    expect(api.putQueue).not.toHaveBeenCalled()
  })

  it('is writable again once a real load lands', async () => {
    vi.spyOn(api, 'getQueue').mockRejectedValueOnce(new Error('offline'))
    const q = useQueueStore()
    await q.load()
    expect(q.stale).toBe(true)

    server = ['a', 'b']
    expect(await q.load()).toBe(true)
    expect(q.stale).toBe(false)
    expect(await q.move('b', -1)).toBe(true)
    expect(api.putQueue).toHaveBeenCalledWith(['b', 'a'])
  })
})
