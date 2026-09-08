import { beforeEach, describe, expect, it, vi } from 'vitest'
import { anyStale, resetStaleness, useSectionState } from './useSectionState'

const readCached = vi.fn(
  async (_k: string, _isValid?: (v: unknown) => boolean): Promise<unknown> => null,
)
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
  readCached: (k: string, isValid?: (v: unknown) => boolean) => readCached(k, isValid),
  writeCached: (k: string, v: unknown) => writeCached(k, v),
}))

/**
 * Direct tests for the #1591 primitive. It is used by seven sections, so its behaviour under
 * failure is the behaviour of the whole Home page — but until now it was only covered indirectly
 * through HomeView, which tests the composition rather than the contract.
 */
describe('useSectionState', () => {
  it('starts in loading, before anything is fetched', () => {
    const s = useSectionState<string[]>([])
    expect(s.phase.value).toBe('loading')
    expect(s.isLoading.value).toBe(true)
    expect(s.data.value).toEqual([])
  })

  it('records data and becomes ready on success', async () => {
    const s = useSectionState<string[]>([])
    await s.load(async () => ['a', 'b'])
    expect(s.phase.value).toBe('ready')
    expect(s.isReady.value).toBe(true)
    expect(s.data.value).toEqual(['a', 'b'])
  })

  it('a successful EMPTY result is ready, not an error', async () => {
    // The distinction the whole issue turns on: "the system has nothing" and "the request failed"
    // are different states that used to render identically.
    const s = useSectionState<string[]>([])
    await s.load(async () => [])
    expect(s.phase.value).toBe('ready')
    expect(s.isError.value).toBe(false)
  })

  it('a rejection becomes an error and does NOT collapse into the initial value', async () => {
    // `.catch(() => [])` — collapsing failure into emptiness — is the defect this replaces.
    const s = useSectionState<string[]>(['seed'])
    await s.load(async () => {
      throw new Error('boom')
    })
    expect(s.phase.value).toBe('error')
    expect(s.isError.value).toBe(true)
    expect(s.isReady.value).toBe(false)
    // Previous data is left alone rather than being wiped by the failure.
    expect(s.data.value).toEqual(['seed'])
  })

  it('does not reject — callers use `await load()` without try/catch', async () => {
    // Every call site is `void load()` or `await load()`. If this rethrew, each would need its own
    // handler and the ones using `void` would raise unhandled rejections.
    const s = useSectionState<string[]>([])
    await expect(
      s.load(async () => {
        throw new Error('boom')
      }),
    ).resolves.toBeUndefined()
  })

  it('retrying after a failure recovers', async () => {
    const s = useSectionState<string[]>([])
    const fetcher = vi
      .fn<() => Promise<string[]>>()
      .mockRejectedValueOnce(new Error('boom'))
      .mockResolvedValueOnce(['recovered'])

    await s.load(fetcher)
    expect(s.phase.value).toBe('error')

    await s.load(fetcher)
    expect(s.phase.value).toBe('ready')
    expect(s.data.value).toEqual(['recovered'])
  })

  it('returns to loading while a retry is in flight', async () => {
    const s = useSectionState<string[]>([])
    await s.load(async () => {
      throw new Error('boom')
    })
    expect(s.phase.value).toBe('error')

    let release!: (v: string[]) => void
    const pending = s.load(() => new Promise<string[]>((res) => (release = res)))
    // The error must clear immediately, or a retry looks like it did nothing.
    expect(s.phase.value).toBe('loading')
    release(['ok'])
    await pending
    expect(s.phase.value).toBe('ready')
  })
})

/**
 * #1909 scoped "snapshot-on-successful-load + hydrate-then-revalidate for library, queue, favourites
 * AND the Home rails". The rails were the half that never landed, so with no network Home rendered a
 * column of identical "Couldn't load this right now" cards instead of what the user had already
 * loaded. The requirement was "everything I loaded last time is still there, just stale".
 */
describe('useSectionState with a cacheKey (#1909)', () => {
  beforeEach(() => {
    readCached.mockReset().mockResolvedValue(null)
    writeCached.mockReset().mockResolvedValue(undefined)
    resetStaleness()
  })

  it('paints the snapshot BEFORE the fetch answers', async () => {
    readCached.mockResolvedValue(['from disk'])
    const s = useSectionState<string[]>([], { cacheKey: 'home.test' })
    let release!: (v: string[]) => void
    const pending = s.load(() => new Promise<string[]>((r) => (release = r)))
    // Two microtask turns: one for the settle-wrapper, one for the cache read.
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()
    expect(s.data.value, 'the snapshot did not paint while the fetch was in flight').toEqual([
      'from disk',
    ])
    expect(s.phase.value).toBe('ready')
    expect(s.stale.value).toBe(true)
    release(['fresh'])
    await pending
  })

  it('a fast response never flickers through the snapshot first', async () => {
    // The snapshot fills a WAIT. When there is no wait it must not create one, and it must not
    // paint content that is already obsolete — a rail that shows last week's rows for a frame
    // before the fresh ones is a flicker backwards.
    let releaseCache!: (v: string[]) => void
    readCached.mockReturnValue(new Promise<string[]>((r) => (releaseCache = r)))
    const s = useSectionState<string[]>([], { cacheKey: 'home.test' })
    await s.load(async () => ['fresh'])
    expect(s.data.value).toEqual(['fresh'])
    expect(s.stale.value, 'a won race still marked the section stale').toBe(false)
    releaseCache(['from disk'])
    await Promise.resolve()
    expect(s.data.value, 'a late snapshot overwrote fresh data').toEqual(['fresh'])
  })

  it('fresh data supersedes the snapshot and is no longer stale', async () => {
    readCached.mockResolvedValue(['from disk'])
    const s = useSectionState<string[]>([], { cacheKey: 'home.test' })
    await s.load(async () => ['fresh'])
    expect(s.data.value).toEqual(['fresh'])
    expect(s.stale.value).toBe(false)
    expect(anyStale.value).toBe(false)
  })

  it('a failure with a snapshot KEEPS the content and reports stale, never error', async () => {
    // The governing rule of the arc: only a 401/403 may destroy cached state; a transport error
    // never may. An error card over content the user already had is exactly that destruction.
    readCached.mockResolvedValue(['from disk'])
    const s = useSectionState<string[]>([], { cacheKey: 'home.test' })
    await s.load(async () => {
      throw new Error('offline')
    })
    expect(s.phase.value, 'a cached section still showed an error card').toBe('ready')
    expect(s.isError.value).toBe(false)
    expect(s.data.value).toEqual(['from disk'])
    expect(s.stale.value).toBe(true)
    expect(anyStale.value).toBe(true)
  })

  it('ignores a cached value that no longer matches the section\'s shape', async () => {
    // A section's cached value goes into `data` and then into a `v-for`. A stale-format entry from
    // an older build does not degrade the rail, it makes the render throw — so it is a MISS, and
    // the fetch already in flight covers it.
    // The guard is applied HERE rather than in the shared factory, whose extra await would shift
    // the microtask counts the ordering tests above depend on. If the section stops passing a
    // validator, this mock returns the raw object and the assertions below fail — which is the
    // break being guarded against.
    readCached.mockImplementation(async (_k, isValid) => {
      const stored = { not: 'an array' }
      return isValid && !isValid(stored) ? null : stored
    })
    const s = useSectionState<string[]>([], { cacheKey: 'home.test' })
    await s.load(async () => {
      throw new Error('offline')
    })
    expect(s.data.value, 'a wrong-shaped snapshot was painted').toEqual([])
    expect(s.phase.value, 'it pretended to have content').toBe('error')
  })

  it('a failure with NOTHING cached is still an error — that has not changed', async () => {
    readCached.mockResolvedValue(null)
    const s = useSectionState<string[]>([], { cacheKey: 'home.test' })
    await s.load(async () => {
      throw new Error('offline')
    })
    expect(s.phase.value).toBe('error')
    expect(anyStale.value).toBe(false)
  })

  it('a failure AFTER a success keeps the fresh data rather than blanking it', async () => {
    const s = useSectionState<string[]>([], { cacheKey: 'home.test' })
    await s.load(async () => ['fresh'])
    await s.load(async () => {
      throw new Error('offline')
    })
    expect(s.data.value).toEqual(['fresh'])
    expect(s.phase.value).toBe('ready')
    expect(s.stale.value).toBe(true)
  })

  it('snapshots every successful load', async () => {
    const s = useSectionState<string[]>([], { cacheKey: 'home.test' })
    await s.load(async () => ['a'])
    expect(writeCached).toHaveBeenCalledWith('home.test', ['a'])
  })

  it('revalidating with content on screen does NOT flash a skeleton', async () => {
    // Returning to Home must not blank a correct rail behind a loading state.
    const s = useSectionState<string[]>([], { cacheKey: 'home.test' })
    await s.load(async () => ['a'])
    let release!: (v: string[]) => void
    const pending = s.load(() => new Promise<string[]>((r) => (release = r)))
    expect(s.phase.value, 'a revalidation dropped back to loading').toBe('ready')
    release(['b'])
    await pending
  })

  it('reads the snapshot ONCE, not on every revalidation', async () => {
    const s = useSectionState<string[]>([], { cacheKey: 'home.test' })
    await s.load(async () => ['a'])
    await s.load(async () => ['b'])
    expect(readCached).toHaveBeenCalledTimes(1)
  })

  it('a section with no cacheKey never touches the cache', async () => {
    const s = useSectionState<string[]>([])
    await s.load(async () => ['a'])
    expect(readCached).not.toHaveBeenCalled()
    expect(writeCached).not.toHaveBeenCalled()
  })

  it('the stale tally is per-key and clears when a key recovers', async () => {
    readCached.mockResolvedValue(['disk'])
    const a = useSectionState<string[]>([], { cacheKey: 'home.a' })
    const b = useSectionState<string[]>([], { cacheKey: 'home.b' })
    await a.load(async () => {
      throw new Error('offline')
    })
    await b.load(async () => {
      throw new Error('offline')
    })
    expect(anyStale.value).toBe(true)
    await a.load(async () => ['fresh'])
    expect(anyStale.value, 'one rail recovering cleared the whole tally').toBe(true)
    await b.load(async () => ['fresh'])
    expect(anyStale.value).toBe(false)
  })
})
