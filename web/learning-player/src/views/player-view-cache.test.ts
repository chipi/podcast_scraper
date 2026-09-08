import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  MAX,
  PERSIST_DEBOUNCE_MS,
  PERSIST_MAX,
  PLAYER_SNAPSHOT_KEY,
  clearPlayerViewCache,
  flushPlayerViewSnapshots,
  getPlayerViewSnapshot,
  hydratePlayerViewCache,
  setPlayerViewSnapshot,
  type PlayerViewSnapshot,
} from './player-view-cache'

const readCached = vi.fn(async (_k: string): Promise<unknown> => null)
const writeCached = vi.fn(async (_k: string, _v: unknown): Promise<void> => {})
vi.mock('../services/contentCache', () => ({
  readCached: (k: string) => readCached(k),
  writeCached: (k: string, v: unknown) => writeCached(k, v),
}))

/** Slugs enough to overflow whatever the bound is, so the test does not restate the constant. */
const overflow = (n: number) => Array.from({ length: n }, (_, i) => `ep-${i}`)

function snap(title: string): PlayerViewSnapshot {
  return {
    episode: { title } as PlayerViewSnapshot['episode'],
    segments: [],
    audioUrl: null,
    insights: [],
    topics: [],
    persons: [],
    relatedEpisodes: [],
    stats: null,
  }
}

describe('player-view-cache', () => {
  beforeEach(() => clearPlayerViewCache())

  it('round-trips a snapshot by slug', () => {
    setPlayerViewSnapshot('a', snap('A'))
    expect(getPlayerViewSnapshot('a')?.episode?.title).toBe('A')
    expect(getPlayerViewSnapshot('missing')).toBeUndefined()
  })

  it('evicts the oldest once past the bound (LRU by insertion)', () => {
    // Derived from MAX rather than restating it: #1909 called the old bound of 4 wrong, and a test
    // that hardcodes the number fails for the change rather than for a defect.
    const slugs = overflow(MAX + 1)
    for (const s of slugs) setPlayerViewSnapshot(s, snap(s))
    expect(getPlayerViewSnapshot(slugs[0])).toBeUndefined()
    for (const s of slugs.slice(1)) expect(getPlayerViewSnapshot(s)?.episode?.title).toBe(s)
  })

  it('re-setting a slug refreshes it to most-recently-used, sparing it from eviction', () => {
    const slugs = overflow(MAX)
    for (const s of slugs) setPlayerViewSnapshot(s, snap(s))
    setPlayerViewSnapshot(slugs[0], snap('A2')) // touch → MRU; slugs[1] becomes oldest
    setPlayerViewSnapshot('one-more', snap('one-more')) // evicts slugs[1], not slugs[0]
    expect(getPlayerViewSnapshot(slugs[0])?.episode?.title).toBe('A2')
    expect(getPlayerViewSnapshot(slugs[1])).toBeUndefined()
  })

  /**
   * #1909 named this module: "a persistent generalisation of the in-memory episode LRU… whose
   * storage (RAM, 4 entries) is wrong." In RAM the instant paint existed only within a session —
   * relaunch the app and every episode was a cold spinner again, including one already downloaded.
   */
  describe('persistence (#1909)', () => {
    beforeEach(() => {
      readCached.mockReset().mockResolvedValue(null)
      writeCached.mockReset().mockResolvedValue(undefined)
    })

    it('writes the snapshots to the per-account cache', async () => {
      setPlayerViewSnapshot('a', snap('A'))
      await flushPlayerViewSnapshots()
      expect(writeCached).toHaveBeenCalledWith(PLAYER_SNAPSHOT_KEY, [
        ['a', expect.objectContaining({ episode: { title: 'A' } })],
      ])
    })

    it('persists on its OWN timer, with no flush call', async () => {
      // Every other test here flushes explicitly, which would pass even if nothing ever scheduled
      // the write — i.e. even if the cache never persisted anything in the running app.
      vi.useFakeTimers()
      try {
        setPlayerViewSnapshot('a', snap('A'))
        expect(writeCached, 'wrote synchronously instead of on a debounce').not.toHaveBeenCalled()
        await vi.advanceTimersByTimeAsync(PERSIST_DEBOUNCE_MS)
        expect(writeCached, 'the debounced write never fired').toHaveBeenCalledTimes(1)
      } finally {
        vi.useRealTimers()
      }
    })

    it('coalesces a burst into ONE write', async () => {
      // The caller is a watcher over eight refs that land separately as the page's rails arrive, so
      // an un-coalesced write is a dozen disk writes per episode opened.
      vi.useFakeTimers()
      try {
        for (const s of ['a', 'b', 'c']) setPlayerViewSnapshot(s, snap(s))
        await vi.advanceTimersByTimeAsync(PERSIST_DEBOUNCE_MS)
        expect(writeCached).toHaveBeenCalledTimes(1)
        const payload = writeCached.mock.calls[0][1] as [string, unknown][]
        expect(payload.map((e) => e[0]), 'the coalesced write lost entries').toEqual(['a', 'b', 'c'])
      } finally {
        vi.useRealTimers()
      }
    })

    it('does NOT persist transcripts — the one part big enough to matter', async () => {
      // ~76 KB per episode, below the fold and opt-in. A downloaded episode reads its own from disk.
      setPlayerViewSnapshot('a', {
        ...snap('A'),
        segments: [{ start_ms: 0, text: 'x' } as unknown as PlayerViewSnapshot['segments'][number]],
      })
      await flushPlayerViewSnapshots()
      const payload = writeCached.mock.calls[0][1] as [string, Record<string, unknown>][]
      expect(payload[0][1]).not.toHaveProperty('segments')
    })

    it('persists only the most recent few, not the whole RAM bound', async () => {
      for (const s of overflow(MAX)) setPlayerViewSnapshot(s, snap(s))
      await flushPlayerViewSnapshots()
      const payload = writeCached.mock.calls[0][1] as [string, unknown][]
      expect(payload).toHaveLength(PERSIST_MAX)
      expect(payload[payload.length - 1][0]).toBe(`ep-${MAX - 1}`)
    })

    it('hydrates a stored snapshot so a RELAUNCH paints instantly', async () => {
      readCached.mockResolvedValue([['a', { episode: { title: 'A' }, audioUrl: null }]])
      await hydratePlayerViewCache()
      expect(getPlayerViewSnapshot('a')?.episode?.title).toBe('A')
      // Transcripts were not stored, so a hydrated snapshot reports "not loaded yet" — the state
      // the view would have been in regardless.
      expect(getPlayerViewSnapshot('a')?.segments).toEqual([])
    })

    it('never lets a stored snapshot overwrite one loaded this session', async () => {
      setPlayerViewSnapshot('a', snap('LIVE'))
      readCached.mockResolvedValue([['a', { episode: { title: 'STALE' } }]])
      await hydratePlayerViewCache()
      expect(getPlayerViewSnapshot('a')?.episode?.title).toBe('LIVE')
    })

    it('survives a corrupt or foreign stored value', async () => {
      readCached.mockResolvedValue({ not: 'an array' })
      await expect(hydratePlayerViewCache()).resolves.toBeUndefined()
      readCached.mockResolvedValue([null, ['only-one'], [1, {}]])
      await expect(hydratePlayerViewCache()).resolves.toBeUndefined()
    })

    it('clearing memory does NOT erase the stored copy', async () => {
      // Sign-out clears it through CACHE_KEYS; an account SWITCH should leave the previous
      // account's snapshots intact for when they come back.
      setPlayerViewSnapshot('a', snap('A'))
      clearPlayerViewCache()
      expect(writeCached).not.toHaveBeenCalled()
    })
  })
})
