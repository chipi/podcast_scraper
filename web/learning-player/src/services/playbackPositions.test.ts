import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as deviceStore from './deviceStore'
import {
  ANON_NAMESPACE,
  CLOCK_SKEW_MARGIN_MS,
  positionsKeyFor,
  __resetPositions,
  flushPendingPositions,
  hydratePositions,
  localPosition,
  pendingPositions,
  recordPosition,
  shouldPush,
} from './playbackPositions'

let disk: Record<string, unknown> = {}

beforeEach(() => {
  disk = {}
  __resetPositions()
  vi.spyOn(deviceStore, 'setDeviceJson').mockImplementation(async (k, v) => {
    disk[k] = JSON.parse(JSON.stringify(v))
  })
  vi.spyOn(deviceStore, 'getDeviceJson').mockImplementation(async (k) => (disk[k] ?? null) as never)
})
afterEach(() => vi.restoreAllMocks())

describe('playback positions', () => {
  it('records and reads a position synchronously', () => {
    recordPosition('a', 42, false, true, 1000)
    expect(localPosition('a')).toEqual({ seconds: 42, finished: false, updatedAt: 1000 })
  })

  it('returns null for an episode never played', () => {
    expect(localPosition('nope')).toBeNull()
  })

  it('persists across a restart', async () => {
    // Hydrate first, as the app does at boot — persist() deliberately refuses to write before
    // that, so it cannot clobber the stored map with a partial one.
    await hydratePositions(ANON_NAMESPACE)
    recordPosition('a', 42, false, true, 1000)
    await vi.waitFor(() => expect(disk[positionsKeyFor(ANON_NAMESPACE)]).toBeTruthy())
    __resetPositions()
    await hydratePositions(ANON_NAMESPACE)
    expect(localPosition('a')?.seconds).toBe(42)
  })

  it('marks an unsynced write pending, and a synced one not', () => {
    recordPosition('off', 10, false, false, 1)
    recordPosition('on', 20, false, true, 2)
    expect(pendingPositions().map((p) => p.slug)).toEqual(['off'])
  })

  it('flushes pending positions oldest-first and clears the flag', async () => {
    recordPosition('b', 20, false, false, 200)
    recordPosition('a', 10, false, false, 100)
    const push = vi.fn().mockResolvedValue(undefined)

    await expect(flushPendingPositions(push)).resolves.toBe(2)
    expect(push.mock.calls.map((c) => c[0])).toEqual(['a', 'b'])
    expect(pendingPositions()).toEqual([])
  })

  it('keeps positions pending when still offline', async () => {
    recordPosition('a', 10, false, false, 100)
    const push = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'))
    await expect(flushPendingPositions(push)).resolves.toBe(0)
    expect(pendingPositions().map((p) => p.slug)).toEqual(['a'])
  })

  it('does not clear a position that moved again during the flush', async () => {
    recordPosition('a', 10, false, false, 100)
    const push = vi.fn().mockImplementation(async () => {
      // The user kept listening while the flush was in flight.
      recordPosition('a', 99, false, false, 300)
    })
    await expect(flushPendingPositions(push)).resolves.toBe(0)
    expect(localPosition('a')?.seconds).toBe(99)
    expect(pendingPositions().map((p) => p.slug)).toEqual(['a'])
  })

  it('stops at the first failure rather than hammering a dead network', async () => {
    recordPosition('a', 1, false, false, 100)
    recordPosition('b', 2, false, false, 200)
    const push = vi.fn().mockRejectedValue(new Error('down'))
    await flushPendingPositions(push)
    expect(push).toHaveBeenCalledTimes(1)
  })

  // #1906 — do not overwrite newer progress from another device.

  it('shouldPush: pushes when the server holds nothing', () => {
    expect(shouldPush({ seconds: 10, finished: false, updatedAt: 1000 }, null)).toBe(true)
  })

  it('shouldPush: refuses when the server record landed after we went offline', () => {
    // Its arrival stamp is in SECONDS, ours in ms.
    const local = { seconds: 500, finished: false, updatedAt: 1_000_000 }
    // 2_000s = 2_000_000ms, a clear margin beyond our 1_000_000ms write.
    const server = { seconds: 20, finished: false, updatedAt: 2_000 }
    // Server wrote at 2,000,000ms — after our 1,000,000ms write — so it knows more, even though
    // our position is further along.
    expect(shouldPush(local, server)).toBe(false)
  })

  it('shouldPush: pushes when our write is the newer one', () => {
    const local = { seconds: 500, finished: false, updatedAt: 3_000_000 }
    // 2_000s = 2_000_000ms, a clear margin behind our 3_000_000ms write.
    const server = { seconds: 900, finished: false, updatedAt: 2_000 }
    expect(shouldPush(local, server)).toBe(true)
  })

  it('shouldPush: with no server timestamp, only moves progress forward', () => {
    const base = { finished: false, updatedAt: 1000 }
    expect(shouldPush({ ...base, seconds: 90 }, { seconds: 50, finished: false, updatedAt: null })).toBe(true)
    expect(shouldPush({ ...base, seconds: 10 }, { seconds: 50, finished: false, updatedAt: null })).toBe(false)
    // Finishing is worth reporting even from behind.
    expect(shouldPush({ ...base, seconds: 10, finished: true }, { seconds: 50, finished: false, updatedAt: null })).toBe(true)
  })

  it('flush skips a clobbering write but stops it being pending forever', async () => {
    recordPosition('a', 10, false, false, 1_000_000)
    const push = vi.fn().mockResolvedValue(undefined)
    const read = vi.fn().mockResolvedValue({ seconds: 900, finished: false, updatedAt: 2_000 })

    await expect(flushPendingPositions(push, read)).resolves.toBe(1)
    expect(push).not.toHaveBeenCalled()
    // Retrying a write the server has already beaten, on every reconnect forever, is noise.
    expect(pendingPositions()).toEqual([])
  })

  it('flush still pushes when ours is newer', async () => {
    recordPosition('a', 500, false, false, 3_000_000)
    const push = vi.fn().mockResolvedValue(undefined)
    const read = vi.fn().mockResolvedValue({ seconds: 20, finished: false, updatedAt: 2_000 })
    await flushPendingPositions(push, read)
    expect(push).toHaveBeenCalledWith('a', 500, false)
  })

  // #1906 follow-ups from review.

  it('within the clock-skew margin, falls back to forward-only', () => {
    // The two stamps come from different clocks, so a near-tie tells us nothing reliable.
    const base = 10_000_000
    const server = { seconds: 900, finished: false, updatedAt: base / 1000 }
    // Ours looks newer, but only by less than the margin — do not trust it over a bigger position.
    expect(
      shouldPush({ seconds: 10, finished: false, updatedAt: base + CLOCK_SKEW_MARGIN_MS / 2 }, server),
    ).toBe(false)
    // Same near-tie, but ours is genuinely further along — forward-only allows it.
    expect(
      shouldPush({ seconds: 950, finished: false, updatedAt: base + CLOCK_SKEW_MARGIN_MS / 2 }, server),
    ).toBe(true)
  })

  it('a write before the first hydrate does not clobber the stored positions', async () => {
    // Regression: persist ran unconditionally, so a position recorded before hydratePositions
    // resolved overwrote the whole stored map — the same defect fixed in the downloads registry
    // and then reintroduced here.
    disk[positionsKeyFor(ANON_NAMESPACE)] = { old: { seconds: 5, finished: false, updatedAt: 1 } }

    recordPosition('fresh', 42, false, true, 1000)
    // Nothing may have been written yet.
    expect(Object.keys(disk[positionsKeyFor(ANON_NAMESPACE)] as object)).toEqual(['old'])

    await hydratePositions(ANON_NAMESPACE)
    expect(localPosition('old')?.seconds).toBe(5)
    expect(localPosition('fresh')?.seconds).toBe(42)
    await vi.waitFor(() =>
      expect(Object.keys(disk[positionsKeyFor(ANON_NAMESPACE)] as object).sort()).toEqual([
        'fresh',
        'old',
      ]),
    )
  })

  it('positions are per account, so a switch cannot leak or flush across users', async () => {
    disk[positionsKeyFor('u_alice')] = { ep: { seconds: 300, finished: false, updatedAt: 1 } }
    await hydratePositions('u_alice')
    expect(localPosition('ep')?.seconds).toBe(300)

    await hydratePositions('u_bob')
    // Bob must not resume at Alice's position, and must not flush it under his session.
    expect(localPosition('ep')).toBeNull()
    expect(pendingPositions()).toEqual([])

    await hydratePositions('u_alice')
    expect(localPosition('ep')?.seconds).toBe(300)
  })

  it('writes land in the account that recorded them', async () => {
    await hydratePositions('u_alice')
    recordPosition('ep', 10, false, true, 1000)
    await hydratePositions('u_bob')
    recordPosition('ep', 20, false, true, 2000)

    await vi.waitFor(() => expect(disk[positionsKeyFor('u_bob')]).toBeTruthy())
    expect((disk[positionsKeyFor('u_alice')] as Record<string, { seconds: number }>).ep.seconds).toBe(10)
    expect((disk[positionsKeyFor('u_bob')] as Record<string, { seconds: number }>).ep.seconds).toBe(20)
  })
})
