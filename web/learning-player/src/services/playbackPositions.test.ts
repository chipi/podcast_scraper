import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as deviceStore from './deviceStore'
import {
  POSITIONS_KEY,
  __resetPositions,
  flushPendingPositions,
  hydratePositions,
  localPosition,
  pendingPositions,
  recordPosition,
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
    recordPosition('a', 42, false, true, 1000)
    await vi.waitFor(() => expect(disk[POSITIONS_KEY]).toBeTruthy())
    __resetPositions()
    await hydratePositions()
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
})
