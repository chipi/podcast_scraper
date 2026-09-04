import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as deviceStore from './deviceStore'
import {
  ANON_NAMESPACE,
  MAX_PENDING,
  __resetListenLog,
  flushListenLog,
  hydrateListenLog,
  pendingKeyFor,
  pendingListens,
  queueListen,
} from './listenLog'

let disk: Record<string, unknown> = {}

beforeEach(() => {
  disk = {}
  __resetListenLog()
  vi.spyOn(deviceStore, 'setDeviceJson').mockImplementation(async (k, v) => {
    disk[k] = JSON.parse(JSON.stringify(v))
  })
  vi.spyOn(deviceStore, 'getDeviceJson').mockImplementation(async (k) => (disk[k] ?? null) as never)
})
afterEach(() => vi.restoreAllMocks())

describe('listenLog (#1924)', () => {
  it('queues an undelivered listen with the moment it happened', async () => {
    await hydrateListenLog(ANON_NAMESPACE)
    queueListen('ep-1', 1000)
    expect(pendingListens()).toEqual([{ slug: 'ep-1', ts: 1000 }])
  })

  it('survives a restart', async () => {
    await hydrateListenLog(ANON_NAMESPACE)
    queueListen('ep-1', 1000)
    await vi.waitFor(() => expect(disk[pendingKeyFor(ANON_NAMESPACE)]).toBeTruthy())
    __resetListenLog()
    await hydrateListenLog(ANON_NAMESPACE)
    expect(pendingListens()).toEqual([{ slug: 'ep-1', ts: 1000 }])
  })

  it('flushes oldest first, carrying the original timestamp', async () => {
    await hydrateListenLog(ANON_NAMESPACE)
    queueListen('later', 2000)
    queueListen('earlier', 1000)
    const push = vi.fn().mockResolvedValue(true)

    await expect(flushListenLog(push)).resolves.toBe(2)
    // The timestamp is the whole point: a week offline must not land as one spike on reconnect.
    expect(push.mock.calls).toEqual([
      ['earlier', 1000],
      ['later', 2000],
    ])
    expect(pendingListens()).toEqual([])
  })

  it('stops at the first failure and keeps the rest', async () => {
    await hydrateListenLog(ANON_NAMESPACE)
    queueListen('a', 1000)
    queueListen('b', 2000)
    const push = vi.fn().mockResolvedValueOnce(true).mockResolvedValue(false)

    await expect(flushListenLog(push)).resolves.toBe(1)
    expect(push).toHaveBeenCalledTimes(2)
    expect(pendingListens().map((p) => p.slug)).toEqual(['b'])
  })

  it('a throwing push is a failure, not a crash', async () => {
    await hydrateListenLog(ANON_NAMESPACE)
    queueListen('a', 1000)
    await expect(flushListenLog(async () => { throw new TypeError('offline') })).resolves.toBe(0)
    expect(pendingListens()).toHaveLength(1)
  })

  it('caps the queue so a long offline stretch cannot grow without bound', async () => {
    await hydrateListenLog(ANON_NAMESPACE)
    for (let i = 0; i < MAX_PENDING + 25; i += 1) queueListen(`ep-${i}`, 1000 + i)
    expect(pendingListens()).toHaveLength(MAX_PENDING)
    // The OLDEST are dropped — recent listening is the more useful signal.
    expect(pendingListens()[0].slug).toBe('ep-25')
  })

  it('keeps accounts apart', async () => {
    await hydrateListenLog('u_alice')
    queueListen('alice-ep', 1000)
    await hydrateListenLog('u_bob')
    expect(pendingListens()).toEqual([])
    await hydrateListenLog('u_alice')
    expect(pendingListens().map((p) => p.slug)).toEqual(['alice-ep'])
  })
})
