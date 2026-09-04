import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as deviceStore from './deviceStore'
import { hydrateListenLog, pendingListens, queueListen } from './listenLog'
import { enqueue, hydrateOutbox, pendingWrites } from './outbox'
import { hydratePositions, localPosition, recordPosition } from './playbackPositions'
import { ANON_NAMESPACE, readCached, setCacheNamespace, writeCached } from './contentCache'
import { purgeAnonymousState } from './anonState'

let disk: Record<string, unknown> = {}

beforeEach(() => {
  disk = {}
  vi.spyOn(deviceStore, 'setDeviceJson').mockImplementation(async (k, v) => {
    disk[k] = JSON.parse(JSON.stringify(v))
  })
  vi.spyOn(deviceStore, 'getDeviceJson').mockImplementation(async (k) => (disk[k] ?? null) as never)
  vi.spyOn(deviceStore, 'removeDeviceKey').mockImplementation(async (k) => {
    delete disk[k]
  })
})
afterEach(() => vi.restoreAllMocks())

/**
 * Signing out leaves every device-local store on the `anon` fallback. On a shared device that made
 * sign-out a leak: the previous account's positions and queued writes were still there for
 * whoever picked the phone up next (#1925 review).
 */
describe('purgeAnonymousState', () => {
  it('clears positions, the listen queue, the outbox and the cache', async () => {
    setCacheNamespace(ANON_NAMESPACE)
    await hydratePositions(ANON_NAMESPACE)
    await hydrateListenLog(ANON_NAMESPACE)
    await hydrateOutbox(ANON_NAMESPACE)

    recordPosition('show/ep', 120, false, false)
    queueListen('show/ep', 1000)
    enqueue({ op: 'unfollow', feedId: 'p05' }, 1000)
    await writeCached('queue', ['a'])
    await vi.waitFor(() => expect(localPosition('show/ep')).toBeTruthy())

    await purgeAnonymousState()

    expect(localPosition('show/ep')).toBeNull()
    expect(pendingListens()).toHaveLength(0)
    expect(pendingWrites()).toHaveLength(0)
    setCacheNamespace(ANON_NAMESPACE)
    expect(await readCached('queue')).toBeNull()
    // ...and nothing is left on disk for the next hydrate to resurrect.
    await hydratePositions(ANON_NAMESPACE)
    expect(localPosition('show/ep')).toBeNull()
  })

  it('leaves another account untouched', async () => {
    await hydratePositions('u_a')
    recordPosition('show/ep', 90, false, false)
    await vi.waitFor(() => expect(disk['playback.positions.u_a']).toBeTruthy())

    await purgeAnonymousState()

    expect(disk['playback.positions.u_a']).toBeTruthy()
  })
})
