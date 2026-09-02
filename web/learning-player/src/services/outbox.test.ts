import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from './api'
import * as deviceStore from './deviceStore'
import {
  ANON_NAMESPACE,
  MAX_PENDING,
  __resetOutbox,
  enqueue,
  flushOutbox,
  hydrateOutbox,
  outboxKeyFor,
  pendingWrites,
} from './outbox'

let disk: Record<string, unknown> = {}

beforeEach(() => {
  disk = {}
  __resetOutbox()
  vi.spyOn(deviceStore, 'setDeviceJson').mockImplementation(async (k, v) => {
    disk[k] = JSON.parse(JSON.stringify(v))
  })
  vi.spyOn(deviceStore, 'getDeviceJson').mockImplementation(async (k) => (disk[k] ?? null) as never)
})
afterEach(() => vi.restoreAllMocks())

describe('outbox (#1910)', () => {
  it('queues a write and survives a restart', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'follow', feedId: 'p05', title: 'Show' }, 1000)
    await vi.waitFor(() => expect(disk[outboxKeyFor(ANON_NAMESPACE)]).toBeTruthy())

    __resetOutbox()
    await hydrateOutbox(ANON_NAMESPACE)
    expect(pendingWrites()).toHaveLength(1)
  })

  it('a newer action on the same target supersedes the older one', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'follow', feedId: 'p05' }, 1000)
    enqueue({ op: 'unfollow', feedId: 'p05' }, 2000)
    // Follow-then-unfollow offline must replay as ONE unfollow, not two contradictory writes.
    expect(pendingWrites()).toHaveLength(1)
    expect(pendingWrites()[0].action).toMatchObject({ op: 'unfollow', feedId: 'p05' })
  })

  it('keeps actions on different targets apart', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'follow', feedId: 'p05' }, 1000)
    enqueue({ op: 'follow', feedId: 'p06' }, 1001)
    enqueue({ op: 'favorite.add', kind: 'episode', ref: 'ep-1' }, 1002)
    expect(pendingWrites()).toHaveLength(3)
  })

  it('replays oldest first and clears what landed', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'follow', feedId: 'later' }, 2000)
    enqueue({ op: 'follow', feedId: 'earlier' }, 1000)
    const applied: string[] = []

    await expect(
      flushOutbox(async (a) => {
        applied.push(a.op === 'follow' ? a.feedId : 'other')
      }),
    ).resolves.toBe(2)
    expect(applied).toEqual(['earlier', 'later'])
    expect(pendingWrites()).toEqual([])
  })

  it('stops at the first failure and keeps the rest for next time', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'follow', feedId: 'a' }, 1000)
    enqueue({ op: 'follow', feedId: 'b' }, 2000)
    let calls = 0

    await expect(
      flushOutbox(async () => {
        calls += 1
        if (calls > 1) throw new TypeError('Failed to fetch')
      }),
    ).resolves.toBe(1)
    expect(pendingWrites()).toHaveLength(1)
  })

  it('caps the queue so a long offline stretch cannot grow without bound', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    for (let i = 0; i < MAX_PENDING + 10; i += 1) enqueue({ op: 'follow', feedId: `p${i}` }, 1000 + i)
    expect(pendingWrites()).toHaveLength(MAX_PENDING)
  })

  it('keeps accounts apart — a queued write belongs to who made it', async () => {
    await hydrateOutbox('u_alice')
    enqueue({ op: 'follow', feedId: 'alice-show' }, 1000)
    await hydrateOutbox('u_bob')
    expect(pendingWrites()).toEqual([])
    await hydrateOutbox('u_alice')
    expect(pendingWrites()).toHaveLength(1)
  })

  it('drops a permanently refused write instead of wedging everything behind it', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'favorite.add', kind: 'episode', ref: 'gone' }, 1000)
    enqueue({ op: 'follow', feedId: 'fine' }, 2000)
    const applied: string[] = []

    // A removed episode 404s on EVERY reconnect. Treating that as transient parked every entry
    // behind it forever, invisibly.
    await flushOutbox(async (a) => {
      if (a.op === 'favorite.add') throw new ApiError(404, 'gone')
      applied.push('fine')
    })

    expect(applied).toEqual(['fine'])
    expect(pendingWrites()).toEqual([])
  })

  it('keeps a write the server could not answer for', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'follow', feedId: 'a' }, 1000)
    // 5xx and 429 are worth another attempt; a 4xx verdict is not.
    await flushOutbox(async () => { throw new ApiError(503, 'unavailable') })
    expect(pendingWrites()).toHaveLength(1)
    await flushOutbox(async () => { throw new ApiError(429, 'slow down') })
    expect(pendingWrites()).toHaveLength(1)
  })
})
