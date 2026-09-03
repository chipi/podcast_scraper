import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from './api'
import * as deviceStore from './deviceStore'
import { __resetIdentityEpoch, bumpIdentityEpoch } from './identity'
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
  __resetIdentityEpoch()
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

/**
 * The namespace switch used to wipe `pending` outright. Anything enqueued BEFORE the first hydrate
 * exists only in memory — `persist` refuses until the stored list has been merged in — so an
 * offline unfollow followed quickly by a sign-in simply evaporated (#1925 review).
 */
describe('outbox namespace switch', () => {
  it('parks writes made before hydration under the namespace that made them', async () => {
    // Enqueued while anonymous and un-hydrated: `persist` refuses, so this lives ONLY in memory.
    enqueue({ op: 'unfollow', feedId: 'p05' }, 1000)
    expect(disk[outboxKeyFor(ANON_NAMESPACE)]).toBeUndefined()

    // Signing in switches the namespace. This used to drop the entry on the floor.
    await hydrateOutbox('u_a')
    expect(pendingWrites()).toHaveLength(0)
    await vi.waitFor(() => expect(disk[outboxKeyFor(ANON_NAMESPACE)]).toHaveLength(1))

    // It belongs to the anonymous session and is replayed when that session comes back.
    await hydrateOutbox(ANON_NAMESPACE)
    expect(pendingWrites().map((e) => e.action.op)).toEqual(['unfollow'])
  })

  it('merges parked writes with what that namespace already had on disk', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'follow', feedId: 'p01' }, 1000)
    await vi.waitFor(() => expect(disk[outboxKeyFor(ANON_NAMESPACE)]).toHaveLength(1))

    // A relaunch: the store forgets, a write lands before hydration, then the identity resolves.
    __resetOutbox()
    enqueue({ op: 'follow', feedId: 'p02' }, 2000)
    await hydrateOutbox('u_a')
    await vi.waitFor(() => expect(disk[outboxKeyFor(ANON_NAMESPACE)]).toHaveLength(2))

    await hydrateOutbox(ANON_NAMESPACE)
    expect(pendingWrites().map((e) => e.ts)).toEqual([1000, 2000])
  })

  it('does not replay one account queue under another account session', async () => {
    await hydrateOutbox('u_a')
    enqueue({ op: 'follow', feedId: 'p05' }, 1000)
    await hydrateOutbox('u_b')
    const applied: string[] = []
    await flushOutbox(async (a) => {
      applied.push(a.op)
    })
    expect(applied).toEqual([])
  })
})

/**
 * The queue joined the outbox once it had item-level routes (#1925). Reordering did not: it goes
 * through a whole-list PUT, and "swap these two" replayed against a list someone else has since
 * changed does not mean what the user did.
 */
describe('outbox queue ops', () => {
  it('supersedes an earlier intent for the same slug', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'queue.add', slug: 'ep-1' }, 1000)
    enqueue({ op: 'queue.remove', slug: 'ep-1' }, 2000)
    // Queue-then-unqueue offline replays as ONE removal, not two contradictory writes.
    expect(pendingWrites().map((e) => e.action.op)).toEqual(['queue.remove'])
  })

  it('keeps a follow and a queue add for the same episode apart', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'queue.add', slug: 'p05' }, 1000)
    enqueue({ op: 'follow', feedId: 'p05' }, 2000)
    // Different targets despite the identical id — a slug is not a feed id.
    expect(pendingWrites()).toHaveLength(2)
  })

  it('replays oldest-first so the final queue matches what the user did', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'queue.add', slug: 'a' }, 1000)
    enqueue({ op: 'queue.add', slug: 'b', after: 'a' }, 2000)
    const applied: string[] = []
    await flushOutbox(async (action) => {
      applied.push(action.op === 'queue.add' ? `add:${action.slug}` : action.op)
    })
    expect(applied).toEqual(['add:a', 'add:b'])
    expect(pendingWrites()).toHaveLength(0)
  })
})

/**
 * Capture joined the outbox once the client minted the ids (#1925). Keying on that id is what
 * makes capture-then-undo offline collapse to nothing rather than replaying as two writes.
 */
describe('outbox capture ops', () => {
  const body = { episode_slug: 'show-ep01', kind: 'moment' as const, client_id: 'hc_1' }

  it('a create and the delete that undoes it collapse to one entry', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'highlight.create', body }, 1000)
    enqueue({ op: 'highlight.remove', id: 'hc_1' }, 2000)
    expect(pendingWrites().map((e) => e.action.op)).toEqual(['highlight.remove'])
  })

  it('keeps captures on different ids apart', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'highlight.create', body }, 1000)
    enqueue({ op: 'highlight.create', body: { ...body, client_id: 'hc_2' } }, 2000)
    expect(pendingWrites()).toHaveLength(2)
  })

  it('a note and a highlight with the same id are different targets', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'highlight.remove', id: 'x1' }, 1000)
    enqueue({ op: 'note.remove', id: 'x1' }, 2000)
    expect(pendingWrites()).toHaveLength(2)
  })

  it('survives a restart with its body intact — the id is what makes the replay safe', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'highlight.create', body }, 1000)
    await vi.waitFor(() => expect(disk[outboxKeyFor(ANON_NAMESPACE)]).toHaveLength(1))

    __resetOutbox()
    await hydrateOutbox(ANON_NAMESPACE)
    const [entry] = pendingWrites()
    expect(entry.action.op === 'highlight.create' && entry.action.body.client_id).toBe('hc_1')
  })
})

/**
 * A DEAD SESSION MUST NOT CONSUME THE QUEUE (advisor 1.1).
 *
 * Conflating "the credential died" with "the server refused" was the arc's worst data-loss bug: a
 * week offline, an expired cookie, and every queued write — follows, favourites, queue operations
 * and the user's own highlights and NOTES — was dropped one entry at a time on reconnect, a
 * minute before they signed back in.
 */
describe('a 401 pauses the flush, it never drops a write', () => {
  it('keeps every entry when the session is dead', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'follow', feedId: 'p01' }, 1000)
    enqueue({ op: 'favorite.add', kind: 'episode', ref: 'a' }, 2000)
    enqueue({ op: 'queue.add', slug: 'b' }, 3000)

    const flushed = await flushOutbox(async () => {
      throw new ApiError(401, 'signed out')
    })
    expect(flushed).toBe(0)
    expect(pendingWrites()).toHaveLength(3)
  })

  it('stops at the dead session rather than burning through the rest', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'follow', feedId: 'p01' }, 1000)
    enqueue({ op: 'follow', feedId: 'p02' }, 2000)

    let calls = 0
    await flushOutbox(async () => {
      calls += 1
      throw new ApiError(401, 'signed out')
    })
    // One attempt, then stop — not one per entry, which is how they all got dropped.
    expect(calls).toBe(1)
    expect(pendingWrites()).toHaveLength(2)
  })

  it('403 behaves the same as 401', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'follow', feedId: 'p01' }, 1000)
    await flushOutbox(async () => {
      throw new ApiError(403, 'forbidden')
    })
    expect(pendingWrites()).toHaveLength(1)
  })

  it('still DROPS a genuine refusal, so one dead episode cannot wedge the queue', async () => {
    await hydrateOutbox(ANON_NAMESPACE)
    enqueue({ op: 'favorite.add', kind: 'episode', ref: 'gone' }, 1000)
    enqueue({ op: 'follow', feedId: 'p01' }, 2000)

    const applied: string[] = []
    await flushOutbox(async (action) => {
      if (action.op === 'favorite.add') throw new ApiError(404, 'gone')
      applied.push(action.op)
    })
    expect(applied).toEqual(['follow'])
    expect(pendingWrites()).toHaveLength(0)
  })
})

/**
 * The interleaving the suites never created (advisor 4.3) — which is why this defect class shipped
 * four times before being caught.
 */
describe('an identity switch MID-flush', () => {
  it('abandons the flush rather than delivering A writes under B session', async () => {
    await hydrateOutbox('u_a')
    enqueue({ op: 'follow', feedId: 'p01' }, 1000)
    enqueue({ op: 'follow', feedId: 'p02' }, 2000)

    const applied: string[] = []
    await flushOutbox(async (action) => {
      applied.push(action.op === 'follow' ? action.feedId : action.op)
      // The account changes while the first write is in flight.
      bumpIdentityEpoch()
    })
    // The second entry is NOT sent under the new identity.
    expect(applied).toHaveLength(1)
  })

  it('does not prune the new account queue with the old account results', async () => {
    await hydrateOutbox('u_a')
    enqueue({ op: 'follow', feedId: 'p01' }, 1000)

    await flushOutbox(async () => {
      bumpIdentityEpoch()
      await hydrateOutbox('u_b')
    })
    // u_b's (empty) queue is untouched, and u_a's delivered entry is not filtered out of it.
    expect(pendingWrites()).toHaveLength(0)
    await hydrateOutbox('u_a')
    expect(pendingWrites()).toHaveLength(1)
  })
})

/**
 * Coalescing across namespaces pointed the module at the WRONG account (advisor 1.5).
 */
describe('hydrate coalescing is per namespace', () => {
  it('does not hand a hydrate for one account to a caller asking for another', async () => {
    const a = hydrateOutbox('u_a')
    const b = hydrateOutbox('u_b')
    expect(a).not.toBe(b)
    await Promise.all([a, b])
    enqueue({ op: 'follow', feedId: 'p09' }, 1000)
    await vi.waitFor(() => expect(disk[outboxKeyFor('u_b')]).toHaveLength(1))
    expect(disk[outboxKeyFor('u_a')] ?? []).toHaveLength(0)
  })

  it('still coalesces two calls for the SAME namespace', () => {
    expect(hydrateOutbox('u_a')).toBe(hydrateOutbox('u_a'))
  })
})
