/**
 * Writes made offline, replayed on reconnect (#1910).
 *
 * DELIBERATELY LIMITED to item-level, naturally idempotent operations: follow/unfollow a show, and
 * add/remove a favourite. Applying any of those twice lands on the same state, so a replay cannot
 * corrupt anything.
 *
 * What is NOT here, and why:
 *
 * - **The queue.** `putQueue` replaces the WHOLE list, so replaying a stale offline write is
 *   last-writer-wins clobbering of edits made meanwhile on another device — a conflict-resolution
 *   problem, not a retry problem. It needs a version primitive on the endpoint first. The queue
 *   store already refuses to write from a stale baseline rather than guess.
 * - **Highlights and notes.** Append-only POSTs, so a retry whose response was lost creates a
 *   DUPLICATE. They need a client-supplied idempotency key server-side before they can be replayed
 *   safely; adding them here without one would trade a lost write for a duplicated one.
 *
 * Namespaced per account, like every other device-local store here: a queued write belongs to the
 * identity that made it and must not be replayed under someone else's session.
 */

import { ApiError } from './api'
import { getDeviceJson, setDeviceJson } from './deviceStore'
import type { FavoriteKind } from './types'

export const OUTBOX_KEY_PREFIX = 'outbox.pending'
export const ANON_NAMESPACE = 'anon'
/** A long offline stretch must not grow the queue without bound. */
export const MAX_PENDING = 200

export type OutboxOp =
  | { op: 'follow'; feedId: string; title?: string | null }
  | { op: 'unfollow'; feedId: string }
  | { op: 'favorite.add'; kind: FavoriteKind; ref: string }
  | { op: 'favorite.remove'; kind: FavoriteKind; ref: string }

export interface OutboxEntry {
  id: string
  ts: number
  action: OutboxOp
}

export function outboxKeyFor(namespace: string): string {
  return `${OUTBOX_KEY_PREFIX}.${namespace}`
}

let pending: OutboxEntry[] = []
let hydrated = false
let namespace = ANON_NAMESPACE
/**
 * Coalesces concurrent hydrates. Without this two overlapping calls both pass the `hydrated`
 * check, both read disk, and the second concatenates what the first already merged — duplicating
 * every stored entry. The shell creates exactly that overlap (the identity watcher vs the awaited
 * call at boot). Same guard the downloads store has.
 */
let inflightHydrate: Promise<void> | null = null
/** One flush at a time: boot and networkStatusChange both fire, and two runs deliver everything twice. */
let flushing = false
let seq = 0

export function hydrateOutbox(ns: string = ANON_NAMESPACE): Promise<void> {
  if (inflightHydrate) return inflightHydrate
  const run = hydrateInner(ns).finally(() => {
    if (inflightHydrate === run) inflightHydrate = null
  })
  inflightHydrate = run
  return run
}

async function hydrateInner(ns: string): Promise<void> {
  const next = ns || ANON_NAMESPACE
  if (next !== namespace) {
    // Entries queued BEFORE the first hydrate exist only in memory — `persist` refuses until the
    // stored list has been merged in, so it never wrote them. Dropping them here lost writes the
    // user had already made (#1925 review): sign in fast enough after an offline unfollow and the
    // unfollow simply evaporated. Park them under the namespace that made them, then switch.
    if (!hydrated && pending.length) await parkUnhydrated(namespace, pending)
    pending = []
    hydrated = false
    namespace = next
  }
  if (hydrated) return
  const stored = (await getDeviceJson<OutboxEntry[]>(outboxKeyFor(namespace))) ?? []
  const inMemory = pending
  pending = [...stored, ...inMemory]
  hydrated = true
  if (inMemory.length) persist()
}

/**
 * Merge orphaned in-memory entries into the stored list of the namespace they belong to. Ordered
 * oldest-first afterwards so the next flush replays them in the order they were made.
 */
async function parkUnhydrated(ns: string, orphans: OutboxEntry[]): Promise<void> {
  const key = outboxKeyFor(ns)
  const stored = (await getDeviceJson<OutboxEntry[]>(key)) ?? []
  const byTarget = new Map<string, OutboxEntry>()
  for (const e of [...stored, ...orphans]) byTarget.set(targetOf(e.action), e)
  const merged = [...byTarget.values()].sort((a, b) => a.ts - b.ts).slice(-MAX_PENDING)
  await setDeviceJson(key, merged).catch(() => {})
}

function persist(): void {
  // Refuses before hydration for the same reason the registry does: `pending` is not yet the union
  // of what is on disk, so writing it would drop queued writes made in a previous session.
  if (!hydrated) return
  void setDeviceJson(outboxKeyFor(namespace), pending).catch(() => {})
}

/**
 * Queue a write that did not land. A newer action on the SAME target supersedes the older one —
 * follow-then-unfollow offline should replay as one unfollow, not two contradictory writes.
 */
export function enqueue(action: OutboxOp, ts: number = Date.now()): void {
  const target = targetOf(action)
  pending = pending.filter((e) => targetOf(e.action) !== target)
  seq += 1
  pending.push({ id: `${ts}-${seq}`, ts, action })
  if (pending.length > MAX_PENDING) pending = pending.slice(-MAX_PENDING)
  persist()
}

function targetOf(action: OutboxOp): string {
  return action.op === 'follow' || action.op === 'unfollow'
    ? `show:${action.feedId}`
    : `fav:${action.kind}:${action.ref}`
}

export function pendingWrites(): readonly OutboxEntry[] {
  return pending
}

/**
 * Replay everything queued, oldest first, stopping at the first failure so a dead network is not
 * hammered. A delivered write is dropped; the rest stay for the next reconnect.
 */
/** A 4xx is the server's verdict; 408/429 and 5xx are worth trying again. */
function isPermanent(err: unknown): boolean {
  if (!(err instanceof ApiError)) return false
  if (err.status === 408 || err.status === 429) return false
  return err.status >= 400 && err.status < 500
}

export async function flushOutbox(
  apply: (action: OutboxOp) => Promise<void>,
): Promise<number> {
  if (flushing) return 0
  flushing = true
  try {
  if (!pending.length) return 0
  const ordered = [...pending].sort((a, b) => a.ts - b.ts)
  const done = new Set<string>()
  let flushed = 0
  for (const entry of ordered) {
    try {
      await apply(entry.action)
      done.add(entry.id)
      flushed += 1
    } catch (err: unknown) {
      if (isPermanent(err)) {
        // The server ANSWERED and refused — a removed episode (404), an expired session (401).
        // Retrying cannot help, and keeping it would wedge every entry behind it FOREVER, on
        // every reconnect. Drop it and carry on.
        done.add(entry.id)
        continue
      }
      // The request never landed. Stop rather than hammer a dead network; the rest keep.
      break
    }
  }
  if (done.size) {
    pending = pending.filter((e) => !done.has(e.id))
    persist()
  }
    return flushed
  } finally {
    flushing = false
  }
}

/** Test seam. */
export function __resetOutbox(): void {
  pending = []
  hydrated = false
  namespace = ANON_NAMESPACE
  seq = 0
}
