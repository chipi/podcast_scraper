/**
 * Writes made offline, replayed on reconnect (#1910).
 *
 * Everything here is ITEM-level and idempotent, so applying it twice lands on the same state and a
 * replay cannot corrupt anything: follow/unfollow a show, add/remove a favourite, add/remove one
 * queued episode, and — since #1925 — create/delete a highlight or a note.
 *
 * Capture was the last holdout, and the reason was real: append-only POSTs, where a retry whose
 * RESPONSE was lost creates a DUPLICATE. The client mints the id now, so the server stores the
 * first write and returns it unchanged for every replay. That is what made capture safe to queue.
 *
 * What is still NOT here:
 *
 * - **Queue REORDERING.** `putQueue` replaces the WHOLE list, so replaying a stale offline write
 *   is last-writer-wins clobbering of edits made meanwhile on another device. Adding and removing
 *   ONE slug has its own endpoints and is here; `move` still needs a live queue, and the arrows
 *   disable while the list is stale.
 * - **EDITS** — a note's text, a highlight's colour. Last-writer-wins on a field, with no
 *   timestamp on the wire to order two devices' edits. Same shape of problem as reordering, and it
 *   wants the same fix first.
 *
 * Namespaced per account, like every other device-local store here: a queued write belongs to the
 * identity that made it and must not be replayed under someone else's session.
 */

import { ApiError } from './api'
import { getDeviceJson, setDeviceJson } from './deviceStore'
import type { FavoriteKind, HighlightCreate, NoteCreate } from './types'

export const OUTBOX_KEY_PREFIX = 'outbox.pending'
export const ANON_NAMESPACE = 'anon'
/** A long offline stretch must not grow the queue without bound. */
export const MAX_PENDING = 200

export type OutboxOp =
  | { op: 'follow'; feedId: string; title?: string | null }
  | { op: 'unfollow'; feedId: string }
  | { op: 'favorite.add'; kind: FavoriteKind; ref: string }
  | { op: 'favorite.remove'; kind: FavoriteKind; ref: string }
  | { op: 'queue.add'; slug: string; after?: string | null }
  | { op: 'queue.remove'; slug: string }
  | { op: 'highlight.create'; body: HighlightCreate }
  | { op: 'highlight.remove'; id: string }
  | { op: 'note.create'; body: NoteCreate }
  | { op: 'note.remove'; id: string }

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
  if (action.op === 'follow' || action.op === 'unfollow') return `show:${action.feedId}`
  if (action.op === 'queue.add' || action.op === 'queue.remove') return `queue:${action.slug}`
  // Keyed by the CLIENT-minted id, which is why capture can be here at all: a create and the
  // delete that undoes it name the same row, so capture-then-undo offline replays as neither
  // rather than as two writes racing each other (#1925).
  if (action.op === 'highlight.create') return `hl:${action.body.client_id}`
  if (action.op === 'highlight.remove') return `hl:${action.id}`
  if (action.op === 'note.create') return `note:${action.body.client_id}`
  if (action.op === 'note.remove') return `note:${action.id}`
  return `fav:${action.kind}:${action.ref}`
}

export function pendingWrites(): readonly OutboxEntry[] {
  return pending
}

/**
 * Replay everything queued, oldest first, stopping at the first failure so a dead network is not
 * hammered. A delivered write is dropped; the rest stay for the next reconnect.
 */
/**
 * A 4xx is the server's verdict; 408/429 and 5xx are worth trying again.
 *
 * Exported because every offline-capable seam needs exactly this distinction, and having each one
 * answer it differently is how a 502 ends up destroying a capture the user made (#1925): a bad
 * gateway is not the server saying no, it is the server not answering.
 */
export function isPermanent(err: unknown): boolean {
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
