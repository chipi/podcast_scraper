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
import { identityChangedSince, identityEpoch } from './identity'
import type { FavoriteKind, HighlightCreate, NoteCreate, CollectionItemRef } from './types'

export const OUTBOX_KEY_PREFIX = 'outbox.pending'
export const ANON_NAMESPACE = 'anon'
/** A long offline stretch must not grow the queue without bound. */
export const MAX_PENDING = 200

export type OutboxOp =
  | { op: 'follow'; feedId: string; title?: string | null }
  | { op: 'unfollow'; feedId: string }
  | { op: 'favorite.add'; kind: FavoriteKind; ref: string }
  | { op: 'favorite.remove'; kind: FavoriteKind; ref: string }
  // A colour EDIT on a saved item (RFC-121 ph. 4). Keyed apart from add/remove (see `targetOf`) so
  // it never evicts a still-pending add of the same item; replayed as PATCH, last-writer-wins on the
  // field, and a 404 (the favorite is gone) drops harmlessly like any permanent refusal.
  | { op: 'favorite.color'; kind: FavoriteKind; ref: string; color: string | null }
  | { op: 'queue.add'; slug: string; after?: string | null }
  | { op: 'queue.remove'; slug: string }
  | { op: 'completed.add'; slug: string }
  | { op: 'completed.remove'; slug: string }
  // Follow / unfollow an interest token (topic-cluster / topic / person). Item-level and idempotent
  // (add/remove have their own endpoints, server dedups), so a replay lands on the same state — the
  // same shape as favourites, which is why interests can queue at all (#2004 #7).
  | { op: 'interest.add'; token: string }
  | { op: 'interest.remove'; token: string }
  | { op: 'highlight.create'; body: HighlightCreate }
  | { op: 'highlight.edit'; id: string; color: string | null }
  | { op: 'highlight.remove'; id: string }
  | { op: 'note.create'; body: NoteCreate }
  | { op: 'note.edit'; id: string; text: string }
  | { op: 'note.remove'; id: string }
  /**
   * Collections replay like every other per-user write (#2004 item 13).
   *
   * They were the ONLY one without an outbox op — favourites, queue, highlights, notes and follows
   * all queue and replay, so a dropped request is invisible in those and permanent in collections.
   * That asymmetry, not a collections-specific bug, is why only collections lost data.
   *
   * `client_id` is carried so a replayed create is idempotent and the item that followed it can
   * still find its collection.
   */
  | { op: 'collection.create'; name: string; clientId: string }
  | { op: 'collection.addItem'; collectionId: string; item: CollectionItemRef }

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
/** Which namespace `inflightHydrate` is for — coalescing across namespaces is a data leak. */
let inflightNs: string | null = null
/** One flush at a time: boot and networkStatusChange both fire, and two runs deliver everything twice. */
let flushing = false
let seq = 0

export function hydrateOutbox(ns: string = ANON_NAMESPACE): Promise<void> {
  // Coalesce only for the SAME namespace. Returning an in-flight hydrate for a DIFFERENT one
  // left the module pointed at the old namespace, so the new account's offline writes persisted
  // under the previous key — later deleted by purgeAnonymousState, or replayed under whoever was
  // signed in (advisor 1.5). The shell produces exactly that overlap: the identity watcher's
  // fire-and-forget adopt racing the awaited boot adopt.
  const target = ns || ANON_NAMESPACE
  if (inflightHydrate && inflightNs === target) return inflightHydrate
  const run = hydrateInner(ns).finally(() => {
    if (inflightHydrate === run) {
      inflightHydrate = null
      inflightNs = null
    }
  })
  inflightHydrate = run
  inflightNs = target
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
  const stored = (await getDeviceJson<OutboxEntry[]>(outboxKeyFor(next))) ?? []
  // The namespace can have moved while we were reading. Coalescing only dedupes calls for the
  // SAME namespace, so two different-namespace hydrates genuinely run concurrently — and without
  // this the loser merges A's stored entries into memory under B, then `persist()` writes A's
  // outbox onto B's disk key. The positions module got this check; these two did not, so the
  // defect class the first review named survived in two of its three homes.
  if (namespace !== next) return
  const inMemory = pending
  pending = [...stored, ...inMemory]
  hydrated = true
  if (inMemory.length && namespace === next) persist()
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
  if (action.op === 'completed.add' || action.op === 'completed.remove')
    return `completed:${action.slug}`
  if (action.op === 'interest.add' || action.op === 'interest.remove')
    return `interest:${action.token}`
  // Keyed by the CLIENT-minted id, which is why capture can be here at all: a create and the
  // delete that undoes it name the same row, so capture-then-undo offline replays as neither
  // rather than as two writes racing each other (#1925).
  if (action.op === 'highlight.create') return `hl:${action.body.client_id}`
  // edit + remove share the highlight's key with its create: the latest write for a given highlight
  // wins the slot (edit-then-edit coalesces to the last colour; remove-after-edit replaces it).
  if (action.op === 'highlight.remove' || action.op === 'highlight.edit') return `hl:${action.id}`
  if (action.op === 'collection.create') return `col:${action.clientId}`
  if (action.op === 'collection.addItem')
    return `colitem:${action.collectionId}:${action.item.kind}:${action.item.ref}`
  if (action.op === 'note.create') return `note:${action.body.client_id}`
  // edit + remove share the note's key: the latest of them for a given note wins the queue slot
  // (edit-then-edit coalesces to the last text; remove-after-edit replaces the edit).
  if (action.op === 'note.remove' || action.op === 'note.edit') return `note:${action.id}`
  // A colour edit gets its OWN slot, distinct from `fav:` — else queuing it would evict a still
  // -pending add of the same item and lose the save. Oldest-first replay still runs the add first.
  if (action.op === 'favorite.color') return `favcolor:${action.kind}:${action.ref}`
  return `fav:${action.kind}:${action.ref}`
}

/**
 * Withdraw a queued CREATE for `clientId`, returning whether one was there.
 *
 * Capture is the only pair where the create and its undo can both be offline: the row exists on
 * screen under a client-minted id and has never reached the server. Deleting it there 404s, and a
 * 404 is a refusal — which would restore the row and leave it undeletable until the outbox
 * happened to create it (advisor 3.1). Withdrawing the create instead makes create-then-delete
 * offline collapse to nothing, which is exactly what the user did.
 */
export function withdrawPendingCreate(clientId: string): boolean {
  // NEVER while a flush is in flight. `flushOutbox` iterates a SNAPSHOT and prunes afterwards, so
  // withdrawing mid-flush is unsafe in both directions (advisor-2 #2): withdraw after the create
  // was applied but before the prune, and the caller skips the server delete while the row exists;
  // withdraw before it is applied, and the snapshot creates it anyway. Reporting `false` sends the
  // caller down the ordinary delete path, which is always correct — the flush's own 404-drop
  // absorbs the case where the row never existed.
  if (flushing) return false
  const before = pending.length
  pending = pending.filter(
    (e) =>
      !(
        (e.action.op === 'highlight.create' && e.action.body.client_id === clientId) ||
        (e.action.op === 'note.create' && e.action.body.client_id === clientId) ||
        // The server cascades notes when a highlight is deleted; a WITHDRAWN create never reaches
        // the server, so its queued notes would replay against a highlight that never existed and
        // land as orphans (`POST /notes` does not validate its target). Withdraw them together
        // (advisor-2 #6).
        (e.action.op === 'note.create' && e.action.body.target_id === clientId)
      ),
  )
  if (pending.length !== before) {
    persist()
    return true
  }
  return false
}

/**
 * Fold an edit into a still-queued CREATE for `id`, returning whether one was there.
 *
 * A note created offline lives under a client-minted id and has never reached the server. Editing
 * it must NOT queue a `note.edit` — that replays as `PATCH /notes/<client_id>` on a note the server
 * has never seen (404). Instead, rewrite the queued create's text so it lands, once, with the final
 * text. Same flush-safety rule as `withdrawPendingCreate`: never while a flush is iterating its
 * snapshot; report `false` and let the caller queue an ordinary edit (harmless once the create has
 * been delivered, since then the note DOES exist server-side).
 */
export function updatePendingNoteText(id: string, text: string): boolean {
  if (flushing) return false
  const entry = pending.find(
    (e) => e.action.op === 'note.create' && e.action.body.client_id === id,
  )
  if (!entry || entry.action.op !== 'note.create') return false
  entry.action.body = { ...entry.action.body, text }
  persist()
  return true
}

/**
 * Is a CREATE for this note still queued (incl. mid-flush)? Read-only, flush-safe.
 *
 * The queue keys `note.create`, `note.edit` and `note.remove` for one note under the SAME slot, so
 * enqueuing a `note.edit` would EVICT a still-queued create — and if that create then fails to
 * deliver, the note is lost (the edit replays as a PATCH/404 on a note the server never got). Callers
 * use this to refuse queuing a `note.edit` while a create is pending; they fold the text in via
 * `updatePendingNoteText` instead (a no-op mid-flush, where the create simply replays as-is).
 */
export function hasPendingNoteCreate(id: string): boolean {
  return pending.some((e) => e.action.op === 'note.create' && e.action.body.client_id === id)
}

/**
 * Fold a colour edit into a still-queued highlight CREATE for `id` — the highlight analogue of
 * `updatePendingNoteText`. A highlight created offline lives under a client-minted id the server has
 * never seen, so a `highlight.edit` would replay as `PATCH /highlights/<client_id>` (404). Rewrite
 * the queued create's colour instead so it lands once, with the final colour. No-op mid-flush.
 */
export function updatePendingHighlightColor(id: string, color: string | null): boolean {
  if (flushing) return false
  const entry = pending.find(
    (e) => e.action.op === 'highlight.create' && e.action.body.client_id === id,
  )
  if (!entry || entry.action.op !== 'highlight.create') return false
  entry.action.body = { ...entry.action.body, color }
  persist()
  return true
}

/** Is a CREATE for this highlight still queued (incl. mid-flush)? Read-only, flush-safe. Callers
 *  use it to refuse an evicting `highlight.edit` while a create pends — same rule as notes. */
export function hasPendingHighlightCreate(id: string): boolean {
  return pending.some((e) => e.action.op === 'highlight.create' && e.action.body.client_id === id)
}

export function pendingWrites(): readonly OutboxEntry[] {
  return pending
}

/**
 * Replay everything queued, oldest first, stopping at the first failure so a dead network is not
 * hammered. A delivered write is dropped; the rest stay for the next reconnect.
 */
/**
 * Did the SESSION die? 401/403 is not a verdict on the write — it is a verdict on the credential,
 * and it is the one failure that re-authenticating repairs.
 *
 * This is separated from `isPermanent` because conflating them destroyed data: a week of offline
 * use, an expired cookie, and every queued write — follows, favourites, queue operations, and the
 * user's own highlights and NOTES — was dropped one entry at a time on reconnect, silently, a
 * minute before they signed back in (advisor 1.1).
 */
export function isAuthFailure(err: unknown): boolean {
  return err instanceof ApiError && (err.status === 401 || err.status === 403)
}

/**
 * A 4xx is the server's verdict; 408/429 and 5xx are worth trying again.
 *
 * Exported because every offline-capable seam needs exactly this distinction, and having each one
 * answer it differently is how a 502 ends up destroying a capture the user made (#1925): a bad
 * gateway is not the server saying no, it is the server not answering.
 *
 * 401/403 is deliberately NOT permanent here — see `isAuthFailure`. A dead session must pause a
 * flush, never consume it.
 */
export function isPermanent(err: unknown): boolean {
  if (!(err instanceof ApiError)) return false
  if (err.status === 408 || err.status === 429) return false
  if (err.status === 401 || err.status === 403) return false
  return err.status >= 400 && err.status < 500
}

export async function flushOutbox(
  apply: (action: OutboxOp) => Promise<void>,
): Promise<number> {
  if (flushing) return 0
  flushing = true
  try {
  if (!pending.length) return 0
  // The identity this flush belongs to. A switch mid-loop means the remaining entries would be
  // delivered under the NEW account's session, and the bookkeeping below would then filter and
  // persist the new namespace's array — so the delivered entries stay on the old account's key
  // and are replayed again at its next sign-in (advisor 1.4).
  const generation = identityEpoch()
  const startedIn = namespace
  const ordered = [...pending].sort((a, b) => a.ts - b.ts)
  const done = new Set<string>()
  let flushed = 0
  for (const entry of ordered) {
    if (identityChangedSince(generation) || namespace !== startedIn) break
    try {
      await apply(entry.action)
      done.add(entry.id)
      flushed += 1
    } catch (err: unknown) {
      // A dead session stops the flush and keeps EVERYTHING. Re-authenticating repairs it; the
      // old code dropped the queue entry by entry instead (advisor 1.1).
      if (isAuthFailure(err)) break
      if (isPermanent(err)) {
        // The server ANSWERED and refused — a removed episode (404). Retrying cannot help, and
        // keeping it would wedge every entry behind it FOREVER, on every reconnect. Drop it.
        done.add(entry.id)
        continue
      }
      // The request never landed. Stop rather than hammer a dead network; the rest keep.
      break
    }
  }
  // Only reconcile when this is still the account we started as; otherwise `pending` is now
  // somebody else's array and filtering it would delete their queued writes.
  if (done.size && !identityChangedSince(generation) && namespace === startedIn) {
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
