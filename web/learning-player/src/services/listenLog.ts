/**
 * Listen events that could not be delivered (#1924).
 *
 * `logListen` is best-effort, and offline it simply failed — so a plane trip left no trace at all.
 * Events are now queued on the device WITH the moment they happened and flushed on reconnect; the
 * server accepts that timestamp (clamped) so a week of offline listening lands on the days it
 * happened rather than as one spike on the day the device came back.
 *
 * Namespaced per account for the same reason the downloads registry and positions are: listening
 * history must not cross accounts on a shared device.
 */

import { getDeviceJson, setDeviceJson } from './deviceStore'
import { identityChangedSince, identityEpoch } from './identity'

export const PENDING_KEY_PREFIX = 'listen.pending'
export const ANON_NAMESPACE = 'anon'

export interface PendingListen {
  slug: string
  /** Unix SECONDS — the wire format the server clamps. */
  ts: number
}

export function pendingKeyFor(namespace: string): string {
  return `${PENDING_KEY_PREFIX}.${namespace}`
}

let pending: PendingListen[] = []
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

/** Load this account's queue. Call at boot and whenever the signed-in user changes. */
export function hydrateListenLog(ns: string = ANON_NAMESPACE): Promise<void> {
  // Coalesce only for the SAME namespace. Returning an in-flight hydrate for a DIFFERENT one
  // left the module pointed at the old namespace, so the new account's offline writes persisted
  // under the previous key — later deleted by purgeAnonymousState, or replayed under whoever was
  // signed in (advisor 1.5). The shell produces exactly that overlap: the identity watcher's
  // fire-and-forget adopt racing the awaited boot adopt.
  if (inflightHydrate && inflightNs === (ns || ANON_NAMESPACE)) return inflightHydrate
  const target = ns || ANON_NAMESPACE
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
    pending = []
    hydrated = false
    namespace = next
  }
  if (hydrated) return
  const stored = (await getDeviceJson<PendingListen[]>(pendingKeyFor(next))) ?? []
  // The namespace can have moved while we were reading — see the same guard in outbox.ts and
  // playbackPositions.ts. Without it the loser of two concurrent different-namespace hydrates
  // merges A's listens into memory under B and persists them to B's key.
  if (namespace !== next) return
  // Anything recorded before this resolved is newer than the disk copy — same rule as the
  // downloads registry, and `persist` refuses to write before hydration so it cannot be clobbered.
  const inMemory = pending
  pending = [...stored, ...inMemory]
  hydrated = true
  if (inMemory.length && namespace === next) persist()
}

function persist(): void {
  if (!hydrated) return
  void setDeviceJson(pendingKeyFor(namespace), pending).catch(() => {})
}

/** Queue a listen that did not reach the server. Capped so a long offline stretch cannot grow without bound. */
export const MAX_PENDING = 500

export function queueListen(slug: string, ts: number = Math.floor(Date.now() / 1000)): void {
  pending.push({ slug, ts })
  if (pending.length > MAX_PENDING) pending = pending.slice(-MAX_PENDING)
  persist()
}

export function pendingListens(): readonly PendingListen[] {
  return pending
}

/**
 * Push everything queued, oldest first, stopping at the first failure so a dead network is not
 * hammered. Delivered events are dropped; the rest stay for the next reconnect.
 */
export async function flushListenLog(
  push: (slug: string, ts: number) => Promise<boolean>,
): Promise<number> {
  if (flushing) return 0
  flushing = true
  try {
  if (!pending.length) return 0
  // Same rule as the outbox (advisor 1.4): a switch mid-loop would log A's listens into B's
  // history, and then prune A's queue using B's array.
  const generation = identityEpoch()
  const startedIn = namespace
  const ordered = [...pending].sort((a, b) => a.ts - b.ts)
  let flushed = 0
  for (const item of ordered) {
    if (identityChangedSince(generation) || namespace !== startedIn) break
    let ok = false
    try {
      ok = await push(item.slug, item.ts)
    } catch {
      ok = false
    }
    if (!ok) break
    flushed += 1
  }
  if (flushed && !identityChangedSince(generation) && namespace === startedIn) {
    const delivered = new Set(ordered.slice(0, flushed))
    pending = pending.filter((p) => !delivered.has(p))
    persist()
  }
    return flushed
  } finally {
    flushing = false
  }
}

/** Test seam. */
export function __resetListenLog(): void {
  pending = []
  hydrated = false
  namespace = ANON_NAMESPACE
}
