/**
 * Device-local playback positions (#1906 Phase 1).
 *
 * A podcast app that forgets where you were is broken at its core, and `GET /playback` fails
 * offline — so without this, every downloaded episode resumes at 0. Positions are also the ONE
 * offline write that ships early; a general write outbox is deliberately out of scope (the queue
 * is a whole-list PUT, which makes replay a conflict-resolution problem, not a retry problem).
 *
 * Held in memory so reads are synchronous — `PlayerView` needs the resume point on its critical
 * path, and the store's position persister is called on a throttle from a sync context.
 */

import { getDeviceJson, setDeviceJson } from './deviceStore'

/**
 * Namespaced per ACCOUNT, exactly like the downloads registry (#1905). It was device-global, which
 * meant on a shared phone: account A's offline positions were flushed into B's server state on
 * reconnect, B resumed at A's position, and `reclaimFinished` read A's `finished` flags to
 * auto-evict B's *unplayed* download — breaking two product rulings at once.
 */
export const POSITIONS_KEY_PREFIX = 'playback.positions'
export const ANON_NAMESPACE = 'anon'

export function positionsKeyFor(namespace: string): string {
  return `${POSITIONS_KEY_PREFIX}.${namespace}`
}

/**
 * How far ahead of the server's stamp ours must be before we trust it.
 *
 * `server.updatedAt` is the server's arrival clock; `local.updatedAt` is the device's. A device
 * clock running a few minutes fast would otherwise let a stale offline write out-stamp a genuinely
 * newer write from another device — the exact clobber this comparison exists to prevent. Inside
 * the margin we fall back to the conservative forward-only rule.
 */
export const CLOCK_SKEW_MARGIN_MS = 5 * 60 * 1000

export interface LocalPosition {
  seconds: number
  finished: boolean
  updatedAt: number
  /** Written while offline (or the server write failed) and not yet pushed. */
  pending?: boolean
}

let positions: Record<string, LocalPosition> = {}
let hydrated = false
let namespace = ANON_NAMESPACE

/**
 * Load this account's positions. Call at boot and again whenever the signed-in user changes;
 * switching accounts drops the in-memory map rather than leaking it across the switch.
 */
export async function hydratePositions(ns: string = ANON_NAMESPACE): Promise<void> {
  const next = ns || ANON_NAMESPACE
  if (next !== namespace) {
    positions = {}
    hydrated = false
    namespace = next
  }
  if (hydrated) return
  const stored =
    (await getDeviceJson<Record<string, LocalPosition>>(positionsKeyFor(namespace))) ?? {}
  // Anything already in memory was recorded before this resolved and is strictly newer than the
  // disk copy — the same rule the downloads registry uses. Assigning `stored` outright would drop
  // it, and `persist()` refuses to write before hydration precisely so it cannot be clobbered.
  const pending = Object.keys(positions).length > 0
  positions = { ...stored, ...positions }
  hydrated = true
  if (pending) persist()
}

/**
 * Refuses to write before the first hydrate: `positions` is not yet the union of what is on disk,
 * so writing it would clobber every stored position for this account.
 */
function persist(): void {
  if (!hydrated) return
  // Swallowed: a lost position is small and self-correcting, and this runs from a throttled sync
  // context that must not reject.
  void setDeviceJson(positionsKeyFor(namespace), positions).catch(() => {})
}

/** Synchronous by design — see the module note. */
export function localPosition(slug: string): LocalPosition | null {
  return positions[slug] ?? null
}

/**
 * Record where the user is. `synced` is false when the server write failed or was never
 * attempted, which is what marks it for a later push.
 */
export function recordPosition(
  slug: string,
  seconds: number,
  finished: boolean,
  synced: boolean,
  now: number = Date.now(),
): void {
  positions[slug] = { seconds, finished, updatedAt: now, ...(synced ? {} : { pending: true }) }
  persist()
}

/** Positions written offline, oldest first. */
export function pendingPositions(): Array<{ slug: string } & LocalPosition> {
  return Object.entries(positions)
    .filter(([, p]) => p.pending)
    .map(([slug, p]) => ({ slug, ...p }))
    .sort((a, b) => a.updatedAt - b.updatedAt)
}

/** What the server currently holds for an episode. `updatedAt` is UNIX seconds, or null. */
export interface RemotePosition {
  seconds: number
  finished: boolean
  updatedAt: number | null
}

/**
 * Should an offline write overwrite what the server holds?
 *
 * The server DOES stamp `updated_at` on playback records, but `put_playback` sets it to
 * `int(time.time())` — the moment the write ARRIVED, not the moment the listener was there. So a
 * position we recorded in airplane mode and push an hour later would out-stamp a laptop write
 * made in between, and true write-time last-writer-wins still needs the server to accept a
 * client timestamp.
 *
 * What the arrival stamp does tell us reliably: if the server's record landed AFTER we went
 * offline, some other device has newer information and we must not clobber it. Where the stamps
 * are equal or missing, only move progress forward.
 */
export function shouldPush(local: LocalPosition, server: RemotePosition | null): boolean {
  if (!server) return true
  if (server.updatedAt != null) {
    // Both stamps are known, but they come from different clocks — so ours must be clear of the
    // skew margin before we trust it over the server's. Beyond the margin the newer write wins
    // outright, which is what lets a REWIND made offline sync back; the forward-only rule below
    // would silently discard it.
    if (local.updatedAt > server.updatedAt * 1000 + CLOCK_SKEW_MARGIN_MS) return true
    if (server.updatedAt * 1000 > local.updatedAt + CLOCK_SKEW_MARGIN_MS) return false
    // Inside the margin the stamps tell us nothing reliable; fall through to forward-only.
  }
  // No server stamp to compare against, so fall back to the only safe assumption: progress
  // moves forward, and finishing is worth reporting even from behind.
  return local.finished || local.seconds > server.seconds
}

/**
 * Push everything written offline, oldest first, without overwriting newer progress from another
 * device. A position we decline to push stops being pending — the server's value is better, and
 * retrying it on every reconnect forever would be pure noise.
 */
export async function flushPendingPositions(
  push: (slug: string, seconds: number, finished: boolean) => Promise<void>,
  read?: (slug: string) => Promise<RemotePosition | null>,
): Promise<number> {
  const pending = pendingPositions()
  let flushed = 0
  for (const p of pending) {
    try {
      const server = read ? await read(p.slug) : null
      if (shouldPush(p, server)) await push(p.slug, p.seconds, p.finished)
      const current = positions[p.slug]
      // Left alone if it moved again mid-flush; the next flush takes it.
      if (current && current.updatedAt === p.updatedAt) {
        delete current.pending
        flushed += 1
      }
    } catch {
      // Still offline. Keep it pending and try again on the next reconnect.
      break
    }
  }
  if (flushed) persist()
  return flushed
}

/** Test seam. */
export function __resetPositions(): void {
  positions = {}
  hydrated = false
  namespace = ANON_NAMESPACE
}
