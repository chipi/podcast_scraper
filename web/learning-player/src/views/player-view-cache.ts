/**
 * Per-slug snapshot of a PlayerView's loaded surface (#16), persisted per account (#1909).
 *
 * The player view is a route, so it is unmounted the moment you leave it — reopening an episode
 * (typically the one already playing, via the mini-player) would otherwise blank the page and
 * re-fetch every rail behind a spinner even though nothing changed. This cache outlives any single
 * view instance, so a reopen can paint instantly from the last snapshot while the view revalidates
 * in place.
 *
 * ## Why it is on disk now
 *
 * #1909 named this module directly: *"a persistent generalisation of the in-memory episode LRU in
 * `views/player-view-cache.ts`, whose interaction contract is already right — paint from snapshot,
 * revalidate in place, never wipe on reopen — and whose storage (RAM, 4 entries) is wrong."* That
 * half never landed. In RAM the cache died on every app relaunch, so the instant paint existed only
 * within a session: reopen the app and every episode was a cold spinner again, including one whose
 * audio was already downloaded to the device. Four entries also meant a listener moving between
 * five episodes evicted the one they started from.
 *
 * ## The RAM map stays, and is still the only read path
 *
 * `getPlayerViewSnapshot` is called synchronously inside `PlayerView.load()`, before the first
 * await, which is what makes the paint instant. Making it async would put a disk read in front of
 * the render and lose the property the cache exists for. So the map is the cache; disk is its
 * backing store, hydrated once per identity at boot.
 *
 * ## Transcripts are deliberately NOT persisted
 *
 * A segment list runs to ~76 KB per episode — an order of magnitude more than everything else in a
 * snapshot combined, for the one surface that is below the fold and opt-in. Persisting eight of them
 * would write most of a megabyte on a debounce timer for content most reopens never display. A
 * downloaded episode reads its transcript from its own file (`localTranscriptFor`), and any other
 * episode refetches it exactly as it does today. Snapshots hydrated from disk therefore come back
 * with an empty `segments`, which the view treats as "not loaded yet" — the state it would have been
 * in anyway.
 *
 * Snapshots hold the ref *values* — never the reactive refs — and PlayerView REASSIGNS its refs on
 * load rather than mutating in place, so a cached array can never be aliased and clobbered by a
 * later load.
 */
import { readCached, writeCached } from '../services/contentCache'
import type {
  EpisodeDetail,
  EpisodeStats,
  EpisodeSummary,
  Entity,
  Insight,
  Segment,
  Topic,
} from '../services/types'

export interface PlayerViewSnapshot {
  episode: EpisodeDetail | null
  segments: Segment[]
  audioUrl: string | null
  insights: Insight[]
  topics: Topic[]
  persons: Entity[]
  relatedEpisodes: EpisodeSummary[]
  stats: EpisodeStats | null
}

/** What actually goes to disk — see the note above about transcripts. */
type PersistedSnapshot = Omit<PlayerViewSnapshot, 'segments'>

/** The cache key, namespaced per account by `contentCache`. */
export const PLAYER_SNAPSHOT_KEY = 'player.snapshots'

/**
 * In RAM. Larger than the old 4 now that eviction is the only thing bounding it — a listener moving
 * between a handful of episodes should not evict the one they started from — but still bounded,
 * because a snapshot holds a whole transcript while it lives here.
 */
export const MAX = 12

/** On disk. Smaller than the RAM bound: this is "the last few episodes", not a library. */
export const PERSIST_MAX = 8

const cache = new Map<string, PlayerViewSnapshot>()

export function getPlayerViewSnapshot(slug: string): PlayerViewSnapshot | undefined {
  return cache.get(slug)
}

/**
 * Coalesced, because the caller is a watcher over eight refs that each land separately as the
 * page's rails arrive — a write per ref would be a dozen disk writes per episode opened, for a
 * value only the last of which is correct.
 */
let persistTimer: ReturnType<typeof setTimeout> | null = null
export const PERSIST_DEBOUNCE_MS = 1000

function schedulePersist(): void {
  if (persistTimer) return
  persistTimer = setTimeout(() => {
    persistTimer = null
    void persistNow()
  }, PERSIST_DEBOUNCE_MS)
}

async function persistNow(): Promise<void> {
  // Map preserves insertion order and `set` re-inserts, so the tail is the most recently used.
  const recent = [...cache.entries()].slice(-PERSIST_MAX)
  const payload: [string, PersistedSnapshot][] = recent.map(([slug, snap]) => {
    const { segments: _segments, ...rest } = snap
    return [slug, rest]
  })
  await writeCached(PLAYER_SNAPSHOT_KEY, payload)
}

/** Write any pending snapshot immediately. For tests, and for a deliberate flush before teardown. */
export async function flushPlayerViewSnapshots(): Promise<void> {
  if (persistTimer) {
    clearTimeout(persistTimer)
    persistTimer = null
  }
  await persistNow()
}

export function setPlayerViewSnapshot(slug: string, snapshot: PlayerViewSnapshot): void {
  // delete-then-set makes this the most-recently-used entry (Map keeps insertion order), so eviction
  // below drops the genuine oldest.
  cache.delete(slug)
  cache.set(slug, snapshot)
  while (cache.size > MAX) {
    const oldest = cache.keys().next().value
    if (oldest === undefined) break
    cache.delete(oldest)
  }
  schedulePersist()
}

/**
 * Load this account's snapshots into the map. Called from `adoptIdentity()` — the one place every
 * per-account store is pointed at the signed-in user — AFTER the cache namespace is set, so it can
 * only ever read the identity that owns them.
 *
 * Never overwrites a live entry: anything already in the map was loaded this session and is
 * therefore fresher than anything on disk.
 */
export async function hydratePlayerViewCache(): Promise<void> {
  const stored = await readCached<[string, PersistedSnapshot][]>(PLAYER_SNAPSHOT_KEY)
  if (!Array.isArray(stored)) return
  for (const entry of stored) {
    if (!Array.isArray(entry) || entry.length !== 2) continue
    const [slug, snap] = entry
    if (typeof slug !== 'string' || !snap || cache.has(slug)) continue
    cache.set(slug, { ...snap, segments: [] })
  }
}

/**
 * Forget everything in memory. Called between tests as "a fresh app launch", and on an identity
 * change — one account's snapshots must never paint for the next. Deliberately does NOT erase the
 * stored copy: sign-out clears it through `CACHE_KEYS`, and an account SWITCH should leave the
 * previous account's snapshots intact for when they come back.
 */
export function clearPlayerViewCache(): void {
  if (persistTimer) {
    clearTimeout(persistTimer)
    persistTimer = null
  }
  cache.clear()
}
