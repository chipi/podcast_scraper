/**
 * Per-slug snapshot of a PlayerView's loaded surface (#16). The player view is a route, so it is
 * unmounted the moment you leave it — reopening an episode (typically the one already playing, via
 * the mini-player) would otherwise blank the page and re-fetch every rail behind a spinner even
 * though nothing changed. This module-scope cache outlives any single view instance, so a reopen can
 * paint instantly from the last snapshot while the view revalidates in place.
 *
 * Bounded LRU (episode detail is immutable within a session, so staleness is a non-issue; the bound
 * only caps memory). Snapshots hold the ref *values* — never the reactive refs — and PlayerView
 * REASSIGNS its refs on load rather than mutating in place, so a cached array can never be aliased
 * and clobbered by a later load.
 */
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

const MAX = 4
const cache = new Map<string, PlayerViewSnapshot>()

export function getPlayerViewSnapshot(slug: string): PlayerViewSnapshot | undefined {
  return cache.get(slug)
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
}

export function clearPlayerViewCache(): void {
  cache.clear()
}
