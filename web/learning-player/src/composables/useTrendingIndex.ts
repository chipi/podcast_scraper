import { computed, ref, watch, type Ref } from "vue"
import { getTrending } from "../services/api"
import type { TrendingScope } from "./useTrendingScope"

/** Velocity + weekly series for one trending entity, keyed by its id. */
export type TrendingEntry = { v: number; series: number[] }

/**
 * Fetch the trending set for a kind ONCE and expose it as an id→{velocity, series} map, so a
 * detail surface can look up its own entity's momentum by id. This was the same ten lines copied in
 * the topic card and the storyline sheet (fetch top-N, build the map, swallow errors); momentum is
 * decoration, so a failed fetch leaves an empty map and the surface renders without a badge.
 *
 * `scope` defaults to `corpus`: a card's momentum badge says how the entity moves across everyone.
 * It used to follow the Trends lens, but "mine" became strictly the listener's own world
 * (2026-10-07), which would strip the badge from every entity outside it. Pass it to override.
 *
 * The scope is tracked REACTIVELY: when the lens resolves late (prefs load after mount) or the user
 * flips it, the map re-fetches — the old version snapshotted the lens once at setup and could fetch
 * the wrong lens forever.
 */
export function useTrendingIndex(
  kind: string,
  scope?: TrendingScope,
  limit = 50
): Ref<Record<string, TrendingEntry>> {
  const scopeRef = computed<TrendingScope>(() => scope ?? "corpus")
  const index = ref<Record<string, TrendingEntry>>({})
  watch(
    scopeRef,
    (resolved) => {
      void getTrending(kind, resolved, limit)
        .then((rows) => {
          const m: Record<string, TrendingEntry> = {}
          for (const r of rows) m[r.entity_id] = { v: r.velocity, series: r.series }
          index.value = m
        })
        .catch(() => {
          /* momentum is decoration; the surface renders without it */
        })
    },
    { immediate: true }
  )
  return index
}
