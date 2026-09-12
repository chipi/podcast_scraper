import { computed, ref, watch, type Ref } from "vue"
import { getTrending } from "../services/api"
import { useTrendingScope, type TrendingScope } from "./useTrendingScope"

/** Velocity + weekly series for one trending entity, keyed by its id. */
export type TrendingEntry = { v: number; series: number[] }

/**
 * Fetch the trending set for a kind ONCE and expose it as an id→{velocity, series} map, so a
 * detail surface can look up its own entity's momentum by id. This was the same ten lines copied in
 * the topic card and the storyline sheet (fetch top-N, build the map, swallow errors); momentum is
 * decoration, so a failed fetch leaves an empty map and the surface renders without a badge.
 *
 * `scope` defaults to the app-level trending lens (#2030) so card/sheet momentum badges follow the
 * same Corpus ⇄ My-listening choice as the Home rails; pass it explicitly to override.
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
  const scopeRef = scope != null ? computed(() => scope) : useTrendingScope().scope
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
