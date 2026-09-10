import { ref, type Ref } from "vue"
import { getTrending } from "../services/api"

/** Velocity + weekly series for one trending entity, keyed by its id. */
export type TrendingEntry = { v: number; series: number[] }

/**
 * Fetch the corpus trending set for a kind ONCE and expose it as an id→{velocity, series} map, so a
 * detail surface can look up its own entity's momentum by id. This was the same ten lines copied in
 * the topic card and the storyline sheet (fetch top-N, build the map, swallow errors); momentum is
 * decoration, so a failed fetch leaves an empty map and the surface renders without a badge.
 */
export function useTrendingIndex(
  kind: string,
  scope: "corpus" | "mine" = "corpus",
  limit = 50
): Ref<Record<string, TrendingEntry>> {
  const index = ref<Record<string, TrendingEntry>>({})
  void getTrending(kind, scope, limit)
    .then((rows) => {
      const m: Record<string, TrendingEntry> = {}
      for (const r of rows) m[r.entity_id] = { v: r.velocity, series: r.series }
      index.value = m
    })
    .catch(() => {
      /* momentum is decoration; the surface renders without it */
    })
  return index
}
