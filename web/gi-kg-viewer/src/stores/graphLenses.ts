import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'

/**
 * RFC-080 — graph visualization lens flags.
 *
 * Each lens is a render-only behaviour the user can toggle without
 * reloading the corpus. Defaults match the RFC rollout table:
 *
 *   - V1 `aggregatedEdges` — **off** initially. Renders Episode↔Topic
 *     and Episode↔Person aggregated edges (`ABOUT_AGG`,
 *     `SPOKE_IN_AGG`). Default-on decision deferred to a corpus-level
 *     validation round.
 *   - V5 `nodeSizeByDegree` — **off** initially. Replaces fixed Topic
 *     and Episode width/height with `mapData(degreeHeat, ...)`.
 *     Visually invasive, so staged before promotion.
 *
 * V2 (Insight grounding + tier classes) is unconditional: the
 * stylesheet selectors only fire when the matching class is assigned
 * at element-build time, so legacy artifacts get no behaviour change.
 *
 * Flags persist to `localStorage` under a single JSON key so the
 * browser remembers the user's last lens state across reloads.
 */

const STORAGE_KEY = 'ps_graph_lenses'

export interface GraphLensFlags {
  aggregatedEdges: boolean
  nodeSizeByDegree: boolean
}

const DEFAULT_FLAGS: GraphLensFlags = {
  aggregatedEdges: false,
  nodeSizeByDegree: false,
}

function readInitialFlags(): GraphLensFlags {
  try {
    if (typeof localStorage === 'undefined') return { ...DEFAULT_FLAGS }
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return { ...DEFAULT_FLAGS }
    const parsed = JSON.parse(raw) as Partial<GraphLensFlags>
    return {
      aggregatedEdges: typeof parsed.aggregatedEdges === 'boolean'
        ? parsed.aggregatedEdges
        : DEFAULT_FLAGS.aggregatedEdges,
      nodeSizeByDegree: typeof parsed.nodeSizeByDegree === 'boolean'
        ? parsed.nodeSizeByDegree
        : DEFAULT_FLAGS.nodeSizeByDegree,
    }
  } catch {
    return { ...DEFAULT_FLAGS }
  }
}

export const useGraphLensesStore = defineStore('graphLenses', () => {
  const initial = readInitialFlags()
  const aggregatedEdges = ref(initial.aggregatedEdges)
  const nodeSizeByDegree = ref(initial.nodeSizeByDegree)

  const flags = computed<GraphLensFlags>(() => ({
    aggregatedEdges: aggregatedEdges.value,
    nodeSizeByDegree: nodeSizeByDegree.value,
  }))

  function setAggregatedEdges(v: boolean): void {
    aggregatedEdges.value = v
  }

  function setNodeSizeByDegree(v: boolean): void {
    nodeSizeByDegree.value = v
  }

  function resetToDefaults(): void {
    aggregatedEdges.value = DEFAULT_FLAGS.aggregatedEdges
    nodeSizeByDegree.value = DEFAULT_FLAGS.nodeSizeByDegree
  }

  watch(
    flags,
    (v) => {
      try {
        if (typeof localStorage !== 'undefined') {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(v))
        }
      } catch {
        /* ignore quota / private mode */
      }
    },
    { deep: false },
  )

  return {
    aggregatedEdges,
    nodeSizeByDegree,
    flags,
    setAggregatedEdges,
    setNodeSizeByDegree,
    resetToDefaults,
  }
})
