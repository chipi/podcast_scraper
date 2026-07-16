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
 *   - V5 `nodeSizeByDegree` — **on** by default (graph-v3 C). Replaces
 *     fixed Topic + Episode width/height with `mapData(degreeHeat, ...)`
 *     so hub structure reads at a glance.
 *   - V6 `themeClusterRegions` — **off** initially (graph-v3 R-V). Paints
 *     a soft underlay tint over every node in each theme cluster (via the
 *     `topic_theme_clusters` enricher artifact + viewer-side propagation
 *     from Topics to Insights/Episodes/Persons/Podcasts/Orgs). Enricher-
 *     gated: the toggle is hidden entirely when the artifact is not
 *     present, so users don't see a dead control.
 *   - V7 `bridgeRing` — **on** by default (graph-v3 K validated on
 *     prod-v2). Rose dashed border on high-betweenness nodes (Topic /
 *     Podcast / Person / Org) so cross-community bridges pop.
 *
 * V2 (Insight grounding + tier classes) is unconditional: the
 * stylesheet selectors only fire when the matching class is assigned
 * at element-build time, so legacy artifacts get no behaviour change.
 *
 * Flags persist to `localStorage` under a single JSON key so the
 * browser remembers the user's last lens state across reloads.
 * Cross-device sync is the USERPREFS-1 arc (out of scope here).
 */

const STORAGE_KEY = 'ps_graph_lenses'

export interface GraphLensFlags {
  aggregatedEdges: boolean
  nodeSizeByDegree: boolean
  themeClusterRegions: boolean
  bridgeRing: boolean
}

const DEFAULT_FLAGS: GraphLensFlags = {
  aggregatedEdges: false,
  /* graph-v3 C — V5 promoted to default-on. */
  nodeSizeByDegree: true,
  /* graph-v3 R-V — V6 opt-in during rollout. Enricher-gated: even when
     set to true the toggle stays hidden if the theme-cluster artifact
     is not available for the current corpus. */
  themeClusterRegions: false,
  /* graph-v3 K/N — V7 default-on because K (bridge betweenness ring) was
     already validated on prod-v2 in commit d8447b8a. Users who find
     the rose rings noisy can toggle off. */
  bridgeRing: true,
}

function readBool(v: unknown, fallback: boolean): boolean {
  return typeof v === 'boolean' ? v : fallback
}

function readInitialFlags(): GraphLensFlags {
  try {
    if (typeof localStorage === 'undefined') return { ...DEFAULT_FLAGS }
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return { ...DEFAULT_FLAGS }
    const parsed = JSON.parse(raw) as Partial<GraphLensFlags> & {
      /** Legacy key from an intermediate graph-v3 iteration (MCL, since
       *  reverted). Migrated once into `themeClusterRegions` so users
       *  who toggled it don't lose their opt-in. */
      communityColours?: boolean
    }
    return {
      aggregatedEdges: readBool(parsed.aggregatedEdges, DEFAULT_FLAGS.aggregatedEdges),
      nodeSizeByDegree: readBool(parsed.nodeSizeByDegree, DEFAULT_FLAGS.nodeSizeByDegree),
      themeClusterRegions: readBool(
        parsed.themeClusterRegions ?? parsed.communityColours,
        DEFAULT_FLAGS.themeClusterRegions,
      ),
      bridgeRing: readBool(parsed.bridgeRing, DEFAULT_FLAGS.bridgeRing),
    }
  } catch {
    return { ...DEFAULT_FLAGS }
  }
}

export const useGraphLensesStore = defineStore('graphLenses', () => {
  const initial = readInitialFlags()
  const aggregatedEdges = ref(initial.aggregatedEdges)
  const nodeSizeByDegree = ref(initial.nodeSizeByDegree)
  const themeClusterRegions = ref(initial.themeClusterRegions)
  const bridgeRing = ref(initial.bridgeRing)

  const flags = computed<GraphLensFlags>(() => ({
    aggregatedEdges: aggregatedEdges.value,
    nodeSizeByDegree: nodeSizeByDegree.value,
    themeClusterRegions: themeClusterRegions.value,
    bridgeRing: bridgeRing.value,
  }))

  function setAggregatedEdges(v: boolean): void {
    aggregatedEdges.value = v
  }

  function setNodeSizeByDegree(v: boolean): void {
    nodeSizeByDegree.value = v
  }

  function setThemeClusterRegions(v: boolean): void {
    themeClusterRegions.value = v
  }

  function setBridgeRing(v: boolean): void {
    bridgeRing.value = v
  }

  function resetToDefaults(): void {
    aggregatedEdges.value = DEFAULT_FLAGS.aggregatedEdges
    nodeSizeByDegree.value = DEFAULT_FLAGS.nodeSizeByDegree
    themeClusterRegions.value = DEFAULT_FLAGS.themeClusterRegions
    bridgeRing.value = DEFAULT_FLAGS.bridgeRing
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
    themeClusterRegions,
    bridgeRing,
    flags,
    setAggregatedEdges,
    setNodeSizeByDegree,
    setThemeClusterRegions,
    setBridgeRing,
    resetToDefaults,
  }
})
