import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'

import { useUserPreferencesStore } from './userPreferences'

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
  /* graph-v3 Tier 5C — enricher-based decoration lenses. */
  velocityHalo: boolean
  personCredibility: boolean
  /* graph-v3 Tier 5D — enricher-based edge overlays. */
  consensusEdges: boolean
  coGuestEdges: boolean
  /* graph-v3 Tier 7-4 — person co-appearance community underlay
     (guest_coappearance enricher v1.1.0+, `communities[]`). */
  personCommunities: boolean
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
  /* graph-v3 Tier 5C/5D — enricher-based lenses, opt-in during rollout.
     Each is enricher-gated in the popover (hidden when the underlying
     artifact is absent). Insight sentiment tint (originally scoped as
     tier 5C-3) deferred pending a corpus-scope sentiment aggregation
     endpoint — the current shape ships per-episode sidecars that would
     need N per-episode fetches to render across the whole graph. */
  velocityHalo: false,
  personCredibility: false,
  consensusEdges: false,
  coGuestEdges: false,
  /* graph-v3 Tier 7-4 — opt-in during rollout; enricher-gated on
     `guest_coappearance.communities[]` presence in the artifact. */
  personCommunities: false,
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
      velocityHalo: readBool(parsed.velocityHalo, DEFAULT_FLAGS.velocityHalo),
      personCredibility: readBool(parsed.personCredibility, DEFAULT_FLAGS.personCredibility),
      consensusEdges: readBool(parsed.consensusEdges, DEFAULT_FLAGS.consensusEdges),
      coGuestEdges: readBool(parsed.coGuestEdges, DEFAULT_FLAGS.coGuestEdges),
      personCommunities: readBool(parsed.personCommunities, DEFAULT_FLAGS.personCommunities),
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
  const velocityHalo = ref(initial.velocityHalo)
  const personCredibility = ref(initial.personCredibility)
  const consensusEdges = ref(initial.consensusEdges)
  const coGuestEdges = ref(initial.coGuestEdges)
  const personCommunities = ref(initial.personCommunities)

  const flags = computed<GraphLensFlags>(() => ({
    aggregatedEdges: aggregatedEdges.value,
    nodeSizeByDegree: nodeSizeByDegree.value,
    themeClusterRegions: themeClusterRegions.value,
    bridgeRing: bridgeRing.value,
    velocityHalo: velocityHalo.value,
    personCredibility: personCredibility.value,
    consensusEdges: consensusEdges.value,
    coGuestEdges: coGuestEdges.value,
    personCommunities: personCommunities.value,
  }))

  function setAggregatedEdges(v: boolean): void { aggregatedEdges.value = v }
  function setNodeSizeByDegree(v: boolean): void { nodeSizeByDegree.value = v }
  function setThemeClusterRegions(v: boolean): void { themeClusterRegions.value = v }
  function setBridgeRing(v: boolean): void { bridgeRing.value = v }
  function setVelocityHalo(v: boolean): void { velocityHalo.value = v }
  function setPersonCredibility(v: boolean): void { personCredibility.value = v }
  function setConsensusEdges(v: boolean): void { consensusEdges.value = v }
  function setCoGuestEdges(v: boolean): void { coGuestEdges.value = v }
  function setPersonCommunities(v: boolean): void { personCommunities.value = v }

  function resetToDefaults(): void {
    aggregatedEdges.value = DEFAULT_FLAGS.aggregatedEdges
    nodeSizeByDegree.value = DEFAULT_FLAGS.nodeSizeByDegree
    themeClusterRegions.value = DEFAULT_FLAGS.themeClusterRegions
    bridgeRing.value = DEFAULT_FLAGS.bridgeRing
    velocityHalo.value = DEFAULT_FLAGS.velocityHalo
    personCredibility.value = DEFAULT_FLAGS.personCredibility
    consensusEdges.value = DEFAULT_FLAGS.consensusEdges
    coGuestEdges.value = DEFAULT_FLAGS.coGuestEdges
    personCommunities.value = DEFAULT_FLAGS.personCommunities
  }

  /* USERPREFS-1 — cross-device write-through. Every flags mutation writes
     to localStorage (offline / anonymous baseline) AND fires a PATCH to
     `/api/app/preferences` under the `graphLenses` key. The server call is
     silent-on-failure — the localStorage mirror is authoritative until the
     user preferences endpoint responds. Suppressed by `applyingRemote` so
     the "server pushed a new value onto local refs" round doesn't echo the
     same payload back to the server. */
  const userPrefs = useUserPreferencesStore()
  let applyingRemote = false

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
      if (applyingRemote) return
      // Fire-and-forget; store handles auth/offline failure by flipping its
      // `available` flag off, so subsequent writes stay local-only.
      void userPrefs.set('graphLenses', v)
    },
    { deep: false },
  )

  /* Once the user preferences store hydrates from the server, apply any
     `graphLenses` payload it fetched onto the local refs. Server value wins
     over the localStorage snapshot only when it's actually present — an
     empty / undefined payload means "user hasn't sync'd yet", so the
     localStorage fallback stays intact. */
  watch(
    () => userPrefs.hydrated,
    (ready) => {
      if (!ready) return
      const remote = userPrefs.get<Partial<GraphLensFlags>>('graphLenses')
      if (!remote || typeof remote !== 'object') return
      applyingRemote = true
      aggregatedEdges.value = readBool(remote.aggregatedEdges, aggregatedEdges.value)
      nodeSizeByDegree.value = readBool(remote.nodeSizeByDegree, nodeSizeByDegree.value)
      themeClusterRegions.value = readBool(
        remote.themeClusterRegions,
        themeClusterRegions.value,
      )
      bridgeRing.value = readBool(remote.bridgeRing, bridgeRing.value)
      velocityHalo.value = readBool(remote.velocityHalo, velocityHalo.value)
      personCredibility.value = readBool(remote.personCredibility, personCredibility.value)
      consensusEdges.value = readBool(remote.consensusEdges, consensusEdges.value)
      coGuestEdges.value = readBool(remote.coGuestEdges, coGuestEdges.value)
      personCommunities.value = readBool(remote.personCommunities, personCommunities.value)
      /* Same async-guard concern as theme.ts / graphLoadMode.ts /
       * graphTopDown.ts: the individual flag-watches above are async
       * (default 'pre' flush), so clearing ``applyingRemote`` in a
       * finally-block would let each remote-applied write loop back
       * into ``userPrefs.set``. Defer the clear to the next microtask. */
      void Promise.resolve().then(() => {
        applyingRemote = false
      })
    },
    { immediate: true },
  )

  return {
    aggregatedEdges,
    nodeSizeByDegree,
    themeClusterRegions,
    bridgeRing,
    velocityHalo,
    personCredibility,
    consensusEdges,
    coGuestEdges,
    personCommunities,
    flags,
    setAggregatedEdges,
    setNodeSizeByDegree,
    setThemeClusterRegions,
    setBridgeRing,
    setVelocityHalo,
    setPersonCredibility,
    setConsensusEdges,
    setCoGuestEdges,
    setPersonCommunities,
    resetToDefaults,
  }
})
