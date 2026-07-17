import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import type { GraphFilterState, ParsedArtifact } from '../types/artifact'
import { useArtifactsStore } from './artifacts'
import { useGraphLoadModeStore } from './graphLoadMode'
import {
  applyGraphDefaultNodeTypeVisibility,
  applyGraphFilters,
  defaultFilterState,
  filtersActive,
  graphTypesDeviateFromGraphSpec,
} from '../utils/parsing'
import { expandFilteredArtifactEgoWithTopicClusterNeighbors } from '../utils/topicClustersOverlay'
import { augmentArtifactWithTrail } from '../utils/graphTrail'
import { useGraphNavigationStore } from './graphNavigation'

export const useGraphFilterStore = defineStore('graphFilters', () => {
  const artifacts = useArtifactsStore()
  const loadMode = useGraphLoadModeStore()
  const nav = useGraphNavigationStore()
  const state = ref<GraphFilterState | null>(null)

  /* graph-v3 tier 8-1 — the full artifact the graph builds against
   * picks between the top-down synthetic slice (super-theme nodes
   * only, ~6-8 nodes) and the merged display artifact based on the
   * load-mode opt-in. Falls back to the display artifact when the
   * top-down slice isn't available (no theme_clusters doc yet). */
  const fullArtifact = computed<ParsedArtifact | null>(() => {
    if (loadMode.isTopDown) {
      const td = artifacts.topDownDisplayArtifact
      if (td) return td
    }
    return artifacts.displayArtifact
  })

  watch(
    fullArtifact,
    (art) => {
      if (!art) {
        state.value = null
        return
      }
      const st = defaultFilterState(art)
      if (st) {
        applyGraphDefaultNodeTypeVisibility(st)
      }
      state.value = st
    },
    { immediate: true },
  )

  const filteredArtifact = computed(() => {
    const full = fullArtifact.value
    const st = state.value
    if (!full || !st) return null
    /* graph-v3 tier 8-1 — top-down slice is already the browse-surface
     * (SuperTheme nodes only). Skip type-filtering so the filter chip
     * doesn't hide the whole graph when the user hasn't opted `SuperTheme`
     * into `allowedTypes` yet. Tier 8-4 will re-introduce filter semantics
     * that reason over the expanded slice. */
    if (loadMode.isTopDown && artifacts.topDownDisplayArtifact === full) {
      return full
    }
    return applyGraphFilters(full, st)
  })

  const filtersAreActive = computed(() =>
    filtersActive(fullArtifact.value, state.value),
  )

  const graphTypesDeviateFromDefaults = computed(() =>
    graphTypesDeviateFromGraphSpec(state.value),
  )

  function viewWithEgo(focusId: string | null): ParsedArtifact | null {
    const base = filteredArtifact.value
    if (!base) return null
    const ego = expandFilteredArtifactEgoWithTopicClusterNeighbors(
      base,
      focusId,
      artifacts.topicClustersDoc,
    )
    // #6 L0 — union in the navigation breadcrumb trail (default-empty → unchanged). Trail nodes are
    // pulled from ``base`` (the full type-filtered graph), so hidden types stay hidden.
    return augmentArtifactWithTrail(ego, nav.trailNodeIds, base)
  }

  function setHideUngrounded(v: boolean): void {
    if (state.value) state.value.hideUngroundedInsights = v
  }

  function setShowGiLayer(v: boolean): void {
    if (state.value) state.value.showGiLayer = v
  }

  function setShowKgLayer(v: boolean): void {
    if (state.value) state.value.showKgLayer = v
  }

  function toggleAllowedType(t: string): void {
    if (!state.value) return
    if (!(t in state.value.allowedTypes)) return
    const cur = state.value.allowedTypes[t]
    state.value.allowedTypes = {
      ...state.value.allowedTypes,
      [t]: !cur,
    }
  }

  function toggleAllowedEdgeType(edgeType: string): void {
    if (!state.value) return
    if (!(edgeType in state.value.allowedEdgeTypes)) return
    const cur = state.value.allowedEdgeTypes[edgeType]
    state.value.allowedEdgeTypes = {
      ...state.value.allowedEdgeTypes,
      [edgeType]: !cur,
    }
  }

  function selectAllEdgeTypes(): void {
    if (!state.value) return
    const next: Record<string, boolean> = { ...state.value.allowedEdgeTypes }
    for (const k of Object.keys(next)) {
      next[k] = true
    }
    state.value.allowedEdgeTypes = next
  }

  function selectAllTypes(): void {
    if (!state.value) return
    const next: Record<string, boolean> = { ...state.value.allowedTypes }
    for (const k of Object.keys(next)) {
      next[k] = true
    }
    state.value.allowedTypes = next
  }

  function deselectAllTypes(): void {
    if (!state.value) return
    const next: Record<string, boolean> = { ...state.value.allowedTypes }
    for (const k of Object.keys(next)) {
      next[k] = false
    }
    state.value.allowedTypes = next
  }

  function resetGraphTypeVisibilityDefaults(): void {
    if (!state.value) return
    applyGraphDefaultNodeTypeVisibility(state.value)
  }

  function setFeedFilter(feedId: string | null): void {
    if (!state.value) return
    const v = feedId == null ? null : feedId.trim()
    state.value.graphFeedFilterId = v && v !== '' ? v : null
  }

  function clearFeedFilter(): void {
    if (!state.value) return
    state.value.graphFeedFilterId = null
  }

  function deselectAllEdgeTypes(): void {
    if (!state.value) return
    const next: Record<string, boolean> = { ...state.value.allowedEdgeTypes }
    for (const k of Object.keys(next)) {
      next[k] = false
    }
    state.value.allowedEdgeTypes = next
  }

  /**
   * Atomic reset across every chip dimension (#658 ``× reset all``).
   * Returns Types to graph-spec defaults (Quote / Speaker off), Sources
   * to all on (gi+kg+show ungrounded), Edges all on, Degree cleared
   * (handled by graphExplorer caller), Feed cleared.
   */
  function resetAllFilters(): void {
    if (!state.value) return
    selectAllEdgeTypes()
    state.value.hideUngroundedInsights = false
    state.value.showGiLayer = true
    state.value.showKgLayer = true
    state.value.graphFeedFilterId = null
    applyGraphDefaultNodeTypeVisibility(state.value)
  }

  return {
    state,
    fullArtifact,
    filteredArtifact,
    filtersAreActive,
    graphTypesDeviateFromDefaults,
    viewWithEgo,
    setHideUngrounded,
    setShowGiLayer,
    setShowKgLayer,
    toggleAllowedType,
    selectAllTypes,
    deselectAllTypes,
    resetGraphTypeVisibilityDefaults,
    toggleAllowedEdgeType,
    selectAllEdgeTypes,
    deselectAllEdgeTypes,
    setFeedFilter,
    clearFeedFilter,
    resetAllFilters,
  }
})
