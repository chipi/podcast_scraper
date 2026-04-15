import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import type { GraphFilterState, ParsedArtifact } from '../types/artifact'
import { useArtifactsStore } from './artifacts'
import {
  applyGraphFilters,
  defaultFilterState,
  filtersActive,
} from '../utils/parsing'
import { expandFilteredArtifactEgoWithTopicClusterNeighbors } from '../utils/topicClustersOverlay'

export const useGraphFilterStore = defineStore('graphFilters', () => {
  const artifacts = useArtifactsStore()
  const state = ref<GraphFilterState | null>(null)

  watch(
    () => artifacts.displayArtifact,
    (art) => {
      state.value = art ? defaultFilterState(art) : null
    },
    { immediate: true },
  )

  const fullArtifact = computed(() => artifacts.displayArtifact)

  const filteredArtifact = computed(() => {
    const full = fullArtifact.value
    const st = state.value
    if (!full || !st) return null
    return applyGraphFilters(full, st)
  })

  const filtersAreActive = computed(() =>
    filtersActive(fullArtifact.value, state.value),
  )

  function viewWithEgo(focusId: string | null): ParsedArtifact | null {
    const base = filteredArtifact.value
    if (!base) return null
    return expandFilteredArtifactEgoWithTopicClusterNeighbors(
      base,
      focusId,
      artifacts.topicClustersDoc,
    )
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

  return {
    state,
    fullArtifact,
    filteredArtifact,
    filtersAreActive,
    viewWithEgo,
    setHideUngrounded,
    setShowGiLayer,
    setShowKgLayer,
    toggleAllowedType,
    selectAllTypes,
    deselectAllTypes,
    toggleAllowedEdgeType,
    selectAllEdgeTypes,
  }
})
