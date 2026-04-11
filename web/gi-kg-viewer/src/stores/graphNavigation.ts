import { defineStore } from 'pinia'
import { ref } from 'vue'

/** Cross-panel graph focus (e.g. search result → Cytoscape). */
export const useGraphNavigationStore = defineStore('graphNavigation', () => {
  const pendingFocusNodeId = ref<string | null>(null)
  /** Raw episode / node ids (e.g. metadata ``episode_id``) → yellow ``search-hit`` ring on graph. */
  const libraryHighlightSourceIds = ref<string[]>([])
  /**
   * Cytoscape ego (1-hop) center id; kept in sync from ``GraphCanvas`` so the App rail can call
   * ``viewWithEgo`` for graph node detail (same subgraph as the in-canvas panel had).
   */
  const graphEgoFocusCyId = ref<string | null>(null)

  function requestFocusNode(nodeId: string): void {
    const id = nodeId.trim()
    pendingFocusNodeId.value = id.length ? id : null
  }

  function clearPendingFocus(): void {
    pendingFocusNodeId.value = null
  }

  function setLibraryEpisodeHighlights(episodeIds: string[]): void {
    libraryHighlightSourceIds.value = episodeIds.map((s) => s.trim()).filter(Boolean)
  }

  function clearLibraryEpisodeHighlights(): void {
    libraryHighlightSourceIds.value = []
  }

  function setGraphEgoFocusCyId(id: string | null): void {
    graphEgoFocusCyId.value = id?.trim() ? id.trim() : null
  }

  return {
    pendingFocusNodeId,
    libraryHighlightSourceIds,
    graphEgoFocusCyId,
    requestFocusNode,
    clearPendingFocus,
    setLibraryEpisodeHighlights,
    clearLibraryEpisodeHighlights,
    setGraphEgoFocusCyId,
  }
})
