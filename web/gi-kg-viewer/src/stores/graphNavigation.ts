import { defineStore } from 'pinia'
import { ref } from 'vue'

/** Cross-panel graph focus (e.g. search result → Cytoscape). */
export const useGraphNavigationStore = defineStore('graphNavigation', () => {
  const pendingFocusNodeId = ref<string | null>(null)
  /** When the primary id does not exist in the merged graph, try this (e.g. episode after topic). */
  const pendingFocusFallbackNodeId = ref<string | null>(null)
  /** Raw episode / node ids (e.g. metadata ``episode_id``) → yellow ``search-hit`` ring on graph. */
  const libraryHighlightSourceIds = ref<string[]>([])
  /**
   * Optional extra graph ids (e.g. topic-cluster ``tc:…`` compound) to include in the camera ``center``
   * bbox when applying pending focus; selection stays on the primary node only.
   */
  const pendingFocusCameraIncludeRawIds = ref<string[]>([])
  /**
   * Cytoscape ego (1-hop) center id; kept in sync from ``GraphCanvas`` so the App rail can call
   * ``viewWithEgo`` for graph node detail (same subgraph as the in-canvas panel had).
   */
  const graphEgoFocusCyId = ref<string | null>(null)

  /**
   * Topic cluster compound ids whose **member Topic nodes** are hidden on the Cytoscape canvas
   * (compound box may remain). Toggle from TopicCluster node detail.
   */
  const topicClusterCanvasCollapsedIds = ref<string[]>([])

  function toggleTopicClusterCanvasCollapsed(compoundId: string): void {
    const id = compoundId.trim()
    if (!id) {
      return
    }
    const arr = [...topicClusterCanvasCollapsedIds.value]
    const i = arr.indexOf(id)
    if (i >= 0) {
      arr.splice(i, 1)
    } else {
      arr.push(id)
    }
    topicClusterCanvasCollapsedIds.value = arr
  }

  function isTopicClusterCanvasCollapsed(compoundId: string): boolean {
    return topicClusterCanvasCollapsedIds.value.includes(compoundId.trim())
  }

  function clearTopicClusterCanvasCollapsed(): void {
    topicClusterCanvasCollapsedIds.value = []
  }

  function requestFocusNode(
    nodeId: string,
    fallbackNodeId?: string | null,
    cameraIncludeRawIds?: string[] | null,
  ): void {
    const id = nodeId.trim()
    pendingFocusNodeId.value = id.length ? id : null
    if (fallbackNodeId === undefined) {
      pendingFocusFallbackNodeId.value = null
    } else {
      const fb = typeof fallbackNodeId === 'string' ? fallbackNodeId.trim() : ''
      pendingFocusFallbackNodeId.value = fb.length ? fb : null
    }
    if (cameraIncludeRawIds === undefined) {
      pendingFocusCameraIncludeRawIds.value = []
    } else {
      pendingFocusCameraIncludeRawIds.value = (cameraIncludeRawIds ?? [])
        .map((s) => (typeof s === 'string' ? s.trim() : ''))
        .filter(Boolean)
    }
  }

  function clearPendingFocus(): void {
    pendingFocusNodeId.value = null
    pendingFocusFallbackNodeId.value = null
    pendingFocusCameraIncludeRawIds.value = []
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
    pendingFocusFallbackNodeId,
    libraryHighlightSourceIds,
    pendingFocusCameraIncludeRawIds,
    graphEgoFocusCyId,
    topicClusterCanvasCollapsedIds,
    toggleTopicClusterCanvasCollapsed,
    isTopicClusterCanvasCollapsed,
    clearTopicClusterCanvasCollapsed,
    requestFocusNode,
    clearPendingFocus,
    setLibraryEpisodeHighlights,
    clearLibraryEpisodeHighlights,
    setGraphEgoFocusCyId,
  }
})
