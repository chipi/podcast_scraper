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
   * #6 L0 — the navigation "breadcrumb trail": node ids appended to the current graph view as you
   * navigate the detail rail *inside the graph* (graph-rail-scoped; see ``NodeDetail``). The ego
   * origin is pinned implicitly (it's always in ``viewWithEgo``); this list holds only the extra
   * navigated-to nodes, LRU-ordered (oldest first) and capped so it can never grow unbounded.
   * Default-empty → ``viewWithEgo`` behaves exactly as before until navigation populates it, and
   * it resets whenever the ego origin changes (a new subject / re-centre → a fresh trail).
   */
  const TRAIL_BUDGET = 28
  const trailNodeIds = ref<string[]>([])

  /** Append (or LRU-touch) a node onto the trail, pruning the oldest beyond the budget. */
  function addToTrail(nodeId: string): void {
    const id = nodeId.trim()
    if (!id || id === graphEgoFocusCyId.value) {
      return // blank, or the pinned origin (already always in view)
    }
    const arr = trailNodeIds.value.filter((x) => x !== id) // move-to-newest if already present
    arr.push(id)
    while (arr.length > TRAIL_BUDGET) {
      arr.shift() // prune oldest
    }
    trailNodeIds.value = arr
  }

  function clearTrail(): void {
    trailNodeIds.value = []
  }

  /** Replace the trail wholesale (replay reconstructs a session's trail at a given step). */
  function setTrail(ids: string[]): void {
    const cleaned = ids.map((s) => s.trim()).filter(Boolean)
    trailNodeIds.value = [...new Set(cleaned)].slice(-TRAIL_BUDGET)
  }

  /**
   * When true, signals GraphCanvas to fit the viewport to visible content after the next layout,
   * instead of preserving the current viewport. Used when loading from external sources without
   * a specific focus target (e.g., Digest category bands without clusters).
   */
  const requestFitAfterLoad = ref(false)

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
    // Clear first so repeat requests for the same id still notify watchers (Show on graph twice).
    clearPendingFocus()
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
    const next = id?.trim() ? id.trim() : null
    if (next !== graphEgoFocusCyId.value) {
      trailNodeIds.value = [] // new ego origin → start a fresh breadcrumb trail
    }
    graphEgoFocusCyId.value = next
  }

  function setRequestFitAfterLoad(): void {
    requestFitAfterLoad.value = true
  }

  function clearRequestFitAfterLoad(): void {
    requestFitAfterLoad.value = false
  }

  return {
    pendingFocusNodeId,
    pendingFocusFallbackNodeId,
    libraryHighlightSourceIds,
    pendingFocusCameraIncludeRawIds,
    graphEgoFocusCyId,
    requestFitAfterLoad,
    topicClusterCanvasCollapsedIds,
    trailNodeIds,
    addToTrail,
    clearTrail,
    setTrail,
    toggleTopicClusterCanvasCollapsed,
    isTopicClusterCanvasCollapsed,
    clearTopicClusterCanvasCollapsed,
    requestFocusNode,
    clearPendingFocus,
    setLibraryEpisodeHighlights,
    clearLibraryEpisodeHighlights,
    setGraphEgoFocusCyId,
    setRequestFitAfterLoad,
    clearRequestFitAfterLoad,
  }
})
