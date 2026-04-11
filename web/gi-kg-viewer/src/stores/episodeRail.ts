import { defineStore } from 'pinia'
import { ref } from 'vue'

export type EpisodeRailPaneKind = 'tools' | 'episode' | 'graph-node'

/**
 * Right sidebar: Search/Explore, Library episode detail, or graph node detail (unified `w-80`).
 * Only one mode is visible at a time.
 */
export const useEpisodeRailStore = defineStore('episodeRail', () => {
  const paneKind = ref<EpisodeRailPaneKind>('tools')
  const toolsTab = ref<'search' | 'explore'>('search')
  const metadataRelativePath = ref<string | null>(null)
  /** Cytoscape node id when ``paneKind === 'graph-node'`` (``NodeDetail`` in App rail). */
  const graphNodeCyId = ref<string | null>(null)
  /**
   * When ``paneKind === 'episode'``, optional Cytoscape id for the **Connections** strip
   * (same graph-neighbor list as non-episode graph nodes).
   */
  const graphConnectionsCyId = ref<string | null>(null)

  function openEpisodePanel(
    metadataPath: string,
    opts?: { graphConnectionsCyId?: string | null },
  ): void {
    const t = metadataPath.trim()
    graphNodeCyId.value = null
    metadataRelativePath.value = t || null
    const cy = opts?.graphConnectionsCyId?.trim()
    graphConnectionsCyId.value = cy || null
    if (t) {
      paneKind.value = 'episode'
    }
  }

  function openGraphNodePanel(cyNodeId: string): void {
    const t = cyNodeId.trim()
    graphNodeCyId.value = t || null
    graphConnectionsCyId.value = null
    metadataRelativePath.value = null
    if (t) {
      paneKind.value = 'graph-node'
    }
  }

  /**
   * Switch right rail to Search / Explore.
   * By default clears stashed graph node id (canvas clear, filter reset, leaving graph tab).
   * From **Graph node** rail, pass ``preserveGraphNodeId: true`` so **Back to details** can restore.
   */
  function showTools(opts?: { preserveGraphNodeId?: boolean }): void {
    paneKind.value = 'tools'
    if (!opts?.preserveGraphNodeId) {
      graphNodeCyId.value = null
    }
  }

  /**
   * After Search & Explore: restore graph node detail if a node id is stashed, else episode rail.
   */
  function resumeDetailPanel(): void {
    if (graphNodeCyId.value?.trim()) {
      paneKind.value = 'graph-node'
      return
    }
    if (metadataRelativePath.value?.trim()) {
      paneKind.value = 'episode'
    }
  }

  /** Clear selection and return to tools (e.g. Library filters reset). */
  function clearEpisodeContext(): void {
    metadataRelativePath.value = null
    graphNodeCyId.value = null
    graphConnectionsCyId.value = null
    paneKind.value = 'tools'
  }

  return {
    paneKind,
    toolsTab,
    metadataRelativePath,
    graphNodeCyId,
    graphConnectionsCyId,
    openEpisodePanel,
    openGraphNodePanel,
    showTools,
    resumeDetailPanel,
    clearEpisodeContext,
  }
})
