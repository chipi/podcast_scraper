import { defineStore } from 'pinia'
import { ref } from 'vue'

/** Cross-panel graph focus (e.g. search result → Cytoscape). */
export const useGraphNavigationStore = defineStore('graphNavigation', () => {
  const pendingFocusNodeId = ref<string | null>(null)

  function requestFocusNode(nodeId: string): void {
    const id = nodeId.trim()
    pendingFocusNodeId.value = id.length ? id : null
  }

  function clearPendingFocus(): void {
    pendingFocusNodeId.value = null
  }

  return {
    pendingFocusNodeId,
    requestFocusNode,
    clearPendingFocus,
  }
})
