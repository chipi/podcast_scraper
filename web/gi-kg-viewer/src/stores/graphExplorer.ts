import { defineStore } from 'pinia'
import { ref } from 'vue'

export type GraphLayoutName = 'cose' | 'breadthfirst' | 'circle' | 'grid'

export const useGraphExplorerStore = defineStore('graphExplorer', () => {
  const preferredLayout = ref<GraphLayoutName>('cose')
  const minimapOpen = ref(false)
  /** Degree histogram bucket id or null = no filter. */
  const activeDegreeBucket = ref<string | null>(null)

  function clearDegreeBucket(): void {
    activeDegreeBucket.value = null
  }

  function toggleDegreeBucket(key: string): void {
    activeDegreeBucket.value = activeDegreeBucket.value === key ? null : key
  }

  function resetForNewArtifact(): void {
    activeDegreeBucket.value = null
  }

  return {
    preferredLayout,
    minimapOpen,
    activeDegreeBucket,
    clearDegreeBucket,
    toggleDegreeBucket,
    resetForNewArtifact,
  }
})
