import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

export const useShellStore = defineStore('shell', () => {
  const corpusPath = ref(
    (import.meta.env.VITE_DEFAULT_CORPUS_PATH as string | undefined) ?? '',
  )
  const healthStatus = ref<string | null>(null)
  const healthError = ref<string | null>(null)
  const artifactsLoading = ref(false)
  const artifactsError = ref<string | null>(null)
  const artifactCount = ref<number | null>(null)
  const artifactList = ref<
    { name: string; relative_path: string; kind: string; size_bytes: number }[]
  >([])
  /** Server-resolved absolute corpus path (returned by /api/artifacts). */
  const resolvedCorpusPath = ref<string | null>(null)

  const hasCorpusPath = computed(() => corpusPath.value.trim().length > 0)

  async function fetchHealth(): Promise<void> {
    healthError.value = null
    try {
      const res = await fetch('/api/health')
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`)
      }
      const body = (await res.json()) as { status?: string }
      healthStatus.value = body.status ?? 'unknown'
    } catch (e) {
      healthStatus.value = null
      healthError.value = e instanceof Error ? e.message : String(e)
    }
  }

  async function fetchArtifactList(): Promise<void> {
    artifactsError.value = null
    artifactCount.value = null
    artifactList.value = []
    resolvedCorpusPath.value = null
    if (!hasCorpusPath.value) {
      artifactsError.value = 'Set a corpus directory path (local output folder).'
      return
    }
    artifactsLoading.value = true
    try {
      const q = new URLSearchParams({ path: corpusPath.value.trim() })
      const res = await fetch(`/api/artifacts?${q.toString()}`)
      if (!res.ok) {
        const detail = await res.text()
        throw new Error(detail || `HTTP ${res.status}`)
      }
      const body = (await res.json()) as {
        path?: string
        artifacts?: {
          name: string
          relative_path: string
          kind: string
          size_bytes: number
        }[]
      }
      const list = Array.isArray(body.artifacts) ? body.artifacts : []
      artifactList.value = list
      artifactCount.value = list.length
      if (typeof body.path === 'string' && body.path.trim()) {
        resolvedCorpusPath.value = body.path.trim()
      }
    } catch (e) {
      artifactsError.value = e instanceof Error ? e.message : String(e)
    } finally {
      artifactsLoading.value = false
    }
  }

  return {
    corpusPath,
    healthStatus,
    healthError,
    artifactsLoading,
    artifactsError,
    artifactCount,
    artifactList,
    resolvedCorpusPath,
    hasCorpusPath,
    fetchHealth,
    fetchArtifactList,
  }
})
