import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { fetchArtifactJson } from '../api/artifactsApi'
import type { ArtifactData, ParsedArtifact } from '../types/artifact'
import { parseArtifact } from '../utils/parsing'
import { buildDisplayArtifact } from '../utils/mergeGiKg'

/** Relative paths selected from /api/artifacts list (basename or full relative path). */
export const useArtifactsStore = defineStore('artifacts', () => {
  const corpusPath = ref('')
  const selectedRelPaths = ref<string[]>([])
  const parsedList = ref<ParsedArtifact[]>([])
  const loadError = ref<string | null>(null)
  const loading = ref(false)

  const giArts = computed(() => parsedList.value.filter((p) => p.kind === 'gi'))
  const kgArts = computed(() => parsedList.value.filter((p) => p.kind === 'kg'))

  const displayArtifact = computed(() =>
    buildDisplayArtifact(giArts.value, kgArts.value),
  )

  /** Offline / no-backend: parse selected .gi.json / .kg.json files in the browser. */
  async function loadFromLocalFiles(files: FileList | null): Promise<void> {
    if (!files || files.length === 0) return
    loadError.value = null
    parsedList.value = []
    selectedRelPaths.value = []
    loading.value = true
    try {
      const out: ParsedArtifact[] = []
      for (const file of Array.from(files)) {
        const lower = file.name.toLowerCase()
        if (!lower.endsWith('.gi.json') && !lower.endsWith('.kg.json')) {
          continue
        }
        const text = await file.text()
        const data = JSON.parse(text) as ArtifactData
        out.push(parseArtifact(file.name, data))
      }
      if (out.length === 0) {
        loadError.value = 'No .gi.json or .kg.json files in selection.'
        return
      }
      parsedList.value = out
      selectedRelPaths.value = out.map((p) => p.name)
    } catch (e) {
      loadError.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  async function loadSelected(): Promise<void> {
    loadError.value = null
    parsedList.value = []
    const root = corpusPath.value.trim()
    if (!root || selectedRelPaths.value.length === 0) {
      loadError.value = 'Set corpus path and select at least one artifact file.'
      return
    }
    loading.value = true
    try {
      const out: ParsedArtifact[] = []
      for (const rel of selectedRelPaths.value) {
        const data = await fetchArtifactJson(root, rel)
        const base = rel.includes('/') ? rel.split('/').pop() || rel : rel
        out.push(parseArtifact(base, data))
      }
      parsedList.value = out
    } catch (e) {
      loadError.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  function setCorpusPath(p: string): void {
    corpusPath.value = p
  }

  function toggleSelection(rel: string): void {
    const i = selectedRelPaths.value.indexOf(rel)
    if (i >= 0) {
      selectedRelPaths.value = selectedRelPaths.value.filter((_, j) => j !== i)
    } else {
      selectedRelPaths.value = [...selectedRelPaths.value, rel]
    }
  }

  function clearSelection(): void {
    selectedRelPaths.value = []
    parsedList.value = []
    loadError.value = null
  }

  /** Check every path from the current corpus list (does not fetch). */
  function selectAllListed(relativePaths: string[]): void {
    selectedRelPaths.value = relativePaths.slice()
  }

  /** Uncheck all listed files (does not clear the graph until you load again). */
  function deselectAllListed(): void {
    selectedRelPaths.value = []
  }

  return {
    corpusPath,
    selectedRelPaths,
    parsedList,
    loadError,
    loading,
    giArts,
    kgArts,
    displayArtifact,
    loadSelected,
    loadFromLocalFiles,
    setCorpusPath,
    toggleSelection,
    clearSelection,
    selectAllListed,
    deselectAllListed,
  }
})
