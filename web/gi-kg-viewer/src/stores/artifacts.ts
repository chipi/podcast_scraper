import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { fetchArtifactJson } from '../api/artifactsApi'
import type { BridgeDocument } from '../types/bridge'
import type { ArtifactData, ParsedArtifact } from '../types/artifact'
import { parseBridgeDocument } from '../utils/bridgeDocument'
import { parseArtifact } from '../utils/parsing'
import { buildDisplayArtifact } from '../utils/mergeGiKg'

/** Relative paths selected from /api/artifacts list (basename or full relative path). */
export const useArtifactsStore = defineStore('artifacts', () => {
  const corpusPath = ref('')
  const selectedRelPaths = ref<string[]>([])
  const parsedList = ref<ParsedArtifact[]>([])
  /** RFC-072 bridge.json for the current corpus selection (optional). */
  const bridgeDocument = ref<BridgeDocument | null>(null)
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
    bridgeDocument.value = null
    selectedRelPaths.value = []
    loading.value = true
    try {
      const out: ParsedArtifact[] = []
      for (const file of Array.from(files)) {
        const lower = file.name.toLowerCase()
        if (lower.endsWith('.bridge.json')) {
          try {
            const text = await file.text()
            bridgeDocument.value = parseBridgeDocument(JSON.parse(text))
          } catch {
            /* ignore invalid bridge */
          }
          continue
        }
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

  /**
   * Load GI/KG artifacts by corpus-relative paths (e.g. from Corpus Library).
   * Replaces current selection and fetches from the API.
   */
  async function loadRelativeArtifacts(relativePaths: string[]): Promise<void> {
    const cleaned = relativePaths.map((p) => p.trim()).filter(Boolean)
    selectedRelPaths.value = cleaned
    await loadSelected()
  }

  async function loadSelected(): Promise<void> {
    loadError.value = null
    parsedList.value = []
    bridgeDocument.value = null
    const root = corpusPath.value.trim()
    if (!root || selectedRelPaths.value.length === 0) {
      loadError.value = 'Set corpus path and select at least one artifact file.'
      return
    }
    loading.value = true
    try {
      const out: ParsedArtifact[] = []
      for (const rel of selectedRelPaths.value) {
        const lower = rel.toLowerCase()
        if (lower.endsWith('.bridge.json')) {
          try {
            const data = await fetchArtifactJson(root, rel)
            bridgeDocument.value = parseBridgeDocument(data)
          } catch {
            /* optional */
          }
          continue
        }
        const data = await fetchArtifactJson(root, rel)
        const base = rel.includes('/') ? rel.split('/').pop() || rel : rel
        out.push(parseArtifact(base, data))
      }
      if (out.length === 0) {
        loadError.value = 'No .gi.json or .kg.json files in selection.'
        return
      }
      parsedList.value = out
      if (!bridgeDocument.value) {
        const giRel = selectedRelPaths.value.find((p) => p.toLowerCase().endsWith('.gi.json'))
        if (giRel) {
          const br = giRel.replace(/\.gi\.json$/i, '.bridge.json')
          if (br !== giRel) {
            try {
              const data = await fetchArtifactJson(root, br)
              bridgeDocument.value = parseBridgeDocument(data)
            } catch {
              /* sibling bridge is optional */
            }
          }
        }
      }
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
    bridgeDocument.value = null
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
    bridgeDocument,
    loadError,
    loading,
    giArts,
    kgArts,
    displayArtifact,
    loadSelected,
    loadRelativeArtifacts,
    loadFromLocalFiles,
    setCorpusPath,
    toggleSelection,
    clearSelection,
    selectAllListed,
    deselectAllListed,
  }
})
