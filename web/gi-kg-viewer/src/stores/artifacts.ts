import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { fetchArtifactJson } from '../api/artifactsApi'
import type { TopicClustersDocument, TopicClustersFetchResult } from '../api/corpusTopicClustersApi'
import { fetchTopicClustersFromApi } from '../api/corpusTopicClustersApi'
import type { BridgeDocument } from '../types/bridge'
import type { ArtifactData, ParsedArtifact } from '../types/artifact'
import { parseBridgeDocument } from '../utils/bridgeDocument'
import { parseArtifact } from '../utils/parsing'
import { fetchResolveEpisodeArtifacts } from '../api/corpusLibraryApi'
import { buildDisplayArtifact } from '../utils/mergeGiKg'
import {
  artifactRelPathsForResolvedRow,
  clusterSiblingEpisodeCap,
  clusterSiblingEpisodeIdCandidates,
  episodeIdsFromParsedArtifacts,
  sortResolvedArtifactsNewestFirst,
} from '../utils/clusterSiblingMerge'
import { withTopicClustersOnDisplay } from '../utils/topicClustersOverlay'
import { StaleGeneration } from '../utils/staleGeneration'

/** Relative paths selected from /api/artifacts list (basename or full relative path). */
export const useArtifactsStore = defineStore('artifacts', () => {
  const corpusPath = ref('')
  const selectedRelPaths = ref<string[]>([])
  const parsedList = ref<ParsedArtifact[]>([])
  /** RFC-072 bridge.json for the current corpus selection (optional). */
  const bridgeDocument = ref<BridgeDocument | null>(null)
  /** RFC-075 ``topic_clusters.json`` from the API when present (API load only). */
  const topicClustersDoc = ref<TopicClustersDocument | null>(null)
  /**
   * How the last load obtained topic clusters: API success, 404, error, local file picker (no API JSON),
   * or idle (cleared / not yet loaded).
   */
  const topicClustersLoadState = ref<
    'idle' | 'ok' | 'missing' | 'error' | 'local_files'
  >('idle')
  const topicClustersErrorDetail = ref<string | null>(null)
  /** Soft schema warning (unknown ``schema_version``); non-blocking. */
  const topicClustersSchemaWarning = ref<string | null>(null)
  const loadError = ref<string | null>(null)
  const loading = ref(false)
  const loadGate = new StaleGeneration()
  /** Inline status after cluster sibling auto-merge (graph tab); errors use ``siblingMergeError`` + banner. */
  const siblingMergeLine = ref<string | null>(null)
  /** True when ``siblingMergeLine`` is from a failed resolve/load (shown as a top alert). */
  const siblingMergeError = ref(false)
  let siblingMergeInFlight: Promise<void> | null = null

  const giArts = computed(() => parsedList.value.filter((p) => p.kind === 'gi'))
  const kgArts = computed(() => parsedList.value.filter((p) => p.kind === 'kg'))

  const displayArtifact = computed(() =>
    withTopicClustersOnDisplay(
      buildDisplayArtifact(giArts.value, kgArts.value),
      topicClustersDoc.value,
    ),
  )

  function applyTopicClustersFetchResult(tc: TopicClustersFetchResult): void {
    if (tc.status === 'ok') {
      topicClustersDoc.value = tc.document
      topicClustersLoadState.value = 'ok'
      topicClustersErrorDetail.value = null
      topicClustersSchemaWarning.value = tc.schemaWarning ?? null
    } else if (tc.status === 'missing') {
      topicClustersDoc.value = null
      topicClustersLoadState.value = 'missing'
      topicClustersErrorDetail.value = null
      topicClustersSchemaWarning.value = null
    } else {
      topicClustersDoc.value = null
      topicClustersLoadState.value = 'error'
      topicClustersErrorDetail.value = tc.message
      topicClustersSchemaWarning.value = null
    }
  }

  /**
   * Fetch ``/api/corpus/topic-clusters`` for the current ``corpusPath`` (no artifact load required).
   * Safe to call as soon as corpus root + healthy API are known so **API · Data** can show status.
   */
  async function syncTopicClustersForCurrentCorpus(): Promise<void> {
    const root = corpusPath.value.trim()
    if (!root) {
      return
    }
    try {
      const tc = await fetchTopicClustersFromApi(root)
      applyTopicClustersFetchResult(tc)
    } catch (e) {
      topicClustersDoc.value = null
      topicClustersLoadState.value = 'error'
      topicClustersErrorDetail.value = e instanceof Error ? e.message : String(e)
      topicClustersSchemaWarning.value = null
    }
  }

  /** Offline / no-backend: parse selected .gi.json / .kg.json files in the browser. */
  async function loadFromLocalFiles(files: FileList | null): Promise<void> {
    if (!files || files.length === 0) return
    loadError.value = null
    parsedList.value = []
    bridgeDocument.value = null
    topicClustersDoc.value = null
    topicClustersLoadState.value = 'idle'
    topicClustersErrorDetail.value = null
    topicClustersSchemaWarning.value = null
    selectedRelPaths.value = []
    const seq = loadGate.bump()
    loading.value = true
    try {
      const out: ParsedArtifact[] = []
      for (const file of Array.from(files)) {
        if (loadGate.isStale(seq)) {
          return
        }
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
      if (loadGate.isStale(seq)) {
        return
      }
      if (out.length === 0) {
        loadError.value = 'No .gi.json or .kg.json files in selection.'
        return
      }
      parsedList.value = out
      selectedRelPaths.value = out.map((p) => p.name)
      topicClustersDoc.value = null
      topicClustersLoadState.value = 'local_files'
      topicClustersErrorDetail.value = null
      topicClustersSchemaWarning.value = null
    } catch (e) {
      if (loadGate.isStale(seq)) {
        return
      }
      loadError.value = e instanceof Error ? e.message : String(e)
    } finally {
      if (loadGate.isCurrent(seq)) {
        loading.value = false
      }
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
    const seq = loadGate.bump()
    loading.value = true
    try {
      const out: ParsedArtifact[] = []
      for (const rel of selectedRelPaths.value) {
        if (loadGate.isStale(seq)) {
          return
        }
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
        out.push(parseArtifact(base, data, rel))
      }
      if (loadGate.isStale(seq)) {
        return
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
              if (loadGate.isStale(seq)) {
                return
              }
              const data = await fetchArtifactJson(root, br)
              if (loadGate.isStale(seq)) {
                return
              }
              bridgeDocument.value = parseBridgeDocument(data)
            } catch {
              /* sibling bridge is optional */
            }
          }
        }
      }
      if (loadGate.isStale(seq)) {
        return
      }
      await syncTopicClustersForCurrentCorpus()
    } catch (e) {
      if (loadGate.isStale(seq)) {
        return
      }
      loadError.value = e instanceof Error ? e.message : String(e)
    } finally {
      if (loadGate.isCurrent(seq)) {
        loading.value = false
      }
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
    topicClustersDoc.value = null
    topicClustersLoadState.value = 'idle'
    topicClustersErrorDetail.value = null
    topicClustersSchemaWarning.value = null
    loadError.value = null
    void import('./graphExpansion').then((m) => {
      m.useGraphExpansionStore().resetExpansionState()
    })
  }

  /** Check every path from the current corpus list (does not fetch). */
  function selectAllListed(relativePaths: string[]): void {
    selectedRelPaths.value = relativePaths.slice()
  }

  /** Uncheck all listed files (does not clear the graph until you load again). */
  function deselectAllListed(): void {
    selectedRelPaths.value = []
    void import('./graphExpansion').then((m) => {
      m.useGraphExpansionStore().resetExpansionState()
    })
  }

  async function appendRelativeArtifacts(extraRelPaths: string[]): Promise<void> {
    const root = corpusPath.value.trim()
    if (!root || extraRelPaths.length === 0) {
      return
    }
    const have = new Set(selectedRelPaths.value.map((p) => p.replace(/\\/g, '/')))
    const add: string[] = []
    for (const raw of extraRelPaths) {
      const p = raw.trim().replace(/\\/g, '/')
      if (!p || have.has(p)) {
        continue
      }
      have.add(p)
      add.push(p)
    }
    if (add.length === 0) {
      return
    }
    selectedRelPaths.value = [...selectedRelPaths.value, ...add]
    await loadSelected()
  }

  /** Remove corpus-relative paths from the current selection and reload the graph. */
  async function removeRelativeArtifacts(relpaths: string[]): Promise<void> {
    const root = corpusPath.value.trim()
    if (!root || relpaths.length === 0) {
      return
    }
    const drop = new Set(relpaths.map((p) => p.trim().replace(/\\/g, '/')).filter(Boolean))
    if (drop.size === 0) {
      return
    }
    const next = selectedRelPaths.value.filter((p) => !drop.has(p.trim().replace(/\\/g, '/')))
    if (next.length === selectedRelPaths.value.length) {
      return
    }
    selectedRelPaths.value = next
    await loadSelected()
  }

  /**
   * After a successful graph artifact load: pull in sibling episodes from ``topic_clusters.json``
   * (catalog-backed), newest first, up to cap. No-op off-graph or without topic clusters.
   */
  function clearSiblingMergeBanner(): void {
    siblingMergeLine.value = null
    siblingMergeError.value = false
  }

  async function maybeMergeClusterSiblingEpisodes(graphTabActive: boolean): Promise<void> {
    siblingMergeLine.value = null
    siblingMergeError.value = false
    if (!graphTabActive) {
      return
    }
    const root = corpusPath.value.trim()
    if (!root || !topicClustersDoc.value) {
      return
    }
    const cap = clusterSiblingEpisodeCap()
    if (cap === 0) {
      return
    }
    if (siblingMergeInFlight) {
      await siblingMergeInFlight
      return
    }

    const loadedIds = episodeIdsFromParsedArtifacts(parsedList.value)
    const { candidateIds, mTotal } = clusterSiblingEpisodeIdCandidates(
      topicClustersDoc.value,
      loadedIds,
    )
    if (candidateIds.length === 0) {
      return
    }

    const run = async (): Promise<void> => {
      try {
        const res = await fetchResolveEpisodeArtifacts(root, candidateIds)
        const sorted = sortResolvedArtifactsNewestFirst(res.resolved)
        const selected = new Set(selectedRelPaths.value.map((p) => p.replace(/\\/g, '/')))
        const pathsToAdd: string[] = []
        let addedEpisodes = 0
        for (const row of sorted) {
          if (addedEpisodes >= cap) {
            break
          }
          const rels = artifactRelPathsForResolvedRow(row)
          if (rels.length === 0) {
            continue
          }
          let anyNew = false
          for (const rel of rels) {
            const norm = rel.replace(/\\/g, '/')
            if (!selected.has(norm)) {
              anyNew = true
              break
            }
          }
          if (!anyNew) {
            continue
          }
          for (const rel of rels) {
            const norm = rel.replace(/\\/g, '/')
            if (!selected.has(norm)) {
              selected.add(norm)
              pathsToAdd.push(rel)
            }
          }
          addedEpisodes += 1
        }
        const z = res.missing_episode_ids.length
        siblingMergeError.value = false
        const miss = z > 0 ? ` · ${z} miss${z === 1 ? '' : 'es'}` : ''
        siblingMergeLine.value =
          `+${addedEpisodes} new · ${mTotal} in cluster · cap ${cap}${miss}`
        if (pathsToAdd.length > 0) {
          await appendRelativeArtifacts(pathsToAdd)
        }
      } catch (e) {
        siblingMergeError.value = true
        siblingMergeLine.value = e instanceof Error ? e.message : String(e)
      }
    }

    siblingMergeInFlight = run().finally(() => {
      siblingMergeInFlight = null
    })
    await siblingMergeInFlight
  }

  return {
    corpusPath,
    selectedRelPaths,
    parsedList,
    bridgeDocument,
    topicClustersDoc,
    topicClustersLoadState,
    topicClustersErrorDetail,
    topicClustersSchemaWarning,
    loadError,
    loading,
    siblingMergeLine,
    siblingMergeError,
    clearSiblingMergeBanner,
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
    syncTopicClustersForCurrentCorpus,
    appendRelativeArtifacts,
    removeRelativeArtifacts,
    maybeMergeClusterSiblingEpisodes,
  }
})
