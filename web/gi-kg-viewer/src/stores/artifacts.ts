import { defineStore } from 'pinia'
import {
  artifactCombiners,
  artifactLoadHooks,
  artifactReaders,
  artifactTransforms,
  type ArtifactReader,
} from '../extensions'
import { computed, ref } from 'vue'
import { fetchArtifactJson } from '../api/artifactsApi'
import type { ArtifactData, ParsedArtifact } from '../types/artifact'
import { parseArtifact } from '../utils/parsing'
import { buildDisplayArtifact } from '../utils/mergeGiKg'
import { StaleGeneration } from '../utils/staleGeneration'
import { useGraphExpansionStore } from './graphExpansion'
import { useGraphExplorerStore } from './graphExplorer'
import {
  calendarPublishYmdFromParsedArtifact,
  GRAPH_DEFAULT_EPISODE_CAP,
  selectParsedArtifactsForGraphLoad,
} from '../utils/graphEpisodeSelection'

/** Relative paths selected from /api/artifacts list (basename or full relative path). */
/**
 * A corpus file this build can draw from: a `.kg.json`, or a file a registered reader takes. The
 * corpus lists every artifact kind; offering one nothing reads would load nothing.
 */
export function artifactFileReadable(relPath: string): boolean {
  const lower = relPath.trim().toLowerCase()
  if (lower.endsWith('.kg.json')) return true
  return artifactReaders.list().some((r) => r.suffixes.some((x) => lower.endsWith(x.toLowerCase())))
}

export const useArtifactsStore = defineStore('artifacts', () => {
  const corpusPath = ref('')
  const selectedRelPaths = ref<string[]>([])
  const parsedList = ref<ParsedArtifact[]>([])
  const loadError = ref<string | null>(null)
  const loading = ref(false)
  const loadGate = new StaleGeneration()
  /**
   * When true, skip ``syncMergedGraphFromCorpusApi`` auto-selection (Digest/Library/episode
   * explicit ``loadRelativeArtifacts`` or local file picker owns ``selectedRelPaths``).
   */
  const manualGraphSelection = ref(false)
  /** 
   * Tracks the source of the last artifact load to determine if auto-merge should run.
   * 'digest-external' or 'subject-external' = user clicked from outside graph → NO auto-merge
   * 'graph-internal' or null = in-graph navigation or normal flow → YES auto-merge
   */
  // Reactive ref (not a plain ``let``) — the GraphCanvas meta-watcher reads
  // ``artifacts.currentLoadSource`` (a computed wrapping this) to decide
  // between incremental-append and full-replacement paths. A plain JS
  // variable cannot trigger Vue's reactivity; ``clearLoadSource()`` would
  // mutate without notifying subscribers, and ``currentLoadSource`` would
  // return its stale first-eval value (or never update at all). On the
  // KG-second-wave the watcher then read ``digest-external`` even though
  // the first wave had already cleared it — full-replacement path fired,
  // ``selectedNodeId.value = null``, redraw rebuilt cy without selection,
  // user saw a ~1 s "selection lost + fit-all camera" window before some
  // downstream effect re-applied. Making this a ``ref`` fixes the
  // reactivity, the watcher correctly takes the incremental path on the
  // KG-second-wave, and the prior selection survives the redraw. (See
  // diagnostic trace 2026-05-15.)
  const lastLoadSource = ref<
    'digest-external' | 'subject-external' | 'graph-internal' | null
  >(null)

  const kgArts = computed(() => parsedList.value.filter((p) => p.kind === 'kg'))

  /** The graph to draw: an extension's combiner when one answers, else the viewer's merge. */
  function combinedArtifact(): ParsedArtifact | null {
    for (const c of artifactCombiners.list()) {
      const out = c.combine(parsedList.value)
      if (out !== undefined) return out
    }
    return buildDisplayArtifact(kgArts.value)
  }

  const displayArtifact = computed(() =>
    artifactTransforms
      .list()
      .reduce<ParsedArtifact | null>((art, t) => t.transform(art), combinedArtifact()),
  )

  /** The extension reader for a file name, if one takes its ending. */
  function readerFor(name: string): ArtifactReader | undefined {
    const lower = name.toLowerCase()
    return artifactReaders.list().find((r) => r.suffixes.some((x) => lower.endsWith(x.toLowerCase())))
  }

  /** Offline / no-backend: parse the selected .kg.json files (and files a reader takes) in the browser. */
  async function loadFromLocalFiles(files: FileList | null): Promise<void> {
    if (!files || files.length === 0) return
    manualGraphSelection.value = true
    loadError.value = null
    parsedList.value = []
    selectedRelPaths.value = []
    for (const h of artifactLoadHooks.list()) h.onReset?.('local-files')
    void import('./graphExpansion').then((m) => {
      m.useGraphExpansionStore().resetExpansionState()
    })
    const seq = loadGate.bump()
    loading.value = true
    try {
      const candidates: {
        art: ParsedArtifact
        relKey: string
        publishYmd: string
        fileLastModifiedMs: number
      }[] = []
      for (const file of Array.from(files)) {
        if (loadGate.isStale(seq)) {
          return
        }
        const lower = file.name.toLowerCase()
        const reader = readerFor(file.name)
        if (reader) {
          const read = reader.read(file.name, JSON.parse(await file.text()) as unknown)
          if (read === 'consumed') continue
          if (read) {
            candidates.push({
              art: read,
              relKey: file.name,
              publishYmd: calendarPublishYmdFromParsedArtifact(read, file.lastModified),
              fileLastModifiedMs: file.lastModified,
            })
            continue
          }
        }
        if (!lower.endsWith('.kg.json')) {
          continue
        }
        const text = await file.text()
        const data = JSON.parse(text) as ArtifactData
        const art = parseArtifact(file.name, data)
        const publishYmd = calendarPublishYmdFromParsedArtifact(art, file.lastModified)
        candidates.push({
          art,
          relKey: file.name,
          publishYmd,
          fileLastModifiedMs: file.lastModified,
        })
      }
      if (loadGate.isStale(seq)) {
        return
      }
      if (candidates.length === 0) {
        loadError.value = 'No graph artifact files (.kg.json) in selection.'
        return
      }
      const graphExplorer = useGraphExplorerStore()
      // Local file picks are explicit: do not apply the graph-tab date lens (it defaults to
      // “last N days” and would hide older fixtures / archival episodes the user chose).
      const { kept, wasCapped } = selectParsedArtifactsForGraphLoad(
        candidates,
        '',
        GRAPH_DEFAULT_EPISODE_CAP,
      )
      graphExplorer.setLastAutoLoadCapped(wasCapped)
      parsedList.value = kept
      selectedRelPaths.value = kept.map((p) => p.name)
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
    manualGraphSelection.value = true
    const cleaned = relativePaths.map((p) => p.trim()).filter(Boolean)
    selectedRelPaths.value = cleaned
    await loadSelected()
  }

  /**
   * Fetch and parse the current ``selectedRelPaths`` into ``parsedList``.
   * By default clears graph expansion state (seeds no longer match after a full replace).
   * Pass ``preserveExpansion: true`` when the selection only grew or shrank via
   * ``appendRelativeArtifacts`` / ``removeRelativeArtifacts`` (expand/collapse).
   */
  async function loadSelected(opts?: { preserveExpansion?: boolean }): Promise<void> {
    loadError.value = null
    // #586 fix: do NOT clear parsedList at the start. An intermediate
    // ``parsedList = []`` triggers the GraphCanvas ``filteredArtifact``
    // watcher → ``redraw()`` on an empty graph while the previous
    // Cytoscape layout may still have a pending rAF callback. That rAF
    // fires against a destroyed renderer and throws
    // ``TypeError: can't access property "notify", renderer is null``.
    // Keep the prior list visible until we atomically swap in ``out`` at
    // the end; expand flows (preserveExpansion=true) especially benefit
    // from this — the user sees a smooth add, not a clear-and-rebuild.
    const root = corpusPath.value.trim()
    if (!root || selectedRelPaths.value.length === 0) {
      parsedList.value = []
      loadError.value = 'Set corpus path and select at least one artifact file.'
      useGraphExpansionStore().resetExpansionState()
      return
    }
    if (!opts?.preserveExpansion) {
      useGraphExpansionStore().resetExpansionState()
    }
    for (const h of artifactLoadHooks.list()) h.onLoadStart?.({ preserveExpansion: Boolean(opts?.preserveExpansion) })
    const seq = loadGate.bump()
    loading.value = true
    try {
      // #768 — the selection's files are fetched in parallel (one ``Promise.all``): a
      // rejection fails the load fast; the stale-check fires once after the join, since
      // in-flight fetches cannot be cancelled. A build's load hooks may fetch side files
      // of their own alongside (`onLoadStart`) and apply them before the swap (`beforeAssign`).
      // Tests: ``artifacts.loadSelected.test.ts`` (search for ``#768``).
      const selected = selectedRelPaths.value
      const mainResults = await Promise.all(selected.map((rel) => fetchArtifactJson(root, rel)))

      // F3a — single stale-check after the join. If the operator clicked
      // something else mid-flight, drop all post-join state writes.
      if (loadGate.isStale(seq)) {
        return
      }

      // Apply results in original selection order so ``parsedList`` is deterministic.
      const out: ParsedArtifact[] = []
      for (let i = 0; i < selected.length; i++) {
        const rel = selected[i]!
        const data = mainResults[i]!
        const lower = rel.toLowerCase()
        const base = rel.includes('/') ? rel.split('/').pop() || rel : rel
        const reader = readerFor(rel)
        if (reader) {
          const read = reader.read(base, data, rel)
          if (read === 'consumed') continue
          if (read) {
            out.push(read)
            continue
          }
        }
        if (!lower.endsWith('.kg.json')) continue
        out.push(parseArtifact(base, data, rel))
      }

      if (out.length === 0) {
        // Only clear on definite "nothing to show" — matches pre-#586
        // behaviour for this branch.
        parsedList.value = []
        loadError.value = 'No graph artifact files (.kg.json) in selection.'
        return
      }

      for (const h of artifactLoadHooks.list()) {
        await h.beforeAssign?.({ root, selected: [...selected], parsed: out })
      }
      if (loadGate.isStale(seq)) {
        return
      }
      parsedList.value = out
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
    const previous = corpusPath.value
    corpusPath.value = String(p)
    if (corpusPath.value.trim() !== previous.trim()) {
      for (const h of artifactLoadHooks.list()) h.onCorpusPathChange?.(corpusPath.value, previous)
    }
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
    manualGraphSelection.value = false
    selectedRelPaths.value = []
    parsedList.value = []
    loadError.value = null
    for (const h of artifactLoadHooks.list()) h.onReset?.('clear')
    void import('./graphExpansion').then((m) => {
      m.useGraphExpansionStore().resetExpansionState()
    })
  }

  /** Check every path from the current corpus list (does not fetch). */
  function selectAllListed(relativePaths: string[]): void {
    selectedRelPaths.value = relativePaths.slice()
  }

  function clearManualGraphSelection(): void {
    manualGraphSelection.value = false
  }

  /** The operator chose what the graph shows (e.g. a show's "Open in graph"): corpus syncs keep it. */
  function markManualGraphSelection(): void {
    manualGraphSelection.value = true
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
    await loadSelected({ preserveExpansion: true })
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
    await loadSelected({ preserveExpansion: true })
  }

  /**
   * Set the source of the current artifact load to control auto-merge behavior.
   * External loads (from Digest/Library) suppress auto-merge for focused views.
   */
  function setLoadSource(source: 'digest-external' | 'subject-external' | 'graph-internal' | null): void {
    lastLoadSource.value = source
  }

  function clearLoadSource(): void {
    lastLoadSource.value = null
  }

  /**
   * Getter for the current load source (readonly).
   * Used by GraphCanvas to determine if incremental appends should skip disruptive ops.
   */
  const currentLoadSource = computed<'digest-external' | 'subject-external' | 'graph-internal' | null>(
    () => lastLoadSource.value,
  )

  return {
    corpusPath,
    manualGraphSelection,
    selectedRelPaths,
    parsedList,
    loadError,
    loading,
    kgArts,
    displayArtifact,
    loadSelected,
    loadRelativeArtifacts,
    loadFromLocalFiles,
    setCorpusPath,
    toggleSelection,
    clearSelection,
    clearManualGraphSelection,
    markManualGraphSelection,
    selectAllListed,
    deselectAllListed,
    appendRelativeArtifacts,
    removeRelativeArtifacts,
    setLoadSource,
    clearLoadSource,
    currentLoadSource,  // Readonly getter for lastLoadSource
  }
})
