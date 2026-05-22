import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { fetchArtifactJson } from '../api/artifactsApi'
import type { TopicClustersDocument, TopicClustersFetchResult } from '../api/corpusTopicClustersApi'
import { fetchTopicClustersFromApi } from '../api/corpusTopicClustersApi'
import type { BridgeDocument } from '../types/bridge'
import type { ArtifactData, ParsedArtifact } from '../types/artifact'
import { parseBridgeDocument } from '../utils/bridgeDocument'
import { findRawNodeInArtifact, parseArtifact } from '../utils/parsing'
import { fetchResolveEpisodeArtifacts } from '../api/corpusLibraryApi'
import { buildDisplayArtifact } from '../utils/mergeGiKg'
import {
  allEpisodeIdsListedForCluster,
  artifactRelPathsForResolvedRow,
  clusterSiblingEpisodeCap,
  clusterSiblingEpisodeIdCandidates,
  episodeIdsFromParsedArtifacts,
  sortResolvedArtifactsNewestFirst,
} from '../utils/clusterSiblingMerge'
import {
  findClusterByCompoundId,
  withTopicClustersOnDisplay,
} from '../utils/topicClustersOverlay'
import { StaleGeneration } from '../utils/staleGeneration'
import { useGraphExpansionStore } from './graphExpansion'
import { useGraphExplorerStore } from './graphExplorer'
import {
  calendarPublishYmdFromParsedArtifact,
  episodeStemFromArtifactRelPath,
  GRAPH_DEFAULT_EPISODE_CAP,
  selectParsedArtifactsForGraphLoad,
} from '../utils/graphEpisodeSelection'

/** Relative paths selected from /api/artifacts list (basename or full relative path). */
export const useArtifactsStore = defineStore('artifacts', () => {
  const corpusPath = ref('')
  const selectedRelPaths = ref<string[]>([])
  const parsedList = ref<ParsedArtifact[]>([])
  /** bridge.json for the current corpus selection (optional). */
  const bridgeDocument = ref<BridgeDocument | null>(null)
  /** ``topic_clusters.json`` from the API when present (API load only). */
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
  /**
   * When true, skip ``syncMergedGraphFromCorpusApi`` auto-selection (Digest/Library/episode
   * explicit ``loadRelativeArtifacts`` or local file picker owns ``selectedRelPaths``).
   */
  const manualGraphSelection = ref(false)
  /** Inline status after cluster sibling auto-merge (graph tab); errors use ``siblingMergeError`` + banner. */
  const siblingMergeLine = ref<string | null>(null)
  /** True when ``siblingMergeLine`` is from a failed resolve/load (shown as a top alert). */
  const siblingMergeError = ref(false)
  let siblingMergeInFlight: Promise<void> | null = null
  /** Tracks selected paths snapshot when auto-merge last ran; prevents re-running on tab switch. */
  let lastAutoMergeSnapshotKey = ''
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

  const giArts = computed(() => parsedList.value.filter((p) => p.kind === 'gi'))
  const kgArts = computed(() => parsedList.value.filter((p) => p.kind === 'kg'))

  const displayArtifact = computed(() =>
    withTopicClustersOnDisplay(
      buildDisplayArtifact(giArts.value, kgArts.value),
      topicClustersDoc.value,
    ),
  )

  // #769 — memoize the topic-clusters HTTP fetch per corpus root.
  //
  // ``ensureTopicClusterCompoundVisible`` calls ``syncTopicClustersForCurrentCorpus``
  // at the top of every invocation, and on first-open graph paths the
  // App.vue ``activateGraphTab`` orchestrator can invoke
  // ``ensureTopicClusterCompoundVisible`` up to 3 times per click (bootstrap
  // path + post-bootstrap call + watcher cascade). The HTTP fetch is
  // identical each time. Memoizing it via this sentinel saves 100-400 ms
  // per redundant call.
  //
  // Invalidation contract: ``topicClustersFetchedForRoot`` is set ONLY by
  // ``applyTopicClustersFetchResult`` on a successful 'ok' result, and
  // cleared at every site that nulls ``topicClustersDoc.value``. The
  // sentinel + doc form the cache key; the memoize path is taken only
  // when BOTH the root matches AND the doc is still populated.
  let topicClustersFetchedForRoot: string | null = null

  function applyTopicClustersFetchResult(tc: TopicClustersFetchResult, root: string): void {
    if (tc.status === 'ok') {
      topicClustersDoc.value = tc.document
      topicClustersFetchedForRoot = root
      topicClustersLoadState.value = 'ok'
      topicClustersErrorDetail.value = null
      topicClustersSchemaWarning.value = tc.schemaWarning ?? null
    } else if (tc.status === 'missing') {
      topicClustersDoc.value = null
      topicClustersFetchedForRoot = null
      topicClustersLoadState.value = 'missing'
      topicClustersErrorDetail.value = null
      topicClustersSchemaWarning.value = null
    } else {
      topicClustersDoc.value = null
      topicClustersFetchedForRoot = null
      topicClustersLoadState.value = 'error'
      topicClustersErrorDetail.value = tc.message
      topicClustersSchemaWarning.value = null
    }
  }

  /**
   * Fetch ``/api/corpus/topic-clusters`` for the current ``corpusPath`` (no artifact load required).
   * Safe to call as soon as corpus root + healthy API are known so the Dashboard corpus workspace can show status.
   *
   * #769 memoized: if the most recent successful fetch was for the same
   * root AND the doc is still populated, the HTTP call is skipped.
   */
  async function syncTopicClustersForCurrentCorpus(): Promise<void> {
    const root = corpusPath.value.trim()
    if (!root) {
      return
    }
    if (topicClustersFetchedForRoot === root && topicClustersDoc.value !== null) {
      return
    }
    try {
      const tc = await fetchTopicClustersFromApi(root)
      applyTopicClustersFetchResult(tc, root)
    } catch (e) {
      topicClustersDoc.value = null
      topicClustersFetchedForRoot = null
      topicClustersLoadState.value = 'error'
      topicClustersErrorDetail.value = e instanceof Error ? e.message : String(e)
      topicClustersSchemaWarning.value = null
    }
  }

  /** Offline / no-backend: parse selected .gi.json / .kg.json files in the browser. */
  async function loadFromLocalFiles(files: FileList | null): Promise<void> {
    if (!files || files.length === 0) return
    manualGraphSelection.value = true
    loadError.value = null
    parsedList.value = []
    bridgeDocument.value = null
    topicClustersDoc.value = null
    topicClustersFetchedForRoot = null
    topicClustersLoadState.value = 'idle'
    topicClustersErrorDetail.value = null
    topicClustersSchemaWarning.value = null
    selectedRelPaths.value = []
    void import('./graphExpansion').then((m) => {
      m.useGraphExpansionStore().resetExpansionState()
    })
    const seq = loadGate.bump()
    loading.value = true
    try {
      type BridgeHold = { name: string; text: string; mtime: number }
      const bridgeHolds: BridgeHold[] = []
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
        if (lower.endsWith('.bridge.json')) {
          try {
            const text = await file.text()
            bridgeHolds.push({
              name: file.name,
              text,
              mtime: file.lastModified,
            })
          } catch {
            /* ignore */
          }
          continue
        }
        if (!lower.endsWith('.gi.json') && !lower.endsWith('.kg.json')) {
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
        loadError.value = 'No .gi.json or .kg.json files in selection.'
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
      const stems = new Set(kept.map((a) => episodeStemFromArtifactRelPath(a.name)))
      bridgeDocument.value = null
      for (const b of bridgeHolds) {
        const st = episodeStemFromArtifactRelPath(b.name)
        if (stems.has(st)) {
          try {
            bridgeDocument.value = parseBridgeDocument(JSON.parse(b.text))
          } catch {
            /* ignore invalid bridge */
          }
          break
        }
      }
      parsedList.value = kept
      selectedRelPaths.value = kept.map((p) => p.name)
      topicClustersDoc.value = null
      topicClustersFetchedForRoot = null
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
    // Clear auto-merge snapshot only on full reloads (not expansion/append)
    if (!opts?.preserveExpansion) {
      lastAutoMergeSnapshotKey = ''
    }
    // #586 fix: do NOT clear parsedList at the start. An intermediate
    // ``parsedList = []`` triggers the GraphCanvas ``filteredArtifact``
    // watcher → ``redraw()`` on an empty graph while the previous
    // Cytoscape layout may still have a pending rAF callback. That rAF
    // fires against a destroyed renderer and throws
    // ``TypeError: can't access property "notify", renderer is null``.
    // Keep the prior list visible until we atomically swap in ``out`` at
    // the end; expand flows (preserveExpansion=true) especially benefit
    // from this — the user sees a smooth add, not a clear-and-rebuild.
    bridgeDocument.value = null
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
    const seq = loadGate.bump()
    loading.value = true
    try {
      // #768 — parallelize: dispatch the main set AND the sibling-bridge
      // candidate (if any) concurrently in one ``Promise.all``. Old code
      // awaited each ``fetchArtifactJson`` serially; with 4–30 files on
      // cold corpora the round-trip latency stacked into the dominant
      // share of cross-surface "Open in graph" wall-clock time.
      //
      // Invariants this refactor preserves vs. the old sequential code:
      //  - Stale-check fires ONCE after the join (the old per-iteration
      //    check couldn't help anyway because in-flight fetches can't be
      //    cancelled — a mid-iteration ``isStale`` just abandons later
      //    fetches, doesn't stop the ones already issued).
      //  - In-selection ``.bridge.json`` WINS over sibling-bridge
      //    fallback (sibling is only consulted when nothing in the main
      //    selection ended up populating ``bridgeDocument``).
      //  - Main-fetch rejection still fails fast via ``Promise.all``
      //    rejecting on first error → outer ``catch`` sets ``loadError``.
      //  - Sibling-bridge rejection stays silent (it's optional metadata).
      //
      // Tests pinning each invariant: ``artifacts.loadSelected.test.ts``
      // (search for ``#768``).
      const selected = selectedRelPaths.value
      const giRelForSibling = selected.find((p) => p.toLowerCase().endsWith('.gi.json'))
      const hasBridgeInSelection = selected.some((p) =>
        p.toLowerCase().endsWith('.bridge.json'),
      )
      const siblingBridgeRel =
        !hasBridgeInSelection && giRelForSibling
          ? (() => {
              const br = giRelForSibling.replace(/\.gi\.json$/i, '.bridge.json')
              return br !== giRelForSibling ? br : null
            })()
          : null

      // Main-set promises: rejections propagate (fail-fast). Sibling-bridge
      // promise is wrapped to resolve-to-null on failure (silent / optional).
      const mainPromises = selected.map((rel) => fetchArtifactJson(root, rel))
      const siblingPromise = siblingBridgeRel
        ? fetchArtifactJson(root, siblingBridgeRel).catch(() => null)
        : Promise.resolve(null)

      const [mainResults, siblingData] = await Promise.all([
        Promise.all(mainPromises),
        siblingPromise,
      ])

      // F3a — single stale-check after the join. If the operator clicked
      // something else mid-flight, drop all post-join state writes.
      if (loadGate.isStale(seq)) {
        return
      }

      // Apply results in original selection order so ``parsedList`` is
      // deterministic and ``bridgeDocument`` resolution is predictable.
      const out: ParsedArtifact[] = []
      let bridgeFromSelection: ReturnType<typeof parseBridgeDocument> = null
      for (let i = 0; i < selected.length; i++) {
        const rel = selected[i]!
        const data = mainResults[i]!
        const lower = rel.toLowerCase()
        if (lower.endsWith('.bridge.json')) {
          try {
            bridgeFromSelection = parseBridgeDocument(data)
          } catch {
            /* optional */
          }
          continue
        }
        const base = rel.includes('/') ? rel.split('/').pop() || rel : rel
        out.push(parseArtifact(base, data, rel))
      }

      if (out.length === 0) {
        // Only clear on definite "nothing to show" — matches pre-#586
        // behaviour for this branch.
        parsedList.value = []
        loadError.value = 'No .gi.json or .kg.json files in selection.'
        return
      }

      /** Align topic-cluster catalog with the artifact slice before assigning ``parsedList``. */
      await syncTopicClustersForCurrentCorpus()
      if (loadGate.isStale(seq)) {
        return
      }

      parsedList.value = out
      // In-selection bridge WINS over sibling-bridge fallback. The
      // sibling fetch already fired in parallel; its result is only
      // applied when the selection itself didn't carry a bridge.
      if (bridgeFromSelection) {
        bridgeDocument.value = bridgeFromSelection
      } else if (siblingData) {
        try {
          bridgeDocument.value = parseBridgeDocument(siblingData)
        } catch {
          /* sibling bridge is optional */
        }
      }
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
    const next = String(p)
    // #769 — invalidate the topic-clusters cache whenever the corpus
    // root changes. The sentinel is also reset by clearSelection /
    // loadFromLocalFiles / fetch-error paths; this handles the direct
    // ``setCorpusPath`` case (operator typed a new path).
    if (next.trim() !== corpusPath.value.trim()) {
      topicClustersFetchedForRoot = null
    }
    corpusPath.value = next
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
    bridgeDocument.value = null
    topicClustersDoc.value = null
    topicClustersFetchedForRoot = null
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

  function clearManualGraphSelection(): void {
    manualGraphSelection.value = false
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
    // Skip if last load was from external source (Digest/Library) - user wants focused view only
    // Don't clear the flag here - let the caller clear it after all their operations complete
    if (lastLoadSource.value === 'digest-external' || lastLoadSource.value === 'subject-external') {
      return
    }
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
    // Don't auto-merge on tab switch if we already ran for this exact artifact set
    const currentSnapshotKey = selectedRelPaths.value.slice().sort().join('|')
    if (currentSnapshotKey === lastAutoMergeSnapshotKey) {
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
      // No candidates found, update snapshot to current state
      lastAutoMergeSnapshotKey = currentSnapshotKey
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
          // Update snapshot AFTER successful append with the NEW state
          lastAutoMergeSnapshotKey = selectedRelPaths.value.slice().sort().join('|')
        } else {
          // No paths added, keep current snapshot
          lastAutoMergeSnapshotKey = currentSnapshotKey
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

  /**
   * Append GI/KG for ``topic_clusters.json`` member ``episode_ids`` until the TopicCluster
   * compound exists in the merged + overlay graph — same catalog path as **Load** on a member
   * row, without replacing the whole corpus selection.
   */
  async function ensureTopicClusterCompoundVisible(compoundParentId: string): Promise<boolean> {
    const cid = compoundParentId.trim()
    if (!cid) {
      return true
    }
    const root = corpusPath.value.trim()
    if (!root) {
      return false
    }
    await syncTopicClustersForCurrentCorpus()
    const doc = topicClustersDoc.value
    const cluster = doc?.clusters?.length
      ? findClusterByCompoundId(doc, cid)
      : null
    if (!cluster) {
      return true
    }

    const mergedHasCompound = (): boolean => {
      const da = displayArtifact.value
      return Boolean(da && findRawNodeInArtifact(da, cid))
    }
    if (mergedHasCompound()) {
      return true
    }

    const cap = clusterSiblingEpisodeCap()
    let pool = allEpisodeIdsListedForCluster(cluster)
    if (pool.length === 0) {
      return false
    }
    let wave = 0
    while (!mergedHasCompound() && pool.length > 0 && wave < 16) {
      wave += 1
      const loaded = episodeIdsFromParsedArtifacts(parsedList.value)
      const candidateIds = pool.filter((e) => !loaded.has(e))
      if (candidateIds.length === 0) {
        break
      }
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
      if (pathsToAdd.length === 0) {
        break
      }
      await appendRelativeArtifacts(pathsToAdd)
      const loaded2 = episodeIdsFromParsedArtifacts(parsedList.value)
      pool = allEpisodeIdsListedForCluster(cluster).filter((e) => !loaded2.has(e))
    }
    return mergedHasCompound()
  }

  /**
   * When ``parsedList`` is empty and the user focuses a TopicCluster compound, try to populate
   * the graph **only** from that cluster's ``members[].episode_ids`` (catalog resolve + append).
   * Returns true if at least one GI/KG file was loaded — then App can **skip** the default
   * capped corpus-wide ``syncMergedGraphFromCorpusApi`` sweep (avoids clear + full reload flash).
   */
  async function maybeBootstrapGraphFromTopicClusterOnly(
    compoundParentId: string,
  ): Promise<boolean> {
    const cid = compoundParentId.trim()
    if (!cid) {
      return false
    }
    await syncTopicClustersForCurrentCorpus()
    const doc = topicClustersDoc.value
    const cluster = doc?.clusters?.length ? findClusterByCompoundId(doc, cid) : null
    if (!cluster || allEpisodeIdsListedForCluster(cluster).length === 0) {
      return false
    }
    await ensureTopicClusterCompoundVisible(cid)
    return parsedList.value.length > 0
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
    clearManualGraphSelection,
    selectAllListed,
    deselectAllListed,
    syncTopicClustersForCurrentCorpus,
    appendRelativeArtifacts,
    removeRelativeArtifacts,
    maybeMergeClusterSiblingEpisodes,
    setLoadSource,
    clearLoadSource,
    currentLoadSource,  // Readonly getter for lastLoadSource
    ensureTopicClusterCompoundVisible,
    maybeBootstrapGraphFromTopicClusterOnly,
  }
})
