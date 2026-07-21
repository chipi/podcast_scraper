import { defineStore } from 'pinia'
import { computed, reactive, ref } from 'vue'
import posthog from 'posthog-js'
import {
  compareSubjects as compareSubjectsApi,
  searchCorpus,
  type CompareSubjectRef,
  type CorpusSearchLiftStats,
  type SearchClusterGroup,
  type SearchCompareResponse,
  type SearchConsensusPair,
  type SearchHit,
} from '../api/searchApi'
import { useGraphNavigationStore } from './graphNavigation'
import { useActiveSearchContextStore } from './activeSearchContext'
import { useSavedQueriesStore } from './savedQueries'
import { useShellStore } from './shell'
import { normalizeFeedIdForViewer } from '../utils/feedId'
import { StaleGeneration } from '../utils/staleGeneration'

/** Friendly text for server search error codes (mirrors the similar-episodes panel). */
function mapSearchError(code: string, detail: string | null | undefined): string {
  if (code === 'no_index') {
    return 'No vector index for this corpus yet. Run indexing to enable search.'
  }
  if (code === 'embed_failed') {
    return 'Embedding failed (model missing or offline) — the index is fine, but the query could not be embedded.'
  }
  if (code === 'empty_query') {
    return 'Enter a search query.'
  }
  return detail ? `${code}: ${detail}` : code
}

export const useSearchStore = defineStore('search', () => {
  const query = ref('')
  const loading = ref(false)
  const error = ref<string | null>(null)
  const apiError = ref<string | null>(null)
  const results = ref<SearchHit[]>([])
  const liftStats = ref<CorpusSearchLiftStats | null>(null)
  /** Detected query intent for the last run (PRD-033 FR1.4); null until a run. */
  const queryType = ref<string | null>(null)
  /** UXS-008: last search reported enrichment failure (server `enrichment_error`). */
  const enrichmentCallFailed = ref(false)

  /**
   * Search v3 §S4b — server operator state. Populated by ``runOperator``;
   * plain ``runSearch`` clears both back to null so a re-run of a bare
   * query drops any stale cluster / consensus panels the UI was showing.
   */
  const clusters = ref<SearchClusterGroup[] | null>(null)
  const consensusPairs = ref<SearchConsensusPair[] | null>(null)
  const operatorLoading = ref<'cluster' | 'consensus' | 'compare' | null>(null)
  const operatorError = ref<string | null>(null)

  /**
   * Search v3 §S8 — Compare (2 subjects) state. Populated by ``runCompare``
   * (POST /api/search/compare); ``runSearch`` and ``runOperator`` do NOT
   * clear this so the caller can leave the panel open across a re-run of
   * the underlying query.
   */
  const compareResult = ref<SearchCompareResponse | null>(null)
  const compareLoading = ref(false)
  const compareError = ref<string | null>(null)

  const searchRunGate = new StaleGeneration()
  const operatorRunGate = new StaleGeneration()
  const compareRunGate = new StaleGeneration()

  /**
   * When set with ``feedFilterHandoffPristine``, Advanced feed input shows this title while
   * ``filters.feed`` stays the catalog id substring for ``GET /api/search`` until the user edits.
   */
  const feedFilterDisplayLabel = ref<string | null>(null)
  const feedFilterHandoffPristine = ref(false)

  const filters = reactive({
    topK: 10,
    groundedOnly: false,
    feed: '',
    since: '',
    speaker: '',
    /** Lowercase doc_type values matching the index (insight, quote, …). */
    types: [] as string[],
    /** Override the default embedding model (blank = server default). */
    embeddingModel: '',
    /**
     * When true (default), server collapses kg_entity/kg_topic rows with the same embedded text.
     */
    dedupeKgSurfaces: true,
    /**
     * Topic-substring filter. Server-side since the S1 follow-up wired ``topic=``
     * into /api/search — passes through to the search API and drives retrieval
     * (kg_topic id/text + insight ABOUT-edge topic labels). No client-side
     * fallback anymore.
     */
    topic: '',
    /**
     * Client-side minimum confidence filter (Search v3 §S1 — Explore merge). Same
     * caveat as ``topic`` — applied over ``results``, not driving retrieval. Empty
     * string means no filter; a numeric string like "0.7" means ≥0.7 confidence.
     */
    minConfidence: '',
    /**
     * Search v3 §S6 — exact ``episode_id`` scope. Set by the "Search
     * within this episode" rail launcher on EpisodeDetailPanel; empty
     * string means no episode scope. Server-side (drives retrieval),
     * not client-side (drops non-matching hits before top-k slice).
     */
    episodeId: '',
    /**
     * Search v3 §S5 — request the server-side QueryEnricher chain
     * (RFC-088 chunk 5). When on, ``searchCorpus`` adds
     * ``enrich_results=true`` and hits come back decorated with
     * ``metadata.query_enrichments.related_topics``; the workspace hero
     * renders an aggregated summary of the top related topics.
     *
     * Default is ``null`` so the UI can auto-adopt the server's
     * capability signal (``shell.enrichedSearchAvailable``): when the
     * server advertises enrichment, we default the chip on; when it
     * doesn't, we default off. The user can still explicitly toggle.
     * Boolean once the user (or the auto-adopt) sets it.
     */
    enrichResults: null as boolean | null,
  })

  /**
   * Results with the client-side ``minConfidence`` filter applied. ``topic`` is
   * server-side since the S1 follow-up; ``minConfidence`` is still client-side
   * because /api/search doesn't accept a numeric confidence threshold param.
   * Keeps ``results`` untouched so callers that need the pre-filter page (intent
   * chip, tier counts) still see the server's top-K.
   */
  const filteredResults = computed<SearchHit[]>(() => {
    const minRaw = filters.minConfidence.trim()
    let minConf: number | null = null
    if (minRaw) {
      const n = Number(minRaw)
      if (Number.isFinite(n)) minConf = n
    }
    if (minConf == null) return results.value
    return results.value.filter((hit) => {
      const conf = (hit.metadata?.confidence as number | undefined) ?? null
      if (conf == null || conf < minConf) return false
      return true
    })
  })

  function applyLibrarySearchHandoff(
    feed: string,
    queryText: string,
    options?: { since?: string; feedDisplayTitle?: string },
  ): void {
    results.value = []
    liftStats.value = null
    apiError.value = null
    error.value = null
    filters.feed = normalizeFeedIdForViewer(feed)
    const title = options?.feedDisplayTitle?.trim()
    if (title && filters.feed.trim()) {
      feedFilterDisplayLabel.value = title
      feedFilterHandoffPristine.value = true
    } else {
      feedFilterDisplayLabel.value = null
      feedFilterHandoffPristine.value = false
    }
    query.value = queryText
    const sinceDay = options?.since?.trim().slice(0, 10)
    if (sinceDay && /^\d{4}-\d{2}-\d{2}$/.test(sinceDay)) {
      filters.since = sinceDay
    }
  }

  /** Advanced feed field: clears Library title pairing once the user edits. */
  function commitFeedFilterUiInput(raw: string): void {
    if (feedFilterHandoffPristine.value) {
      feedFilterHandoffPristine.value = false
      feedFilterDisplayLabel.value = null
    }
    filters.feed = raw
  }

  async function runSearch(corpusPath: string): Promise<void> {
    const q = query.value.trim()
    error.value = null
    apiError.value = null
    if (!q) {
      error.value = 'Enter a search query.'
      return
    }
    useGraphNavigationStore().clearLibraryEpisodeHighlights()
    const root = corpusPath.trim()
    if (!root) {
      error.value = 'Set corpus root first (same folder as List files).'
      return
    }
    const seq = searchRunGate.bump()
    loading.value = true
    results.value = []
    liftStats.value = null
    queryType.value = null
    enrichmentCallFailed.value = false
    clusters.value = null
    consensusPairs.value = null
    operatorError.value = null
    try {
      const body = await searchCorpus(q, {
        path: root,
        types: filters.types.length ? filters.types : undefined,
        feed: filters.feed || undefined,
        since: filters.since || undefined,
        speaker: filters.speaker || undefined,
        topic: filters.topic || undefined,
        episodeId: filters.episodeId || undefined,
        groundedOnly: filters.groundedOnly,
        topK: filters.topK,
        embeddingModel: filters.embeddingModel.trim() || undefined,
        dedupeKgSurfaces: filters.dedupeKgSurfaces,
        // Search v3 §S5 — request enrichment when effectively on.
        // Tri-state resolution: explicit ``true``/``false`` from the user
        // wins; ``null`` (default) auto-adopts the server's advertised
        // capability so first-time users see enricher output without
        // hunting for a toggle.
        enrichResults:
          filters.enrichResults === null
            ? Boolean(useShellStore().enrichedSearchAvailable)
            : filters.enrichResults === true,
      })
      if (searchRunGate.isStale(seq)) {
        return
      }
      if (body.error) {
        apiError.value = mapSearchError(body.error, body.detail)
        results.value = []
        liftStats.value = null
        return
      }
      results.value = body.results
      liftStats.value = body.lift_stats ?? null
      queryType.value = body.query_type ?? null
      // RFC-094 OQ-2: publish the active context so Library/Graph can rank + snippet.
      useActiveSearchContextStore().setContext(q, body.results)
      // Search v3 §S7 — push onto USERPREFS-1 Recent ring buffer. Best-
      // effort: userPrefs.set never rejects, so a persistence failure
      // never blocks the user-visible search flow. Runs off-path so the
      // async PATCH doesn't tack onto the runSearch latency budget.
      void useSavedQueriesStore().pushRecent(q)
      enrichmentCallFailed.value = Boolean(
        body.enrichment_error && String(body.enrichment_error).trim(),
      )
      posthog.capture('search_run', {
        query_length: q.length,
        result_count: body.results.length,
        filter_types: filters.types.length ? filters.types : [],
        has_feed_filter: Boolean(filters.feed),
        has_speaker_filter: Boolean(filters.speaker),
        has_since_filter: Boolean(filters.since),
        grounded_only: filters.groundedOnly,
        top_k: filters.topK,
        enrichment_failed: Boolean(body.enrichment_error && String(body.enrichment_error).trim()),
      })
    } catch (e) {
      if (searchRunGate.isStale(seq)) {
        return
      }
      error.value = e instanceof Error ? e.message : String(e)
      results.value = []
      liftStats.value = null
    } finally {
      if (searchRunGate.isCurrent(seq)) {
        loading.value = false
      }
    }
  }

  function clearResults(): void {
    results.value = []
    apiError.value = null
    error.value = null
    queryType.value = null
    enrichmentCallFailed.value = false
    clusters.value = null
    consensusPairs.value = null
    operatorError.value = null
    useActiveSearchContextStore().clear()
    useGraphNavigationStore().clearLibraryEpisodeHighlights()
  }

  /**
   * Search v3 §S4b — request a server-side operator over the current query.
   * Re-fires the underlying /api/search endpoint with ``operator=…`` and
   * (per RFC-107 §7.4) a ``top_k * 3`` over-fetch so the aggregation has a
   * meaningful sample to group / filter over. The returned page REPLACES
   * ``results`` so the caller renders the operator-scoped hit set (a plain
   * query re-run restores the default top-k).
   *
   * Silent-no-op when there's no query. Never raises — errors surface via
   * ``operatorError``.
   */
  async function runOperator(
    corpusPath: string,
    operator: 'cluster' | 'consensus',
  ): Promise<void> {
    const q = query.value.trim()
    const root = corpusPath.trim()
    if (!q || !root) return
    const seq = operatorRunGate.bump()
    operatorLoading.value = operator
    operatorError.value = null
    try {
      const body = await searchCorpus(q, {
        path: root,
        types: filters.types.length ? filters.types : undefined,
        feed: filters.feed || undefined,
        since: filters.since || undefined,
        speaker: filters.speaker || undefined,
        topic: filters.topic || undefined,
        episodeId: filters.episodeId || undefined,
        groundedOnly: filters.groundedOnly,
        // Over-fetch so the operator has room to group / filter over
        // more than the default top-10. RFC-107 §7.4 sets the multiplier.
        topK: Math.min(100, filters.topK * 3),
        embeddingModel: filters.embeddingModel.trim() || undefined,
        dedupeKgSurfaces: filters.dedupeKgSurfaces,
        operator,
        // Search v3 §S5 followup — thread the same enrichment flag runSearch
        // uses so an operator toggle does NOT strip ``query_enrichments``
        // off the visible hit set (the operator response REPLACES
        // ``results``, and the enriched-answer hero re-derives from that
        // replacement).
        enrichResults:
          filters.enrichResults === null
            ? Boolean(useShellStore().enrichedSearchAvailable)
            : filters.enrichResults === true,
      })
      if (operatorRunGate.isStale(seq)) return
      if (body.error) {
        operatorError.value = mapSearchError(body.error, body.detail)
        return
      }
      // Overwrite the visible hit set with the operator page so the group
      // ``hit_indices`` line up with what the caller renders.
      results.value = body.results
      liftStats.value = body.lift_stats ?? null
      clusters.value = body.clusters ?? null
      consensusPairs.value = body.consensus_pairs ?? null
    } catch (e) {
      if (operatorRunGate.isStale(seq)) return
      operatorError.value = e instanceof Error ? e.message : String(e)
    } finally {
      if (operatorRunGate.isCurrent(seq)) {
        operatorLoading.value = null
      }
    }
  }

  /**
   * Search v3 §S8 — Compare 2 subjects over the corpus.
   *
   * Wraps ``POST /api/search/compare``. Never re-uses the visible hit
   * page — the endpoint runs its own subject-scoped searches for each
   * side. Silent-no-op when either picker slot is empty; errors surface
   * via ``compareError``.
   */
  async function runCompare(
    corpusPath: string,
    subjectA: CompareSubjectRef,
    subjectB: CompareSubjectRef,
    opts: { q?: string; topK?: number; maxTokens?: number } = {},
  ): Promise<void> {
    const root = corpusPath.trim()
    if (!root) return
    if (!subjectA.id.trim() || !subjectB.id.trim()) return
    const seq = compareRunGate.bump()
    compareLoading.value = true
    compareError.value = null
    try {
      const body = await compareSubjectsApi(subjectA, subjectB, {
        path: root,
        q: opts.q ?? query.value,
        topK: opts.topK ?? filters.topK,
        maxTokens: opts.maxTokens,
      })
      if (compareRunGate.isStale(seq)) return
      if (body.error) {
        compareError.value = mapSearchError(body.error, body.detail ?? null)
        compareResult.value = null
        return
      }
      compareResult.value = body
    } catch (e) {
      if (compareRunGate.isStale(seq)) return
      compareError.value = e instanceof Error ? e.message : String(e)
      compareResult.value = null
    } finally {
      if (compareRunGate.isCurrent(seq)) {
        compareLoading.value = false
      }
    }
  }

  function clearCompare(): void {
    compareResult.value = null
    compareError.value = null
  }

  return {
    query,
    loading,
    error,
    apiError,
    results,
    filteredResults,
    liftStats,
    queryType,
    enrichmentCallFailed,
    filters,
    feedFilterDisplayLabel,
    feedFilterHandoffPristine,
    applyLibrarySearchHandoff,
    commitFeedFilterUiInput,
    runSearch,
    clearResults,
    clusters,
    consensusPairs,
    operatorLoading,
    operatorError,
    runOperator,
    compareResult,
    compareLoading,
    compareError,
    runCompare,
    clearCompare,
  }
})
