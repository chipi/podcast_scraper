import { defineStore } from 'pinia'
import { computed, reactive, ref } from 'vue'
import posthog from 'posthog-js'
import {
  searchCorpus,
  type CorpusSearchLiftStats,
  type SearchHit,
} from '../api/searchApi'
import { useGraphNavigationStore } from './graphNavigation'
import { useActiveSearchContextStore } from './activeSearchContext'
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

  const searchRunGate = new StaleGeneration()

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
    try {
      const body = await searchCorpus(q, {
        path: root,
        types: filters.types.length ? filters.types : undefined,
        feed: filters.feed || undefined,
        since: filters.since || undefined,
        speaker: filters.speaker || undefined,
        topic: filters.topic || undefined,
        groundedOnly: filters.groundedOnly,
        topK: filters.topK,
        embeddingModel: filters.embeddingModel.trim() || undefined,
        dedupeKgSurfaces: filters.dedupeKgSurfaces,
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
    useActiveSearchContextStore().clear()
    useGraphNavigationStore().clearLibraryEpisodeHighlights()
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
  }
})
