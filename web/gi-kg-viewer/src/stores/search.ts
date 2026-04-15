import { defineStore } from 'pinia'
import { reactive, ref } from 'vue'
import {
  searchCorpus,
  type CorpusSearchLiftStats,
  type SearchHit,
} from '../api/searchApi'
import { useGraphNavigationStore } from './graphNavigation'
import { normalizeFeedIdForViewer } from '../utils/feedId'
import { StaleGeneration } from '../utils/staleGeneration'

export const useSearchStore = defineStore('search', () => {
  const query = ref('')
  const loading = ref(false)
  const error = ref<string | null>(null)
  const apiError = ref<string | null>(null)
  const results = ref<SearchHit[]>([])
  const liftStats = ref<CorpusSearchLiftStats | null>(null)

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
    try {
      const body = await searchCorpus(q, {
        path: root,
        types: filters.types.length ? filters.types : undefined,
        feed: filters.feed || undefined,
        since: filters.since || undefined,
        speaker: filters.speaker || undefined,
        groundedOnly: filters.groundedOnly,
        topK: filters.topK,
        embeddingModel: filters.embeddingModel.trim() || undefined,
        dedupeKgSurfaces: filters.dedupeKgSurfaces,
      })
      if (searchRunGate.isStale(seq)) {
        return
      }
      if (body.error) {
        apiError.value = body.detail
          ? `${body.error}: ${body.detail}`
          : body.error
        results.value = []
        liftStats.value = null
        return
      }
      results.value = body.results
      liftStats.value = body.lift_stats ?? null
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
    useGraphNavigationStore().clearLibraryEpisodeHighlights()
  }

  return {
    query,
    loading,
    error,
    apiError,
    results,
    liftStats,
    filters,
    feedFilterDisplayLabel,
    feedFilterHandoffPristine,
    applyLibrarySearchHandoff,
    commitFeedFilterUiInput,
    runSearch,
    clearResults,
  }
})
