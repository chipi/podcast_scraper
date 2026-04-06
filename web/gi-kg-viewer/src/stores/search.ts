import { defineStore } from 'pinia'
import { reactive, ref } from 'vue'
import { searchCorpus, type SearchHit } from '../api/searchApi'

export const useSearchStore = defineStore('search', () => {
  const query = ref('')
  const loading = ref(false)
  const error = ref<string | null>(null)
  const apiError = ref<string | null>(null)
  const results = ref<SearchHit[]>([])
  const lastSubmittedQuery = ref('')

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
  })

  async function runSearch(corpusPath: string): Promise<void> {
    const q = query.value.trim()
    error.value = null
    apiError.value = null
    if (!q) {
      error.value = 'Enter a search query.'
      return
    }
    const root = corpusPath.trim()
    if (!root) {
      error.value = 'Set corpus root first (same folder as List files).'
      return
    }
    loading.value = true
    results.value = []
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
      })
      lastSubmittedQuery.value = body.query
      if (body.error) {
        apiError.value = body.detail
          ? `${body.error}: ${body.detail}`
          : body.error
        results.value = []
        return
      }
      results.value = body.results
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      results.value = []
    } finally {
      loading.value = false
    }
  }

  function clearResults(): void {
    results.value = []
    apiError.value = null
    error.value = null
  }

  return {
    query,
    loading,
    error,
    apiError,
    results,
    lastSubmittedQuery,
    filters,
    runSearch,
    clearResults,
  }
})
