import { defineStore } from 'pinia'
import { computed, reactive, ref } from 'vue'
import { fetchExploreFiltered, fetchExploreNaturalLanguage, type ExploreApiBody } from '../api/exploreApi'

export interface ExploreQuote {
  text: string
  speaker_name?: string
  start_ms?: number
  end_ms?: number
}

export interface ExploreInsightRow {
  insight_id: string
  text: string
  grounded?: boolean
  confidence?: number
  episode?: { episode_id?: string; title?: string; publish_date?: string }
  supporting_quotes?: ExploreQuote[]
}

export interface ExploreSummary {
  insight_count: number
  grounded_insight_count: number
  quote_count: number
  episode_count: number
  speaker_count: number
  topic_count: number
  episodes_searched: number
}

export interface ExploreTopSpeaker {
  speaker_id: string
  name: string | null
  quote_count: number
  insight_count: number
}

export interface ExploreTopicRow {
  topic_id: string
  label: string
  insight_count: number
}

export const useExploreStore = defineStore('explore', () => {
  const loading = ref(false)
  const error = ref<string | null>(null)
  const last = ref<ExploreApiBody | null>(null)

  const filters = reactive({
    topic: '',
    speaker: '',
    groundedOnly: false,
    minConfidence: '' as string,
    sortBy: 'confidence' as 'confidence' | 'time',
    limit: 50,
    strict: false,
  })

  const nlQuestion = ref('')

  const insightRows = computed((): ExploreInsightRow[] => {
    const L = last.value
    if (!L || L.error) return []
    const src =
      L.kind === 'explore' ? L.data : L.answer
    if (!src || typeof src !== 'object') return []
    const raw = src.insights
    if (!Array.isArray(raw)) return []
    const out: ExploreInsightRow[] = []
    for (const x of raw) {
      if (!x || typeof x !== 'object') continue
      const o = x as Record<string, unknown>
      const id = o.insight_id
      if (typeof id !== 'string' || !id) continue
      const ep = o.episode
      const episode =
        ep && typeof ep === 'object'
          ? (ep as { episode_id?: string; title?: string; publish_date?: string })
          : undefined

      let quotes: ExploreQuote[] | undefined
      const rawQ = o.supporting_quotes
      if (Array.isArray(rawQ) && rawQ.length > 0) {
        quotes = rawQ
          .filter((q): q is Record<string, unknown> => q != null && typeof q === 'object')
          .map((q) => {
            const spk = q.speaker as Record<string, unknown> | undefined
            return {
              text: typeof q.text === 'string' ? q.text : '',
              speaker_name:
                spk && typeof spk === 'object'
                  ? (typeof spk.name === 'string' ? spk.name : String(spk.speaker_id ?? ''))
                  : undefined,
              start_ms: typeof q.timestamp_start_ms === 'number' ? q.timestamp_start_ms : undefined,
              end_ms: typeof q.timestamp_end_ms === 'number' ? q.timestamp_end_ms : undefined,
            }
          })
      }

      out.push({
        insight_id: id,
        text: typeof o.text === 'string' ? o.text : '',
        grounded: typeof o.grounded === 'boolean' ? o.grounded : undefined,
        confidence: typeof o.confidence === 'number' ? o.confidence : undefined,
        episode,
        supporting_quotes: quotes,
      })
    }
    return out
  })

  const leaderboardRows = computed((): ExploreTopicRow[] => {
    const L = last.value
    if (!L || L.error) return []
    const src = L.kind === 'natural_language' ? L.answer : L.data
    if (!src || typeof src !== 'object') return []
    const topics = src.topics
    if (!Array.isArray(topics) || topics.length === 0) return []
    return topics
      .filter((t): t is Record<string, unknown> => t != null && typeof t === 'object')
      .filter((t) => typeof t.label === 'string' || typeof t.topic_id === 'string')
      .map((t) => ({
        topic_id: typeof t.topic_id === 'string' ? t.topic_id : '',
        label: typeof t.label === 'string' ? t.label : '',
        insight_count: typeof t.insight_count === 'number' ? t.insight_count : 0,
      }))
  })

  const summaryBlock = computed((): ExploreSummary | null => {
    const L = last.value
    if (!L || L.error) return null
    const src = L.kind === 'explore' ? L.data : L.answer
    if (!src || typeof src !== 'object') return null
    const raw = src.summary
    if (!raw || typeof raw !== 'object') return null
    const s = raw as Record<string, unknown>
    const insightCount = typeof s.insight_count === 'number' ? s.insight_count : 0
    const groundedCount = typeof s.grounded_insight_count === 'number' ? s.grounded_insight_count : 0
    const quoteCount = typeof s.quote_count === 'number' ? s.quote_count : 0
    const episodeCount = typeof s.episode_count === 'number' ? s.episode_count : 0
    const speakerCount = typeof s.speaker_count === 'number' ? s.speaker_count : 0
    const topicCount = typeof s.topic_count === 'number' ? s.topic_count : 0
    const epSearched = typeof (src as Record<string, unknown>).episodes_searched === 'number'
      ? (src as Record<string, unknown>).episodes_searched as number
      : 0
    if (insightCount === 0 && groundedCount === 0 && epSearched === 0) return null
    return {
      insight_count: insightCount,
      grounded_insight_count: groundedCount,
      quote_count: quoteCount,
      episode_count: episodeCount,
      speaker_count: speakerCount,
      topic_count: topicCount,
      episodes_searched: epSearched,
    }
  })

  const topSpeakers = computed((): ExploreTopSpeaker[] => {
    const L = last.value
    if (!L || L.error) return []
    const src = L.kind === 'explore' ? L.data : L.answer
    if (!src || typeof src !== 'object') return []
    const raw = src.top_speakers
    if (!Array.isArray(raw) || raw.length === 0) return []
    return raw
      .filter((x): x is Record<string, unknown> => x != null && typeof x === 'object')
      .filter((x) => typeof x.speaker_id === 'string')
      .map((x) => ({
        speaker_id: x.speaker_id as string,
        name: typeof x.name === 'string' ? x.name : null,
        quote_count: typeof x.quote_count === 'number' ? x.quote_count : 0,
        insight_count: typeof x.insight_count === 'number' ? x.insight_count : 0,
      }))
  })

  async function runFilteredExplore(corpusPath: string): Promise<void> {
    error.value = null
    last.value = null
    const root = corpusPath.trim()
    if (!root) {
      error.value = 'Set corpus root first.'
      return
    }
    loading.value = true
    try {
      const minRaw = filters.minConfidence.trim()
      let minConfidence: number | null = null
      if (minRaw) {
        const n = Number(minRaw)
        if (Number.isFinite(n)) minConfidence = n
      }
      last.value = await fetchExploreFiltered(root, {
        topic: filters.topic,
        speaker: filters.speaker,
        groundedOnly: filters.groundedOnly,
        minConfidence,
        sortBy: filters.sortBy,
        limit: filters.limit,
        strict: filters.strict,
      })
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  async function runNaturalLanguage(corpusPath: string): Promise<void> {
    error.value = null
    last.value = null
    const root = corpusPath.trim()
    if (!root) {
      error.value = 'Set corpus root first.'
      return
    }
    const q = nlQuestion.value.trim()
    if (!q) {
      error.value = 'Enter a question.'
      return
    }
    loading.value = true
    try {
      last.value = await fetchExploreNaturalLanguage(root, q, {
        limit: filters.limit,
        strict: filters.strict,
      })
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  function clearOutput(): void {
    last.value = null
    error.value = null
  }

  return {
    loading,
    error,
    last,
    filters,
    nlQuestion,
    insightRows,
    leaderboardRows,
    summaryBlock,
    topSpeakers,
    runFilteredExplore,
    runNaturalLanguage,
    clearOutput,
  }
})
