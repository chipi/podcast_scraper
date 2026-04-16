import { fetchWithTimeout } from './httpClient'

/** RFC-075 query-time join from ``search/topic_clusters.json`` (``kg_topic`` hits only). */
export interface TopicClusterHitMeta {
  graph_compound_parent_id: string
  canonical_label: string
  cil_alias_target_topic_id?: string
}

export interface SearchHit {
  doc_id: string
  score: number
  metadata: Record<string, unknown>
  text: string
  supporting_quotes?: Record<string, unknown>[] | null
  /** RFC-072 chunk-to-Insight lift when the server enriches transcript rows (#528). */
  lifted?: Record<string, unknown> | null
}

/** Counters for the current search response page (after top-k slice). */
export interface CorpusSearchLiftStats {
  transcript_hits_returned: number
  lift_applied: number
}

export interface SearchResponse {
  query: string
  results: SearchHit[]
  error?: string | null
  detail?: string | null
  lift_stats?: CorpusSearchLiftStats | null
}

export interface SearchRequestOptions {
  path: string
  types?: string[]
  feed?: string
  since?: string
  speaker?: string
  groundedOnly?: boolean
  topK?: number
  embeddingModel?: string
  /** Default true: collapse duplicate kg_entity/kg_topic surfaces server-side. */
  dedupeKgSurfaces?: boolean
}

export async function searchCorpus(
  q: string,
  options: SearchRequestOptions,
): Promise<SearchResponse> {
  const root = options.path.trim()
  const params = new URLSearchParams({ path: root, q: q.trim() })
  const tk = options.topK ?? 10
  params.set('top_k', String(Math.min(100, Math.max(1, tk))))
  if (options.groundedOnly) params.set('grounded_only', 'true')
  if (options.feed?.trim()) params.set('feed', options.feed.trim())
  if (options.since?.trim()) params.set('since', options.since.trim())
  if (options.speaker?.trim()) params.set('speaker', options.speaker.trim())
  if (options.embeddingModel?.trim()) {
    params.set('embedding_model', options.embeddingModel.trim())
  }
  for (const t of options.types ?? []) {
    if (t.trim()) params.append('type', t.trim())
  }
  if (options.dedupeKgSurfaces === false) {
    params.set('dedupe_kg_surfaces', 'false')
  }
  const res = await fetchWithTimeout(`/api/search?${params.toString()}`)
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || `HTTP ${res.status}`)
  }
  return (await res.json()) as SearchResponse
}
