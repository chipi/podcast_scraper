export interface SearchHit {
  doc_id: string
  score: number
  metadata: Record<string, unknown>
  text: string
  supporting_quotes?: Record<string, unknown>[] | null
}

export interface SearchResponse {
  query: string
  results: SearchHit[]
  error?: string | null
  detail?: string | null
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
  const res = await fetch(`/api/search?${params.toString()}`)
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || `HTTP ${res.status}`)
  }
  return (await res.json()) as SearchResponse
}
