export interface IndexStatsBody {
  total_vectors: number
  doc_type_counts: Record<string, number>
  feeds_indexed: string[]
  embedding_model: string
  embedding_dim: number
  last_updated: string
  index_size_bytes: number
}

export interface IndexStatsEnvelope {
  available: boolean
  reason?: string | null
  index_path?: string | null
  stats?: IndexStatsBody | null
}

export async function fetchIndexStats(corpusPath?: string): Promise<IndexStatsEnvelope> {
  const p = corpusPath?.trim()
  const url = p
    ? `/api/index/stats?${new URLSearchParams({ path: p })}`
    : '/api/index/stats'
  const res = await fetch(url)
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || `HTTP ${res.status}`)
  }
  return (await res.json()) as IndexStatsEnvelope
}
