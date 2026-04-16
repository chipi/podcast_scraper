import { dedupeInFlight } from './inFlightDedupe'
import { fetchWithTimeout } from './httpClient'

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
  /** Heuristic (GitHub #507): run `podcast index` / rebuild when true. */
  reindex_recommended?: boolean
  reindex_reasons?: string[]
  artifact_newest_mtime?: string | null
  search_root_hints?: string[]
  rebuild_in_progress?: boolean
  rebuild_last_error?: string | null
}

export interface IndexRebuildAccepted {
  accepted: boolean
  corpus_path: string
  rebuild: boolean
}

export async function fetchIndexStats(corpusPath?: string): Promise<IndexStatsEnvelope> {
  const p = corpusPath?.trim()
  const url = p
    ? `/api/index/stats?${new URLSearchParams({ path: p })}`
    : '/api/index/stats'
  return dedupeInFlight(`GET|${url}`, async () => {
    const res = await fetchWithTimeout(url)
    if (!res.ok) {
      const t = await res.text()
      throw new Error(t || `HTTP ${res.status}`)
    }
    return (await res.json()) as IndexStatsEnvelope
  })
}

export async function postIndexRebuild(opts: {
  corpusPath?: string
  rebuild?: boolean
}): Promise<IndexRebuildAccepted> {
  const q = new URLSearchParams()
  if (opts.corpusPath?.trim()) {
    q.set('path', opts.corpusPath.trim())
  }
  if (opts.rebuild) {
    q.set('rebuild', 'true')
  }
  const qs = q.toString()
  const url = qs ? `/api/index/rebuild?${qs}` : '/api/index/rebuild'
  const res = await fetchWithTimeout(url, { method: 'POST' })
  if (res.status === 409) {
    const t = await res.text()
    throw new Error(t || 'Index rebuild already running for this corpus.')
  }
  if (!res.ok) {
    const t = await res.text()
    throw new Error(t || `HTTP ${res.status}`)
  }
  return (await res.json()) as IndexRebuildAccepted
}
