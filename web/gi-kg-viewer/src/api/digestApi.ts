import { dedupeInFlight } from './inFlightDedupe'
import { fetchWithTimeout } from './httpClient'

function raiseDigestHttpError(res: Response, bodyText: string): never {
  if (res.status === 404) {
    throw new Error(
      'Digest endpoint not found (404). Restart the viewer API from a current checkout (pip install -e ".[server]", then make serve-api or podcast serve).',
    )
  }
  throw new Error(bodyText || `HTTP ${res.status}`)
}

export type DigestWindow = 'all' | '24h' | '7d' | '1mo' | 'since'

/** CIL topic chip (bridge identity + optional topic-cluster styling). */
export type CilDigestTopicPill = {
  topic_id: string
  label: string
  in_topic_cluster?: boolean
  topic_cluster_compound_id?: string | null
}

export type CorpusDigestRow = {
  metadata_relative_path: string
  feed_id: string
  feed_display_title?: string | null
  feed_rss_url?: string | null
  feed_description?: string | null
  episode_id: string | null
  episode_title: string
  publish_date: string | null
  summary_title: string | null
  summary_bullets_preview: string[]
  /** Same length as summary_bullets_preview when from current API; topic:{slug} graph hints. */
  summary_bullet_graph_topic_ids?: string[]
  summary_preview?: string | null
  gi_relative_path: string
  kg_relative_path: string
  has_gi: boolean
  has_kg: boolean
  feed_image_url?: string | null
  episode_image_url?: string | null
  duration_seconds?: number | null
  episode_number?: number | null
  feed_image_local_relpath?: string | null
  episode_image_local_relpath?: string | null
  /** Cluster-first order; prefer over summary bullets when non-empty. */
  cil_digest_topics?: CilDigestTopicPill[]
}

export type CorpusDigestTopicHit = {
  metadata_relative_path: string | null
  episode_title: string
  feed_id: string
  feed_display_title?: string | null
  feed_rss_url?: string | null
  feed_description?: string | null
  score: number | null
  summary_preview?: string | null
  episode_id?: string | null
  publish_date?: string | null
  gi_relative_path?: string
  kg_relative_path?: string
  has_gi?: boolean
  has_kg?: boolean
  feed_image_url?: string | null
  episode_image_url?: string | null
  duration_seconds?: number | null
  episode_number?: number | null
  feed_image_local_relpath?: string | null
  episode_image_local_relpath?: string | null
}

export type CorpusDigestTopicBand = {
  topic_id: string
  label: string
  query: string
  /** From API; when missing (older server), Digest falls back to episode-only graph focus. */
  graph_topic_id?: string
  hits: CorpusDigestTopicHit[]
}

export type CorpusDigestResponse = {
  path: string
  window: DigestWindow
  window_start_utc: string
  window_end_utc: string
  compact: boolean
  rows: CorpusDigestRow[]
  topics: CorpusDigestTopicBand[]
  topics_unavailable_reason: string | null
}

export type FetchDigestOptions = {
  window?: DigestWindow
  since?: string
  compact?: boolean
  includeTopics?: boolean
  maxRows?: number
}

export async function fetchCorpusDigest(
  corpusPath: string,
  options: FetchDigestOptions = {},
): Promise<CorpusDigestResponse> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  if (options.window) {
    q.set('window', options.window)
  }
  if (options.since?.trim()) {
    q.set('since', options.since.trim())
  }
  if (options.compact === true) {
    q.set('compact', 'true')
  }
  if (options.includeTopics === false) {
    q.set('include_topics', 'false')
  }
  if (options.maxRows != null) {
    q.set('max_rows', String(options.maxRows))
  }
  const qs = q.toString()
  return dedupeInFlight(`GET|/api/corpus/digest?${qs}`, async () => {
    const res = await fetchWithTimeout(`/api/corpus/digest?${qs}`)
    if (!res.ok) {
      const t = await res.text()
      raiseDigestHttpError(res, t)
    }
    return (await res.json()) as CorpusDigestResponse
  })
}
