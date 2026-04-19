import type { CilDigestTopicPill } from './digestApi'
import { dedupeInFlight } from './inFlightDedupe'
import { fetchWithTimeout } from './httpClient'

function raiseCorpusHttpError(res: Response, bodyText: string): never {
  if (res.status === 404) {
    throw new Error(
      'Corpus Library endpoint not found (404). Restart the viewer API from a current checkout (pip install -e ".[server]", then make serve-api or podcast serve). Ensure Vite proxies /api to that process (default port 8000).',
    )
  }
  throw new Error(bodyText || `HTTP ${res.status}`)
}

export type CorpusFeedItem = {
  feed_id: string
  display_title: string | null
  episode_count: number
  image_url?: string | null
  /** Verified corpus-relative path when feed art was downloaded (Phase 4). */
  image_local_relpath?: string | null
  rss_url?: string | null
  description?: string | null
}

export type CorpusFeedsResponse = {
  path: string
  feeds: CorpusFeedItem[]
}

export type CorpusEpisodeListItem = {
  metadata_relative_path: string
  feed_id: string
  /** Feed title from episode metadata when present (omitted on older APIs). */
  feed_display_title?: string | null
  feed_rss_url?: string | null
  feed_description?: string | null
  /** Summary-derived topic pills (from bullets; capped server-side). */
  topics?: string[]
  /** Same fields as digest rows — list summary line when `summary_preview` is empty. */
  summary_title?: string | null
  /** Up to four bullets (digest parity; prefer for topic pills when present). */
  summary_bullets_preview?: string[]
  /** Server recap for list rows (full text; UI wraps). When empty, client composes title + bullets. */
  summary_preview?: string | null
  episode_id: string | null
  episode_title: string
  publish_date: string | null
  feed_image_url?: string | null
  episode_image_url?: string | null
  duration_seconds?: number | null
  episode_number?: number | null
  feed_image_local_relpath?: string | null
  episode_image_local_relpath?: string | null
  cil_digest_topics?: CilDigestTopicPill[]
  gi_relative_path?: string
  kg_relative_path?: string
  has_gi?: boolean
  has_kg?: boolean
}

export type CorpusEpisodesResponse = {
  path: string
  feed_id: string | null
  items: CorpusEpisodeListItem[]
  next_cursor: string | null
}

export type CorpusEpisodeDetailResponse = {
  path: string
  metadata_relative_path: string
  feed_id: string
  feed_rss_url?: string | null
  feed_description?: string | null
  episode_id: string | null
  episode_title: string
  publish_date: string | null
  summary_title: string | null
  summary_bullets: string[]
  summary_text: string | null
  gi_relative_path: string
  kg_relative_path: string
  bridge_relative_path?: string
  has_gi: boolean
  has_kg: boolean
  has_bridge?: boolean
  feed_image_url?: string | null
  episode_image_url?: string | null
  duration_seconds?: number | null
  episode_number?: number | null
  feed_image_local_relpath?: string | null
  episode_image_local_relpath?: string | null
  cil_digest_topics?: CilDigestTopicPill[]
}

export async function fetchCorpusFeeds(corpusPath: string): Promise<CorpusFeedsResponse> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  const qs = q.toString()
  return dedupeInFlight(`GET|/api/corpus/feeds?${qs}`, async () => {
    const res = await fetchWithTimeout(`/api/corpus/feeds?${qs}`)
    if (!res.ok) {
      const t = await res.text()
      raiseCorpusHttpError(res, t)
    }
    return (await res.json()) as CorpusFeedsResponse
  })
}

export type FetchEpisodesOptions = {
  /** When set (including empty string for ungrouped episodes), filters by ``feed_id``. */
  feedId?: string
  q?: string
  /** Case-insensitive match on summary title or any summary bullet. */
  topicQ?: string
  /** When true, only episodes with at least one CIL topic in a corpus topic cluster (bridge + topic_clusters.json). */
  topicClusterOnly?: boolean
  since?: string
  limit?: number
  cursor?: string | null
}

export async function fetchCorpusEpisodes(
  corpusPath: string,
  options: FetchEpisodesOptions = {},
): Promise<CorpusEpisodesResponse> {
  const q = new URLSearchParams({ path: corpusPath.trim() })
  if (options.feedId !== undefined) {
    q.set('feed_id', options.feedId)
  }
  if (options.q?.trim()) q.set('q', options.q.trim())
  if (options.topicQ?.trim()) q.set('topic_q', options.topicQ.trim())
  if (options.topicClusterOnly) q.set('topic_cluster_only', 'true')
  if (options.since?.trim()) q.set('since', options.since.trim())
  if (options.limit != null) q.set('limit', String(options.limit))
  if (options.cursor) q.set('cursor', options.cursor)
  const qs = q.toString()
  return dedupeInFlight(`GET|/api/corpus/episodes?${qs}`, async () => {
    const res = await fetchWithTimeout(`/api/corpus/episodes?${qs}`)
    if (!res.ok) {
      const t = await res.text()
      raiseCorpusHttpError(res, t)
    }
    return (await res.json()) as CorpusEpisodesResponse
  })
}

export async function fetchCorpusEpisodeDetail(
  corpusPath: string,
  metadataRelpath: string,
): Promise<CorpusEpisodeDetailResponse> {
  const q = new URLSearchParams({
    path: corpusPath.trim(),
    metadata_relpath: metadataRelpath,
  })
  const qs = q.toString()
  return dedupeInFlight(`GET|/api/corpus/episodes/detail?${qs}`, async () => {
    const res = await fetchWithTimeout(`/api/corpus/episodes/detail?${qs}`)
    if (!res.ok) {
      const t = await res.text()
      raiseCorpusHttpError(res, t)
    }
    return (await res.json()) as CorpusEpisodeDetailResponse
  })
}

export type CorpusSimilarEpisodeItem = {
  score: number
  feed_id: string
  episode_id: string | null
  episode_title: string
  metadata_relative_path: string | null
  publish_date: string | null
  doc_type: string | null
  snippet: string
  feed_image_url?: string | null
  episode_image_url?: string | null
  duration_seconds?: number | null
  episode_number?: number | null
  feed_image_local_relpath?: string | null
  episode_image_local_relpath?: string | null
}

export type CorpusSimilarEpisodesResponse = {
  path: string
  source_metadata_relative_path: string
  query_used: string
  items: CorpusSimilarEpisodeItem[]
  error: string | null
  detail: string | null
}

export type FetchSimilarOptions = {
  topK?: number
}

export async function fetchCorpusSimilarEpisodes(
  corpusPath: string,
  metadataRelpath: string,
  options: FetchSimilarOptions = {},
): Promise<CorpusSimilarEpisodesResponse> {
  const q = new URLSearchParams({
    path: corpusPath.trim(),
    metadata_relpath: metadataRelpath,
  })
  if (options.topK != null) {
    q.set('top_k', String(options.topK))
  }
  const qs = q.toString()
  return dedupeInFlight(`GET|/api/corpus/episodes/similar?${qs}`, async () => {
    const res = await fetchWithTimeout(`/api/corpus/episodes/similar?${qs}`)
    if (!res.ok) {
      const t = await res.text()
      raiseCorpusHttpError(res, t)
    }
    return (await res.json()) as CorpusSimilarEpisodesResponse
  })
}

export type CorpusResolvedEpisodeArtifact = {
  episode_id: string
  publish_date: string | null
  gi_relative_path: string | null
  kg_relative_path: string | null
  bridge_relative_path: string | null
}

export type CorpusResolveEpisodesResponse = {
  path: string
  resolved: CorpusResolvedEpisodeArtifact[]
  missing_episode_ids: string[]
}

export async function fetchResolveEpisodeArtifacts(
  corpusPath: string,
  episodeIds: string[],
): Promise<CorpusResolveEpisodesResponse> {
  const root = corpusPath.trim()
  if (!root) {
    throw new Error('Corpus path is required')
  }
  const ids = episodeIds.map((e) => String(e).trim()).filter(Boolean)
  if (ids.length === 0) {
    throw new Error('At least one episode id is required')
  }
  const res = await fetchWithTimeout('/api/corpus/resolve-episode-artifacts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path: root, episode_ids: ids }),
  })
  if (!res.ok) {
    const t = await res.text()
    raiseCorpusHttpError(res, t)
  }
  return (await res.json()) as CorpusResolveEpisodesResponse
}

export type CorpusNodeEpisodeItem = {
  gi_relative_path: string
  kg_relative_path: string
  bridge_relative_path: string
  episode_id?: string | null
}

export type CorpusNodeEpisodesResponse = {
  path: string
  node_id: string
  episodes: CorpusNodeEpisodeItem[]
  truncated: boolean
  total_matched: number | null
}

/** Default max episodes for cross-episode graph expand (viewer → ``POST /api/corpus/node-episodes``). */
export const GRAPH_NODE_EPISODES_EXPAND_MAX = 2048

export async function fetchNodeEpisodes(
  corpusPath: string,
  nodeId: string,
  maxEpisodes?: number | null,
): Promise<CorpusNodeEpisodesResponse> {
  const root = corpusPath.trim()
  if (!root) {
    throw new Error('Corpus path is required')
  }
  const id = nodeId.trim()
  if (!id) {
    throw new Error('node_id is required')
  }
  const body: Record<string, unknown> = { path: root, node_id: id }
  if (maxEpisodes != null && maxEpisodes > 0) {
    body.max_episodes = maxEpisodes
  }
  const capKey = maxEpisodes != null && maxEpisodes > 0 ? String(maxEpisodes) : 'all'
  return dedupeInFlight(`POST|/api/corpus/node-episodes|${root}|${id}|${capKey}`, async () => {
    const res = await fetchWithTimeout('/api/corpus/node-episodes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      const t = await res.text()
      raiseCorpusHttpError(res, t)
    }
    return (await res.json()) as CorpusNodeEpisodesResponse
  })
}
