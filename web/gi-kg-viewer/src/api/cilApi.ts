/**
 * CIL cross-layer HTTP helpers (viewer).
 */

import { fetchWithTimeout } from './httpClient'

export interface CilArcEpisodeBlock {
  episode_id: string
  publish_date: string | null
  episode_title: string | null
  feed_title: string | null
  episode_number: number | null
  episode_image_url: string | null
  episode_image_local_relpath: string | null
  feed_image_url: string | null
  feed_image_local_relpath: string | null
  summary_title?: string | null
  summary_text?: string | null
  insights: Record<string, unknown>[]
}

export interface CilTopicTimelineResponse {
  path: string
  topic_id: string
  episodes: CilArcEpisodeBlock[]
}

export interface CilTopicTimelineMergedResponse {
  path: string
  topic_ids: string[]
  episodes: CilArcEpisodeBlock[]
}

export interface CilIdListResponse {
  path: string
  anchor_id: string
  ids: string[]
}

/** One corpus-wide quote evidence row from ``GET /api/persons/{id}/brief`` (#909). */
export interface CilPersonProfileQuoteRow {
  episode_id: string
  quote: Record<string, unknown>
}

/** Response for ``GET /api/persons/{id}/brief`` — corpus-wide person profile (#909). */
export interface CilPersonProfileResponse {
  path: string
  person_id: string
  topics: Record<string, Record<string, unknown>[]>
  quotes: CilPersonProfileQuoteRow[]
}

/**
 * #909 — fetch a person's corpus-wide profile: every quote they spoke across ALL
 * episodes (server-side ``person_profile`` resolves #852 cross-episode name variants),
 * independent of which episodes are currently loaded into the viewer graph.
 */
export async function fetchPersonProfile(
  corpusPath: string,
  personId: string,
): Promise<CilPersonProfileResponse> {
  const root = corpusPath.trim()
  const pid = personId.trim()
  if (!root) {
    throw new Error('Corpus path is required')
  }
  if (!pid) {
    throw new Error('Person id is required')
  }
  const enc = encodeURIComponent(pid)
  const q = new URLSearchParams({ path: root })
  const res = await fetchWithTimeout(`/api/persons/${enc}/brief?${q.toString()}`)
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail.trim() || `HTTP ${res.status}`)
  }
  return (await res.json()) as CilPersonProfileResponse
}

export async function fetchTopicTimeline(
  corpusPath: string,
  topicId: string,
  opts?: { insightTypes?: string | null },
): Promise<CilTopicTimelineResponse> {
  const root = corpusPath.trim()
  const tid = topicId.trim()
  if (!root) {
    throw new Error('Corpus path is required')
  }
  if (!tid) {
    throw new Error('Topic id is required')
  }
  const enc = encodeURIComponent(tid)
  const q = new URLSearchParams({ path: root })
  const it = opts?.insightTypes?.trim()
  if (it) {
    q.set('insight_types', it)
  }
  const res = await fetchWithTimeout(`/api/topics/${enc}/timeline?${q.toString()}`)
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail.trim() || `HTTP ${res.status}`)
  }
  return (await res.json()) as CilTopicTimelineResponse
}

export interface CilTopicPerspective {
  person_id: string
  person_name: string
  insight_count: number
  episode_count: number
  insights: Array<Record<string, unknown>>
}

export interface CilTopicPerspectivesResponse {
  path: string
  topic_id: string
  perspectives: CilTopicPerspective[]
}

/** Each speaker's grounded insights on a topic, grouped by speaker (#1146). */
export async function fetchTopicPerspectives(
  corpusPath: string,
  topicId: string,
): Promise<CilTopicPerspectivesResponse> {
  const root = corpusPath.trim()
  const tid = topicId.trim()
  if (!root) {
    throw new Error('Corpus path is required')
  }
  if (!tid) {
    throw new Error('Topic id is required')
  }
  const enc = encodeURIComponent(tid)
  const q = new URLSearchParams({ path: root })
  const res = await fetchWithTimeout(`/api/topics/${enc}/perspectives?${q.toString()}`)
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail.trim() || `HTTP ${res.status}`)
  }
  return (await res.json()) as CilTopicPerspectivesResponse
}

export async function fetchTopicTimelineMerged(
  corpusPath: string,
  topicIds: string[],
  opts?: { insightTypes?: string | null },
): Promise<CilTopicTimelineMergedResponse> {
  const root = corpusPath.trim()
  if (!root) {
    throw new Error('Corpus path is required')
  }
  const ids = topicIds.map((t) => String(t).trim()).filter(Boolean)
  if (ids.length === 0) {
    throw new Error('At least one topic id is required')
  }
  const body: Record<string, unknown> = {
    topic_ids: ids,
    path: root,
  }
  const it = opts?.insightTypes?.trim()
  if (it) {
    body.insight_types = it
  }
  const res = await fetchWithTimeout('/api/topics/timeline', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail.trim() || `HTTP ${res.status}`)
  }
  return (await res.json()) as CilTopicTimelineMergedResponse
}

export async function fetchTopicPersons(
  corpusPath: string,
  topicId: string,
): Promise<CilIdListResponse> {
  const root = corpusPath.trim()
  const tid = topicId.trim()
  if (!root) {
    throw new Error('Corpus path is required')
  }
  if (!tid) {
    throw new Error('Topic id is required')
  }
  const enc = encodeURIComponent(tid)
  const q = new URLSearchParams({ path: root })
  const res = await fetchWithTimeout(`/api/topics/${enc}/persons?${q.toString()}`)
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail.trim() || `HTTP ${res.status}`)
  }
  return (await res.json()) as CilIdListResponse
}
