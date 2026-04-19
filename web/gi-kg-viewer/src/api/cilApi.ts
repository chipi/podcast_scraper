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
