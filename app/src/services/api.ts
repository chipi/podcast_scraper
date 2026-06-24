/**
 * Typed client for the consumer platform API (`/api/app/*`, RFC-098/RFC-099).
 *
 * The app is a thin client of this API — no other backend coupling — so the same contract
 * can later serve a native mobile client (RFC-099 §10). Requests send the session cookie
 * (`credentials: 'include'`); reads are open, per-user writes require auth.
 */

import type {
  AudioSource,
  EntitiesResponse,
  EpisodeDetail,
  EpisodesPage,
  InsightsResponse,
  ListEpisodesParams,
  Me,
  PlaybackPosition,
  SearchResponse,
  SegmentsResponse,
} from './types'

const BASE = '/api/app'

/** Raised on a non-2xx response; carries the HTTP status for callers to branch on (401 etc). */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function getJSON<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const url = new URL(`${BASE}${path}`, window.location.origin)
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, String(v))
    }
  }
  const resp = await fetch(url.toString(), {
    credentials: 'include',
    headers: { Accept: 'application/json' },
  })
  if (!resp.ok) {
    throw new ApiError(resp.status, `GET ${path} → ${resp.status}`)
  }
  return (await resp.json()) as T
}

/** Signed-in user, or `null` when not authenticated (401). */
export async function getMe(): Promise<Me | null> {
  try {
    return await getJSON<Me>('/me')
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return null
    throw err
  }
}

/** Catalog: episodes across the corpus, newest-first (paginated). */
export function listEpisodes(params: ListEpisodesParams = {}): Promise<EpisodesPage> {
  return getJSON<EpisodesPage>('/episodes', {
    page: params.page,
    page_size: params.pageSize,
    status: params.status,
    feed_id: params.feedId,
  })
}

/** Catalog: one podcast's episodes, newest-first (paginated). */
export function listPodcastEpisodes(
  feedId: string,
  params: Omit<ListEpisodesParams, 'feedId'> = {},
): Promise<EpisodesPage> {
  return getJSON<EpisodesPage>(`/podcasts/${encodeURIComponent(feedId)}/episodes`, {
    page: params.page,
    page_size: params.pageSize,
    status: params.status,
  })
}

/** Episode detail by slug. */
export function getEpisode(slug: string): Promise<EpisodeDetail> {
  return getJSON<EpisodeDetail>(`/episodes/${encodeURIComponent(slug)}`)
}

/** Transcript segments for the sync engine. */
export function getSegments(slug: string): Promise<SegmentsResponse> {
  return getJSON<SegmentsResponse>(`/episodes/${encodeURIComponent(slug)}/segments`)
}

/** Origin audio descriptor — the client plays `url` directly (bridge, never rehost). */
export function getAudioSource(slug: string): Promise<AudioSource> {
  return getJSON<AudioSource>(`/episodes/${encodeURIComponent(slug)}/audio-source`)
}

/** Grounded GIL insights for an episode (empty when no GI artifact). */
export function getInsights(slug: string): Promise<InsightsResponse> {
  return getJSON<InsightsResponse>(`/episodes/${encodeURIComponent(slug)}/insights`)
}

/** KG entities (persons/orgs/topics) for an episode (empty when no KG artifact). */
export function getEntities(slug: string): Promise<EntitiesResponse> {
  return getJSON<EntitiesResponse>(`/episodes/${encodeURIComponent(slug)}/entities`)
}

/** Episode-scoped grounded search — extractive passages, no request-time LLM (D6). */
export function searchEpisode(slug: string, q: string, topK = 8): Promise<SearchResponse> {
  return getJSON<SearchResponse>(`/episodes/${encodeURIComponent(slug)}/search`, {
    q,
    top_k: topK,
  })
}

/** Saved playback position (auth-gated); `null` when signed out or unset. */
export async function getPlayback(slug: string): Promise<PlaybackPosition | null> {
  try {
    return await getJSON<PlaybackPosition>(`/playback/${encodeURIComponent(slug)}`)
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return null
    throw err
  }
}

/** Persist the playback position (auth-gated); silently no-ops when signed out (401). */
export async function putPlayback(slug: string, positionSeconds: number): Promise<void> {
  const resp = await fetch(`${BASE}/playback/${encodeURIComponent(slug)}`, {
    method: 'PUT',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ position_seconds: positionSeconds }),
  })
  if (!resp.ok && resp.status !== 401) {
    throw new ApiError(resp.status, `PUT /playback → ${resp.status}`)
  }
}

/** The user's play queue (ordered slugs); `[]` when signed out (401). Auth-gated. */
export async function getQueue(): Promise<string[]> {
  try {
    return (await getJSON<{ items: string[] }>('/queue')).items
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return []
    throw err
  }
}

/** Replace the play queue (auth-gated); silently no-ops when signed out (401). */
export async function putQueue(items: string[]): Promise<void> {
  const resp = await fetch(`${BASE}/queue`, {
    method: 'PUT',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items }),
  })
  if (!resp.ok && resp.status !== 401) {
    throw new ApiError(resp.status, `PUT /queue → ${resp.status}`)
  }
}

/** Begin the OAuth login flow (full-page redirect; Google in prod, mock in dev/e2e). */
export function loginUrl(): string {
  return `${BASE}/auth/login`
}

/** Clear the session server-side (deletes the cookie). Best-effort; resolves on 204. */
export async function logout(): Promise<void> {
  await fetch(`${BASE}/auth/logout`, { method: 'POST', credentials: 'include' })
}
