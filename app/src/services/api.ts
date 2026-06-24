/**
 * Typed client for the consumer platform API (`/api/app/*`, RFC-098/RFC-099).
 *
 * The app is a thin client of this API — no other backend coupling — so the same contract
 * can later serve a native mobile client (RFC-099 §10). Requests send the session cookie
 * (`credentials: 'include'`); reads are open, per-user writes require auth.
 */

import type {
  EpisodeDetail,
  EpisodesPage,
  ListEpisodesParams,
  Me,
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

/** Begin the OAuth login flow (full-page redirect; Google in prod, mock in dev/e2e). */
export function loginUrl(): string {
  return `${BASE}/auth/login`
}
