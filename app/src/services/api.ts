/**
 * Typed client for the consumer platform API (`/api/app/*`, RFC-098/RFC-099).
 *
 * The app is a thin client of this API — no other backend coupling — so the same contract
 * can later serve a native mobile client (RFC-099 §10). Requests send the session cookie
 * (`credentials: 'include'`); reads are open, per-user writes require auth.
 */

import type {
  AudioSource,
  CorpusEnrichmentSignals,
  EntitiesResponse,
  EntitySearchResponse,
  EpisodeEnrichmentSignals,
  EpisodeDetail,
  EpisodesPage,
  EpisodeStats,
  FavoriteAdd,
  FavoritesResponse,
  Highlight,
  HighlightCreate,
  HighlightUpdate,
  InsightsResponse,
  InterestCluster,
  ListEpisodesParams,
  Me,
  Note,
  NoteCreate,
  NoteUpdate,
  PersonCard,
  PlaybackPosition,
  Podcast,
  ResurfacingResponse,
  ResurfacingSettings,
  SearchResponse,
  SegmentsResponse,
  TopicCard,
  UserStats,
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

/** "More like this" — semantic peer episodes; empty page when the index is unavailable. */
export function getRelated(slug: string, topK = 6): Promise<EpisodesPage> {
  return getJSON<EpisodesPage>(`/episodes/${encodeURIComponent(slug)}/related`, { top_k: topK })
}

/** Person profile card — appears-in episodes + related people/topics (KG co-occurrence). */
export function getPersonCard(id: string, scope?: 'all' | 'mine'): Promise<PersonCard> {
  // scope='mine' = the guest across the episodes the signed-in user has heard (P3 #1122).
  return getJSON<PersonCard>(`/persons/${encodeURIComponent(id)}`, { scope })
}

/** Topic card — episodes-about + cluster siblings + related people (KG-grounded). */
export function getTopicCard(id: string, scope?: 'all' | 'mine'): Promise<TopicCard> {
  return getJSON<TopicCard>(`/topics/${encodeURIComponent(id)}`, { scope })
}

// Corpus-scope enrichment is one static payload for the whole corpus, read by
// every entity card — fetch it once per session and share the promise. On
// failure the cache is cleared so a later card can retry.
let _corpusEnrichment: Promise<CorpusEnrichmentSignals> | null = null
/** Corpus-scope enrichment signals (RFC-088) — grounding / co-appearance /
 *  contradictions / velocity / similarity / co-occurrence, keyed by enricher id. */
export function getCorpusEnrichment(): Promise<CorpusEnrichmentSignals> {
  if (!_corpusEnrichment) {
    _corpusEnrichment = getJSON<{ signals: CorpusEnrichmentSignals }>('/corpus/enrichment')
      .then((r) => r.signals ?? {})
      .catch((err) => {
        _corpusEnrichment = null
        throw err
      })
  }
  return _corpusEnrichment
}

// Per-episode enrichment (currently insight_density) — cached per slug so
// re-opening the panel doesn't refetch. Cleared on failure so it can retry.
const _episodeEnrichment = new Map<string, Promise<EpisodeEnrichmentSignals>>()
/** Per-episode enrichment signals (RFC-088 episode-scope, e.g. insight_density). */
export function getEpisodeEnrichment(slug: string): Promise<EpisodeEnrichmentSignals> {
  let p = _episodeEnrichment.get(slug)
  if (!p) {
    p = getJSON<{ signals: EpisodeEnrichmentSignals }>(
      `/episodes/${encodeURIComponent(slug)}/enrichment`,
    )
      .then((r) => r.signals ?? {})
      .catch((err) => {
        _episodeEnrichment.delete(slug)
        throw err
      })
    _episodeEnrichment.set(slug, p)
  }
  return p
}

/** Corpus-wide grounded search (Home "Ask your library"); empty when no index. */
export function searchCorpus(
  q: string,
  topK = 12,
  scope?: 'all' | 'mine',
): Promise<SearchResponse> {
  // scope='mine' = grounded recall over the signed-in user's heard∪captured corpus (P3 #1120).
  return getJSON<SearchResponse>('/search', { q, top_k: topK, scope })
}

/** Resolve a query to a person/topic card (exact/near-exact); `entity: null` when none. */
export function resolveEntity(q: string): Promise<EntitySearchResponse> {
  return getJSON<EntitySearchResponse>('/entities/search', { q })
}

/** Home discovery feed — interest-ranked when enabled + signed-in, else recency (the default). */
export function getDiscover(limit = 8): Promise<EpisodesPage> {
  return getJSON<EpisodesPage>('/discover', { limit })
}

/** Top interest clusters for the picker, by corpus prevalence. */
export async function getTopClusters(limit = 12): Promise<InterestCluster[]> {
  return (await getJSON<{ items: InterestCluster[] }>('/clusters', { limit })).items
}

/** The signed-in user's interest cluster ids; `[]` when signed out (401). Auth-gated. */
export async function getUserInterests(): Promise<string[]> {
  try {
    return (await getJSON<{ items: string[] }>('/interests')).items
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return []
    throw err
  }
}

/** The user's favorites grouped by kind; `{episodes:[],insights:[]}` when signed out (401). */
export async function getFavorites(): Promise<FavoritesResponse> {
  try {
    return await getJSON<FavoritesResponse>('/favorites')
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return { episodes: [], insights: [] }
    throw err
  }
}

/** Save an item (auth-gated); returns the updated favorites. */
export async function addFavorite(item: FavoriteAdd): Promise<FavoritesResponse> {
  const resp = await fetch(`${BASE}/favorites`, {
    method: 'PUT',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(item),
  })
  if (!resp.ok) throw new ApiError(resp.status, `PUT /favorites → ${resp.status}`)
  return (await resp.json()) as FavoritesResponse
}

/** Remove a saved item by kind+ref (auth-gated); returns the updated favorites. */
export async function removeFavorite(kind: string, ref: string): Promise<FavoritesResponse> {
  const resp = await fetch(
    `${BASE}/favorites/${encodeURIComponent(kind)}/${encodeURIComponent(ref)}`,
    { method: 'DELETE', credentials: 'include' },
  )
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /favorites → ${resp.status}`)
  return (await resp.json()) as FavoritesResponse
}

/** Follow one interest token — cluster (`tc:`), topic (`topic:`) or person (`person:`). Auth-gated. */
export async function addInterest(token: string): Promise<string[]> {
  const resp = await fetch(`${BASE}/interests/${encodeURIComponent(token)}`, {
    method: 'POST',
    credentials: 'include',
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /interests → ${resp.status}`)
  return ((await resp.json()) as { items: string[] }).items
}

/** Unfollow one interest token (auth-gated); returns the remaining list. */
export async function removeInterest(token: string): Promise<string[]> {
  const resp = await fetch(`${BASE}/interests/${encodeURIComponent(token)}`, {
    method: 'DELETE',
    credentials: 'include',
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /interests → ${resp.status}`)
  return ((await resp.json()) as { items: string[] }).items
}

/** Replace the user's interest cluster ids (auth-gated); returns the stored list. */
export async function putUserInterests(clusterIds: string[]): Promise<string[]> {
  const resp = await fetch(`${BASE}/interests`, {
    method: 'PUT',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items: clusterIds }),
  })
  if (!resp.ok) {
    throw new ApiError(resp.status, `PUT /interests → ${resp.status}`)
  }
  return ((await resp.json()) as { items: string[] }).items
}

/** Shows in the user's library (Home "Your shows"). */
export async function getPodcasts(): Promise<Podcast[]> {
  return (await getJSON<{ items: Podcast[] }>('/podcasts')).items
}

/** Saved playback positions, newest-first (Home "Continue"); `[]` when signed out. */
export async function getPlaybackList(): Promise<PlaybackPosition[]> {
  try {
    return (await getJSON<{ items: PlaybackPosition[] }>('/playback')).items
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return []
    throw err
  }
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

/** Record that the user opened an episode (listen-event log, ). Best-effort; ignores 401. */
export async function logListen(slug: string): Promise<void> {
  try {
    const resp = await fetch(`${BASE}/listen/${encodeURIComponent(slug)}`, {
      method: 'POST',
      credentials: 'include',
    })
    if (!resp.ok && resp.status !== 401) {
      throw new ApiError(resp.status, `POST /listen → ${resp.status}`)
    }
  } catch {
    /* analytics is best-effort — never surface to the listener */
  }
}

/** The signed-in user's own listening analytics; `null` when signed out (401). Auth-gated. */
export async function getMyStats(): Promise<UserStats | null> {
  try {
    return await getJSON<UserStats>('/me/stats')
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return null
    throw err
  }
}

/** Cross-user reach for one episode (public; anonymous aggregate counts). */
export async function getEpisodeStats(slug: string): Promise<EpisodeStats> {
  return getJSON<EpisodeStats>(`/episodes/${encodeURIComponent(slug)}/stats`)
}

/** Begin the OAuth login flow (full-page redirect; Google in prod, mock in dev/e2e). */
export function loginUrl(as?: string): string {
  return as ? `${BASE}/auth/login?as=${encodeURIComponent(as)}` : `${BASE}/auth/login`
}

export interface DevUser {
  hint: string
  name: string
  role: string
}

/**
 * Predefined dev identities for the sign-in picker — populated only when the MOCK provider is on.
 * Never throws: any failure → `{ enabled: false }` (the UI shows the normal sign-in button).
 */
export async function getDevUsers(): Promise<{ enabled: boolean; users: DevUser[] }> {
  try {
    const res = await fetch(`${BASE}/auth/dev-users`, { credentials: 'include' })
    if (!res.ok) return { enabled: false, users: [] }
    const body = (await res.json()) as { enabled?: boolean; users?: DevUser[] }
    return { enabled: body.enabled === true, users: Array.isArray(body.users) ? body.users : [] }
  } catch {
    return { enabled: false, users: [] }
  }
}

/** Clear the session server-side (deletes the cookie). Best-effort; resolves on 204. */
export async function logout(): Promise<void> {
  await fetch(`${BASE}/auth/logout`, { method: 'POST', credentials: 'include' })
}

// --- P2 Capture: highlights + notes (PRD-040 / RFC-098 §7) ---

/** The user's highlights, optionally scoped to one episode; `[]` when signed out (401). */
export async function getHighlights(episode?: string): Promise<Highlight[]> {
  try {
    return (await getJSON<{ items: Highlight[] }>('/highlights', { episode })).items
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return []
    throw err
  }
}

/** Capture a highlight (auth-gated); returns the created record. */
export async function createHighlight(body: HighlightCreate): Promise<Highlight> {
  const resp = await fetch(`${BASE}/highlights`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /highlights → ${resp.status}`)
  return (await resp.json()) as Highlight
}

/** Edit a highlight's colour / captured text (auth-gated); returns the updated record. */
export async function patchHighlight(id: string, body: HighlightUpdate): Promise<Highlight> {
  const resp = await fetch(`${BASE}/highlights/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!resp.ok) throw new ApiError(resp.status, `PATCH /highlights → ${resp.status}`)
  return (await resp.json()) as Highlight
}

/** Remove a highlight by id (auth-gated); returns the remaining list. */
export async function deleteHighlight(id: string): Promise<Highlight[]> {
  const resp = await fetch(`${BASE}/highlights/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    credentials: 'include',
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /highlights → ${resp.status}`)
  return ((await resp.json()) as { items: Highlight[] }).items
}

/** The user's notes, optionally scoped to one target; `[]` when signed out (401). */
export async function getNotes(target?: string, targetId?: string): Promise<Note[]> {
  try {
    return (
      await getJSON<{ items: Note[] }>('/notes', { target, target_id: targetId })
    ).items
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return []
    throw err
  }
}

/** Attach a free-text note to a highlight / insight / episode (auth-gated). */
export async function createNote(body: NoteCreate): Promise<Note> {
  const resp = await fetch(`${BASE}/notes`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /notes → ${resp.status}`)
  return (await resp.json()) as Note
}

/** Edit a note's text (auth-gated); returns the updated record. */
export async function patchNote(id: string, text: string): Promise<Note> {
  const body: NoteUpdate = { text }
  const resp = await fetch(`${BASE}/notes/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!resp.ok) throw new ApiError(resp.status, `PATCH /notes → ${resp.status}`)
  return (await resp.json()) as Note
}

/** Remove a note by id (auth-gated); returns the remaining list. */
export async function deleteNote(id: string): Promise<Note[]> {
  const resp = await fetch(`${BASE}/notes/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    credentials: 'include',
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /notes → ${resp.status}`)
  return ((await resp.json()) as { items: Note[] }).items
}

/** The URL for the Markdown export of all highlights (a download link / new tab). */
export function highlightsExportUrl(): string {
  return `${BASE}/highlights/export.md`
}

// --- P3 Consolidation: spaced resurfacing (RFC-101 §5) ---

/** Highlights due to resurface (+ reflection prompt + paused flag); empty signed out (401). */
export async function getResurfacing(): Promise<ResurfacingResponse> {
  try {
    return await getJSON<ResurfacingResponse>('/resurfacing')
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return { items: [], paused: false }
    throw err
  }
}

/** Record that the user has seen a resurfaced highlight (advances its ladder). Best-effort. */
export async function markSurfaced(id: string): Promise<void> {
  const resp = await fetch(`${BASE}/resurfacing/${encodeURIComponent(id)}/surfaced`, {
    method: 'POST',
    credentials: 'include',
  })
  if (!resp.ok && resp.status !== 401) {
    throw new ApiError(resp.status, `POST /resurfacing/surfaced → ${resp.status}`)
  }
}

/** Update resurfacing pacing (pause/resume); returns the stored settings. */
export async function putResurfacingSettings(paused: boolean): Promise<ResurfacingSettings> {
  const resp = await fetch(`${BASE}/resurfacing/settings`, {
    method: 'PUT',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paused }),
  })
  if (!resp.ok) throw new ApiError(resp.status, `PUT /resurfacing/settings → ${resp.status}`)
  return (await resp.json()) as ResurfacingSettings
}
