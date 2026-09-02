/**
 * Typed client for the consumer platform API (`/api/app/*`, RFC-098/RFC-099).
 *
 * The app is a thin client of this API — no other backend coupling — so the same contract
 * can later serve a native mobile client (RFC-099 §10). Requests send the session cookie
 * (`credentials: 'include'`); reads are open, per-user writes require auth.
 */

import type {
  AudioSource,
  Collection,
  CollectionDetail,
  CollectionItemRef,
  CommsSettings,
  CommsUpdate,
  CorpusEnrichmentSignals,
  EntitiesResponse,
  EntitySearchResponse,
  EpisodeEnrichmentSignals,
  TrendingTopicsResponse,
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
  LibraryItem,
  ListEpisodesParams,
  McpConnection,
  McpConnectionConfig,
  McpTokenCreated,
  McpTokenMeta,
  Me,
  Note,
  NoteCreate,
  NoteUpdate,
  PersonCard,
  PlaybackPosition,
  Podcast,
  PodcastSignals,
  ResurfacingResponse,
  ResurfacingSettings,
  SearchResponse,
  SegmentsResponse,
  Storyline,
  TopicCard,
  TopicConversationArcResponse,
  TopicPerspectivesResponse,
  TrendingEntity,
  UserStats,
  YourWeekResponse,
} from './types'
import { resolveApiBase, resolveGateAuthHeader, resolveMediaUrl } from './tier'

// API base, resolved once at load (#1305/#1310):
//   - web: origin-relative '/api/app' (or a baked VITE_API_BASE_URL).
//   - native prod/release: the live player API (or baked VITE_API_BASE_URL).
//   - native dev (internal build + dev tier): the local machine (make serve-app :5174).
// The dev↔prod switch (services/tier.ts) reloads the app on change so this re-resolves. Every call
// site does `${BASE}${path}` → when BASE is absolute both `new URL(str, origin)` and `fetch(str)`
// ignore the origin, so no other change is needed.
const BASE = resolveApiBase()

/** Raised on a non-2xx response; carries the HTTP status for callers to branch on (401 etc). */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

// Native-shell bearer token (#1310). On the web, auth rides the session cookie (`credentials:
// 'include'`) and this stays null. In the Capacitor shell the OAuth completes in an external
// browser whose cookie the WebView can't see, so we carry the SAME signed session token here and
// send it as `Authorization: Bearer` on every request. Set from the OAuth deep-link callback
// (services/native.ts) and rehydrated from Preferences on launch.
let authToken: string | null = null
export function setAuthToken(token: string | null): void {
  authToken = token
}
export function getAuthToken(): string | null {
  return authToken
}

/**
 * `fetch` wrapper that adds the bearer token when present (native) and keeps every caller's
 * `credentials: 'include'` cookie path intact (web). Callers' own headers win over the injected one.
 */
function apiFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers)
  if (!headers.has('Authorization')) {
    // User session (native OAuth) wins; else the prod coming-soon gate's Basic-auth fallback so open
    // reads reach the gated API pre-launch (services/tier.ts :: resolveGateAuthHeader). Both use the
    // `Authorization` header, so they're mutually exclusive — acceptable until native login lands.
    if (authToken) headers.set('Authorization', `Bearer ${authToken}`)
    else {
      const gate = resolveGateAuthHeader()
      if (gate) headers.set('Authorization', gate)
    }
  }
  return fetch(input, { ...init, headers })
}

async function getJSON<T>(
  path: string,
  params?: Record<string, string | number | undefined>
): Promise<T> {
  const url = new URL(`${BASE}${path}`, window.location.origin)
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, String(v))
    }
  }
  const resp = await apiFetch(url.toString(), {
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
  params: Omit<ListEpisodesParams, 'feedId'> = {}
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
export async function getAudioSource(slug: string, validate = false): Promise<AudioSource> {
  // `validate` HEADs the origin (an extra network call) and is what populates `content_length`.
  // The download path asks for it so a transfer that cannot fit is refused BEFORE it starts;
  // playback does not, because it would put a round trip in front of every play.
  const src = await getJSON<AudioSource>(
    `/episodes/${encodeURIComponent(slug)}/audio-source${validate ? '?validate=true' : ''}`,
  )
  // The bridge can hand back a RELATIVE media url (it does for the fixture corpus). On native that
  // resolves against capacitor://localhost and playback fails silently, so absolutise it here —
  // one place, rather than at each of the three consumers.
  return { ...src, url: resolveMediaUrl(src.url) ?? src.url }
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
// The player view and its embedded KnowledgePanel both ask for the same episode's related list
// (top_k=6) on load, which fired the `/related` vector search twice per open. Memoize per
// (slug, topK) so concurrent callers share one in-flight request; cleared on failure to allow retry.
const _related = new Map<string, Promise<EpisodesPage>>()
export function getRelated(slug: string, topK = 6): Promise<EpisodesPage> {
  const key = `${slug}:${topK}`
  let p = _related.get(key)
  if (!p) {
    p = getJSON<EpisodesPage>(`/episodes/${encodeURIComponent(slug)}/related`, {
      top_k: topK,
    }).catch((err) => {
      _related.delete(key)
      throw err
    })
    _related.set(key, p)
  }
  return p
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

/** Topic perspectives — each speaker's grounded insights on the topic (#1146). */
export function getTopicPerspectives(
  id: string,
  scope?: 'all' | 'mine'
): Promise<TopicPerspectivesResponse> {
  return getJSON<TopicPerspectivesResponse>(`/topics/${encodeURIComponent(id)}/perspectives`, {
    scope,
  })
}

/** Topic conversation arc — weekly volume × sentiment, the aggregate-first overview (ADR-108). */
export function getTopicConversationArc(id: string): Promise<TopicConversationArcResponse> {
  return getJSON<TopicConversationArcResponse>(`/topics/${encodeURIComponent(id)}/conversation-arc`)
}

// Corpus-scope enrichment is one static payload for the whole corpus, read by
// every entity card — fetch it once per session and share the promise. On
// failure the cache is cleared so a later card can retry.
let _corpusEnrichment: Promise<CorpusEnrichmentSignals> | null = null
/** Corpus-scope enrichment signals (RFC-088) — grounding / co-appearance /
 *  velocity / similarity / co-occurrence, keyed by enricher id. */
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

// The Home trending rail's own lean endpoint — server-side top-N rising topics
// (#perf). Replaces reading the full ~25 MB corpus-enrichment payload just to
// render ~12 rows. Cached once per session; cleared on failure so it can retry.
let _trendingTopics: Promise<TrendingTopicsResponse> | null = null
/** Top-N rising topics for the Home trending rail (already filtered + sorted server-side). */
export function getTrendingTopics(): Promise<TrendingTopicsResponse> {
  if (!_trendingTopics) {
    _trendingTopics = getJSON<TrendingTopicsResponse>('/corpus/trending-topics').catch((err) => {
      _trendingTopics = null
      throw err
    })
  }
  return _trendingTopics
}

// Per-entity corpus signals for the entity card — the corpus-enrichment lists
// pre-filtered server-side to the focused person/topic (#perf), so the card
// fetches a few KB instead of the whole corpus. Cached per `${kind}:${id}`.
const _entitySignals = new Map<string, Promise<CorpusEnrichmentSignals>>()
/** Corpus enrichment signals filtered to one entity (same shape as getCorpusEnrichment). */
export function getEntitySignals(
  kind: 'person' | 'topic',
  id: string
): Promise<CorpusEnrichmentSignals> {
  const key = `${kind}:${id}`
  let p = _entitySignals.get(key)
  if (!p) {
    p = getJSON<{ signals: CorpusEnrichmentSignals }>(
      `/corpus/entity-signals?kind=${kind}&id=${encodeURIComponent(id)}`
    )
      .then((r) => r.signals ?? {})
      .catch((err) => {
        _entitySignals.delete(key)
        throw err
      })
    _entitySignals.set(key, p)
  }
  return p
}

// Per-episode enrichment (currently insight_density) — cached per slug so
// re-opening the panel doesn't refetch. Cleared on failure so it can retry.
const _episodeEnrichment = new Map<string, Promise<EpisodeEnrichmentSignals>>()
/** Per-episode enrichment signals (RFC-088 episode-scope, e.g. insight_density). */
export function getEpisodeEnrichment(slug: string): Promise<EpisodeEnrichmentSignals> {
  let p = _episodeEnrichment.get(slug)
  if (!p) {
    p = getJSON<{ signals: EpisodeEnrichmentSignals }>(
      `/episodes/${encodeURIComponent(slug)}/enrichment`
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
  enrichResults?: boolean
): Promise<SearchResponse> {
  // scope='mine' = grounded recall over the signed-in user's heard∪captured corpus (P3 #1120).
  // enrichResults=true asks the server to decorate hits with
  //   metadata.query_enrichments.related_topics (RFC-088, #1261-1). Chain failures on
  //   the server are swallowed — the client should tolerate hits without the field.
  return getJSON<SearchResponse>('/search', {
    q,
    top_k: topK,
    scope,
    enrich_results: enrichResults ? 'true' : undefined,
  })
}

/** Resolve a query to a person/topic card (exact/near-exact); `entity: null` when none. */
export function resolveEntity(q: string): Promise<EntitySearchResponse> {
  return getJSON<EntitySearchResponse>('/entities/search', { q })
}

/** Home discovery feed — interest-ranked when enabled + signed-in, else recency (the default). */
export function getDiscover(limit = 8): Promise<EpisodesPage> {
  return getJSON<EpisodesPage>('/discover', { limit })
}

/** Fire-and-forget: log a click on a discovery-feed episode (its shown rank position) for
 *  ranking telemetry (#11). Silent no-op when signed out or on any network error. */
export function recordDiscoverClick(slug: string, position: number): void {
  void apiFetch(`${BASE}/discover/click`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ slug, position }),
  }).catch(() => {})
}

/** Top interest clusters for the picker, by corpus prevalence. */
export async function getTopClusters(limit = 12): Promise<InterestCluster[]> {
  return (await getJSON<{ items: InterestCluster[] }>('/clusters', { limit })).items
}

/** Top storylines (theme clusters — topics discussed together) for the Home rail + picker. */
export async function getStorylines(limit = 12): Promise<Storyline[]> {
  return (await getJSON<{ items: Storyline[] }>('/theme-clusters', { limit })).items
}

/** Trending entities of a kind (RFC-103 momentum), corpus-wide or the signed-in user's ('mine'). */
/** Trend window presets (RFC-103 R2) — the recent bucket the velocity is measured over. */
export type TrendWindow = '1m' | '3m' | '6m' | '1y'

export async function getTrending(
  kind: string,
  scope: 'corpus' | 'mine' = 'corpus',
  limit = 12,
  window: TrendWindow = '3m'
): Promise<TrendingEntity[]> {
  return (await getJSON<{ items: TrendingEntity[] }>('/trending', { kind, scope, limit, window }))
    .items
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
  const resp = await apiFetch(`${BASE}/favorites`, {
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
  const resp = await apiFetch(
    `${BASE}/favorites/${encodeURIComponent(kind)}/${encodeURIComponent(ref)}`,
    { method: 'DELETE', credentials: 'include' }
  )
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /favorites → ${resp.status}`)
  return (await resp.json()) as FavoritesResponse
}

/** Follow one interest token — cluster (`tc:`), topic (`topic:`) or person (`person:`). Auth-gated. */
export async function addInterest(token: string): Promise<string[]> {
  const resp = await apiFetch(`${BASE}/interests/${encodeURIComponent(token)}`, {
    method: 'POST',
    credentials: 'include',
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /interests → ${resp.status}`)
  return ((await resp.json()) as { items: string[] }).items
}

/** Unfollow one interest token (auth-gated); returns the remaining list. */
export async function removeInterest(token: string): Promise<string[]> {
  const resp = await apiFetch(`${BASE}/interests/${encodeURIComponent(token)}`, {
    method: 'DELETE',
    credentials: 'include',
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /interests → ${resp.status}`)
  return ((await resp.json()) as { items: string[] }).items
}

/** Replace the user's interest cluster ids (auth-gated); returns the stored list. */
export async function putUserInterests(clusterIds: string[]): Promise<string[]> {
  const resp = await apiFetch(`${BASE}/interests`, {
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

/** Distinct shows in the corpus (public, not per-user). */
export async function getPodcasts(): Promise<Podcast[]> {
  return (await getJSON<{ items: Podcast[] }>('/podcasts')).items
}

// --- Feed subscriptions ("follow a show") — the library the Your Week digest reads for its
// "new in your follows" section. NOT the same store as interests (topic:/person: tokens), which
// feed "Recommended for you".

/** The user's followed shows (auth-gated); `[]` when signed out. */
export async function getLibrary(): Promise<LibraryItem[]> {
  try {
    return (await getJSON<{ items: LibraryItem[] }>('/library')).items
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return []
    throw err
  }
}

/** Follow a show (idempotent on feed_id, auth-gated); returns the updated library. */
export async function followShow(
  feedId: string,
  meta: { feedUrl?: string | null; title?: string | null } = {}
): Promise<LibraryItem[]> {
  const resp = await apiFetch(`${BASE}/library`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      feed_id: feedId,
      ...(meta.feedUrl != null ? { feed_url: meta.feedUrl } : {}),
      ...(meta.title != null ? { title: meta.title } : {}),
    }),
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /library → ${resp.status}`)
  return ((await resp.json()) as { items: LibraryItem[] }).items
}

/** Unfollow a show (no-op if absent, auth-gated); returns the remaining library. */
export async function unfollowShow(feedId: string): Promise<LibraryItem[]> {
  const resp = await apiFetch(`${BASE}/library/${encodeURIComponent(feedId)}`, {
    method: 'DELETE',
    credentials: 'include',
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /library → ${resp.status}`)
  return ((await resp.json()) as { items: LibraryItem[] }).items
}

/** Show-level signals for a show page: topics/themes it's about, who's on it, what's trending. */
export function getPodcastSignals(feedId: string, topK?: number): Promise<PodcastSignals> {
  const q = topK != null ? `?top_k=${topK}` : ''
  return getJSON<PodcastSignals>(`/podcasts/${encodeURIComponent(feedId)}/signals${q}`)
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

/** Persist the playback position (auth-gated); silently no-ops when signed out (401).
 *
 * `keepalive` so the save fired from `pagehide` actually leaves the machine: a normal fetch is
 * cancelled when the document goes away, which is precisely the case that save exists for.
 */
export async function putPlayback(
  slug: string,
  positionSeconds: number,
  finished = false,
  clientTs?: number
): Promise<void> {
  const resp = await apiFetch(`${BASE}/playback/${encodeURIComponent(slug)}`, {
    method: 'PUT',
    credentials: 'include',
    keepalive: true,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      position_seconds: positionSeconds,
      finished,
      ...(clientTs ? { client_ts: clientTs } : {}),
    }),
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
  const resp = await apiFetch(`${BASE}/queue`, {
    method: 'PUT',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items }),
  })
  if (!resp.ok && resp.status !== 401) {
    throw new ApiError(resp.status, `PUT /queue → ${resp.status}`)
  }
}

/**
 * Queue ONE episode, optionally right after another ("play next"). Returns the queue the server
 * now holds.
 *
 * Item-level on purpose (#1925): `putQueue` sends the whole list, so a write made offline and
 * replayed later is last-writer-wins over anything another device did in between — which is why
 * the store refuses to write a queue it restored from cache. This is idempotent, so it can go
 * through the outbox instead.
 */
export async function addQueueItem(slug: string, after?: string | null): Promise<string[]> {
  const resp = await apiFetch(`${BASE}/queue/items`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ slug, after: after ?? null }),
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /queue/items → ${resp.status}`)
  return ((await resp.json()) as { items: string[] }).items
}

/** Remove ONE episode from the queue. Idempotent, so a replay cannot fail. */
export async function removeQueueItem(slug: string): Promise<string[]> {
  const resp = await apiFetch(`${BASE}/queue/items/${encodeURIComponent(slug)}`, {
    method: 'DELETE',
    credentials: 'include',
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /queue/items → ${resp.status}`)
  return ((await resp.json()) as { items: string[] }).items
}

/**
 * Record that the user STARTED an episode (listen-event log). Best-effort; ignores 401.
 *
 * Returns whether it landed, so the caller can queue it for a later flush (#1924) — it used to
 * swallow every failure indistinguishably, which is why offline listening vanished. `clientTs`
 * carries when the listen actually happened for events flushed after the fact; the server clamps
 * it, so a wrong device clock cannot write into the far past or the future.
 */
export async function logListen(slug: string, clientTs?: number): Promise<boolean> {
  try {
    const resp = await apiFetch(`${BASE}/listen/${encodeURIComponent(slug)}`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(clientTs ? { client_ts: clientTs } : {}),
    })
    // ANY response is an answer. A 401 means signed out, a 404 means the episode is gone —
    // neither improves by retrying, and treating them as retryable wedged every queued listen
    // behind them forever. Only 408/429 and 5xx are worth another attempt.
    if (resp.ok) return true
    if (resp.status === 408 || resp.status === 429 || resp.status >= 500) return false
    return true
  } catch {
    return false
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
export function loginUrl(as?: string, native = false): string {
  const params = new URLSearchParams()
  if (as) params.set('as', as)
  // Native (#1310): tells the backend to return the signed token via the app's deep link instead of
  // setting a cookie (which an external OAuth browser can't hand back to the WebView).
  if (native) params.set('platform', 'native')
  // Absolute base on native, so build a full URL the external browser can open.
  const base = `${BASE}/auth/login`
  const qs = params.toString()
  return qs ? `${base}?${qs}` : base
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
    const res = await apiFetch(`${BASE}/auth/dev-users`, { credentials: 'include' })
    if (!res.ok) return { enabled: false, users: [] }
    const body = (await res.json()) as { enabled?: boolean; users?: DevUser[] }
    return { enabled: body.enabled === true, users: Array.isArray(body.users) ? body.users : [] }
  } catch {
    return { enabled: false, users: [] }
  }
}

/** Clear the session server-side (deletes the cookie). Best-effort; resolves on 204. */
export async function logout(): Promise<void> {
  await apiFetch(`${BASE}/auth/logout`, { method: 'POST', credentials: 'include' })
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
  const resp = await apiFetch(`${BASE}/highlights`, {
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
  const resp = await apiFetch(`${BASE}/highlights/${encodeURIComponent(id)}`, {
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
  const resp = await apiFetch(`${BASE}/highlights/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    credentials: 'include',
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /highlights → ${resp.status}`)
  return ((await resp.json()) as { items: Highlight[] }).items
}

/** The user's notes, optionally scoped to one target; `[]` when signed out (401). */
export async function getNotes(target?: string, targetId?: string): Promise<Note[]> {
  try {
    return (await getJSON<{ items: Note[] }>('/notes', { target, target_id: targetId })).items
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return []
    throw err
  }
}

/** Attach a free-text note to a highlight / insight / episode (auth-gated). */
export async function createNote(body: NoteCreate): Promise<Note> {
  const resp = await apiFetch(`${BASE}/notes`, {
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
  const resp = await apiFetch(`${BASE}/notes/${encodeURIComponent(id)}`, {
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
  const resp = await apiFetch(`${BASE}/notes/${encodeURIComponent(id)}`, {
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

/**
 * Fetch the highlights Markdown export as text — used by the native shell, where `<a download>`
 * can't save (WKWebView) so we write+share the bytes instead (#1310). Web keeps the link.
 */
export async function fetchHighlightsExport(): Promise<string> {
  const resp = await apiFetch(highlightsExportUrl(), { credentials: 'include' })
  if (!resp.ok) throw new Error(`highlights export failed: ${resp.status}`)
  return resp.text()
}

export interface ObsidianExportResult {
  mode: 'full' | 'incremental'
  revision: number
  /** The server's vault identity. Store it beside `revision` and send both back — a bare
   *  revision cannot identify a snapshot across a server-side state reset (#41). */
  epoch: string
  written: number
  removed: number
}

/**
 * Graph-aware Obsidian export (RFC-113 / #1472). Downloads the vault zip and returns the
 * `X-Export-*` header metadata so the caller can persist the cursor (for the next incremental
 * pull) and show a summary. `since` = the last revision the client applied (0 = full).
 */
export async function exportObsidian(since: number, epoch?: string): Promise<ObsidianExportResult> {
  // `epoch` identifies the server's vault state. A revision number only means something WITHIN one
  // epoch: the server's counter restarts at 0 whenever its export state is lost or unreadable, and
  // then climbs back through values this client may still hold (#41). Echo both back and a
  // collision becomes a full export instead of a delta applied against the wrong world. Omitting
  // it is safe — the server answers full.
  const q = new URLSearchParams({ format: 'obsidian', since: String(since) })
  if (epoch) q.set('epoch', epoch)
  const resp = await apiFetch(`${BASE}/export?${q}`, {
    credentials: 'include',
  })
  if (!resp.ok) throw new ApiError(resp.status, `GET /export → ${resp.status}`)
  const blob = await resp.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'closelistening-obsidian.zip'
  a.click()
  URL.revokeObjectURL(url)
  return {
    mode: (resp.headers.get('X-Export-Mode') as 'full' | 'incremental') ?? 'full',
    revision: Number(resp.headers.get('X-Export-Revision') ?? '0'),
    epoch: resp.headers.get('X-Export-Epoch') ?? '',
    written: Number(resp.headers.get('X-Export-Written') ?? '0'),
    removed: Number(resp.headers.get('X-Export-Removed') ?? '0'),
  }
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
  const resp = await apiFetch(`${BASE}/resurfacing/${encodeURIComponent(id)}/surfaced`, {
    method: 'POST',
    credentials: 'include',
  })
  if (!resp.ok && resp.status !== 401) {
    throw new ApiError(resp.status, `POST /resurfacing/surfaced → ${resp.status}`)
  }
}

/** Update resurfacing pacing (pause/resume); returns the stored settings. */
export async function putResurfacingSettings(paused: boolean): Promise<ResurfacingSettings> {
  const resp = await apiFetch(`${BASE}/resurfacing/settings`, {
    method: 'PUT',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paused }),
  })
  if (!resp.ok) throw new ApiError(resp.status, `PUT /resurfacing/settings → ${resp.status}`)
  return (await resp.json()) as ResurfacingSettings
}

// --- Delivery consent: the "Your Week" digest + push nudges (PRD-046 FR1 / #1414) ---

const COMMS_DEFAULTS: CommsSettings = {
  digest: { enabled: false, cadence: 'weekly', day_of_week: 6, hour: 13, paused: false },
  push: { enabled: false },
  email_verified: false,
  unsubscribe_ref: null,
}

export async function getComms(): Promise<CommsSettings> {
  try {
    return await getJSON<CommsSettings>('/comms')
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return { ...COMMS_DEFAULTS }
    throw err
  }
}

/**
 * The in-app "Your Week" rollup — the same content the email digest sends, served live and
 * DECOUPLED from email consent (a user's own data). Returns empty sections when unauthenticated
 * or nothing is due yet, so callers render-or-hide without special-casing 401.
 */
export async function getYourWeek(): Promise<YourWeekResponse> {
  try {
    return await getJSON<YourWeekResponse>('/your-week')
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) {
      return { sections: [], period_label: '', generated_at: '' }
    }
    throw err
  }
}

export async function putComms(update: CommsUpdate): Promise<CommsSettings> {
  const resp = await apiFetch(`${BASE}/comms`, {
    method: 'PUT',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(update),
  })
  if (!resp.ok) throw new ApiError(resp.status, `PUT /comms → ${resp.status}`)
  return (await resp.json()) as CommsSettings
}

/** The public VAPID key the browser needs to subscribe (throws 503 when push isn't configured). */
export async function getVapidKey(): Promise<string> {
  const resp = await getJSON<{ key: string }>('/push/vapid-key')
  return resp.key
}

/** Register a browser push subscription (also enables the push channel server-side). */
export async function subscribePush(subscription: unknown): Promise<{ count: number }> {
  const resp = await apiFetch(`${BASE}/push/subscribe`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(subscription),
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /push/subscribe → ${resp.status}`)
  return (await resp.json()) as { count: number }
}

/** Remove a browser push subscription (disables the channel when the last one goes). */
export async function unsubscribePush(endpoint: string): Promise<{ count: number }> {
  const resp = await apiFetch(`${BASE}/push/subscribe`, {
    method: 'DELETE',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ endpoint }),
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /push/subscribe → ${resp.status}`)
  return (await resp.json()) as { count: number }
}

// --- Collections / boards (PRD-046 FR4 / #1417) ---

export async function getCollections(): Promise<Collection[]> {
  try {
    return (await getJSON<{ items: Collection[] }>('/collections')).items
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return []
    throw err
  }
}

export async function getCollection(id: string): Promise<CollectionDetail> {
  return getJSON<CollectionDetail>(`/collections/${encodeURIComponent(id)}`)
}

export async function createCollection(name: string): Promise<Collection> {
  const resp = await apiFetch(`${BASE}/collections`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /collections → ${resp.status}`)
  return (await resp.json()) as Collection
}

export async function deleteCollection(id: string): Promise<Collection[]> {
  const resp = await apiFetch(`${BASE}/collections/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    credentials: 'include',
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /collections/${id} → ${resp.status}`)
  return ((await resp.json()) as { items: Collection[] }).items
}

export async function addToCollection(id: string, item: CollectionItemRef): Promise<Collection> {
  const resp = await apiFetch(`${BASE}/collections/${encodeURIComponent(id)}/items`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(item),
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /collections/${id}/items → ${resp.status}`)
  return (await resp.json()) as Collection
}

export async function removeFromCollection(
  id: string,
  kind: string,
  ref: string
): Promise<Collection> {
  const q = `kind=${encodeURIComponent(kind)}&ref=${encodeURIComponent(ref)}`
  const resp = await apiFetch(`${BASE}/collections/${encodeURIComponent(id)}/items?${q}`, {
    method: 'DELETE',
    credentials: 'include',
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /collections/${id}/items → ${resp.status}`)
  return (await resp.json()) as Collection
}

// --- MCP "Connected agents" (RFC-112 §5): connector config + personal-access tokens ---

/** The connector wiring the Profile section shows (resource URL + OAuth status). mcp_access-gated. */
export async function getMcpConfig(): Promise<McpConnectionConfig> {
  const resp = await apiFetch(`${BASE}/mcp/config`, { credentials: 'include' })
  if (!resp.ok) throw new ApiError(resp.status, `GET /mcp/config → ${resp.status}`)
  return (await resp.json()) as McpConnectionConfig
}

/** List the user's MCP tokens (metadata only — the secret is never returned after creation). */
export async function getMcpTokens(): Promise<McpTokenMeta[]> {
  const resp = await apiFetch(`${BASE}/mcp/tokens`, { credentials: 'include' })
  if (!resp.ok) throw new ApiError(resp.status, `GET /mcp/tokens → ${resp.status}`)
  return ((await resp.json()).items ?? []) as McpTokenMeta[]
}

/** Mint a token; the plaintext is returned ONCE (copy-then-forget). */
export async function createMcpToken(label: string): Promise<McpTokenCreated> {
  const resp = await apiFetch(`${BASE}/mcp/tokens`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label }),
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /mcp/tokens → ${resp.status}`)
  return (await resp.json()) as McpTokenCreated
}

/** Revoke a token by id; returns the remaining tokens. */
export async function revokeMcpToken(id: string): Promise<McpTokenMeta[]> {
  const resp = await apiFetch(`${BASE}/mcp/tokens/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    credentials: 'include',
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /mcp/tokens/${id} → ${resp.status}`)
  return ((await resp.json()).items ?? []) as McpTokenMeta[]
}

/** List the OAuth agents (claude.ai etc.) the user has connected. */
export async function getMcpConnections(): Promise<McpConnection[]> {
  const resp = await apiFetch(`${BASE}/mcp/connections`, { credentials: 'include' })
  if (!resp.ok) throw new ApiError(resp.status, `GET /mcp/connections → ${resp.status}`)
  return ((await resp.json()).items ?? []) as McpConnection[]
}

/** Disconnect an OAuth agent (forget consent + drop its live tokens); returns the remaining. */
export async function revokeMcpConnection(clientId: string): Promise<McpConnection[]> {
  const resp = await apiFetch(`${BASE}/mcp/connections/${encodeURIComponent(clientId)}`, {
    method: 'DELETE',
    credentials: 'include',
  })
  if (!resp.ok)
    throw new ApiError(resp.status, `DELETE /mcp/connections/${clientId} → ${resp.status}`)
  return ((await resp.json()).items ?? []) as McpConnection[]
}
