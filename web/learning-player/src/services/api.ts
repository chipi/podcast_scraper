/**
 * Typed client for the consumer platform API (`/api/app/*`, RFC-098/RFC-099).
 *
 * The app is a thin client of this API — no other backend coupling — so the same contract
 * can later serve a native mobile client (RFC-099 §10). Requests send the session cookie
 * (`credentials: 'include'`); reads are open, per-user writes require auth.
 */

import type {
  EpisodeSummary,
  FavoriteKind,
  FavoriteRef,
  ResurfacingItem,
  WhatsNewResponse,
  RecapResponse,
  RecapWindow,
  AudioSource,
  Collection,
  CollectionDetail,
  CollectionItemRef,
  CommsSettings,
  CommsUpdate,
  CorpusEnrichmentSignals,
  HealthInfo,
  KeyVoicesResponse,
  NotificationsResponse,
  EntitiesResponse,
  EntitySearchResponse,
  EpisodeEnrichmentSignals,
  TrendingTopicsResponse,
  EpisodeDetail,
  EpisodeRecap,
  EpisodesPage,
  EpisodeStats,
  FavoriteAdd,
  FavoritesResponse,
  Highlight,
  HighlightCreate,
  HighlightUpdate,
  InsightsResponse,
  InterestCluster,
  InterestHit,
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
  OrgCard,
  PersonCard,
  PlaybackPosition,
  Podcast,
  PodcastSignals,
  ResurfacingResponse,
  ResurfacingSettings,
  SearchResponse,
  SegmentsResponse,
  Storyline,
  ClusterCard,
  TopicCard,
  TopicConversationArcResponse,
  TopicPerspectivesResponse,
  TrendingEntity,
  UserStats,
  YourWeekResponse,
} from "./types"
import { ref } from "vue"
import { isNativeShell, resolveApiBase, resolveGateAuthHeader, resolveMediaUrl } from "./tier"
import { track } from "./analytics"
import { isForcedOffline, isOffline, reportServerReachable } from "../composables/useOnline"

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
    this.name = "ApiError"
    this.status = status
  }
}

// Native-shell bearer token (#1310). On the web, auth rides the session cookie (`credentials:
// 'include'`) and this stays null. In the Capacitor shell the OAuth completes in an external
// browser whose cookie the WebView can't see, so we carry the SAME signed session token here and
// send it as `Authorization: Bearer` on every request. Set from the OAuth deep-link callback
// (services/native.ts) and rehydrated from Preferences on launch.
// REACTIVE on purpose. `auth.hasSession` (stores/auth.ts) is a Pinia getter — a Vue computed — and
// computeds memoize on their REACTIVE dependencies. While this was a plain module variable, reading
// it inside that getter tracked nothing, so `hasSession` cached its first answer: the masthead would
// not gain an avatar when a login stored a token, nor lose it when sign-out cleared one, until some
// unrelated state happened to invalidate the computed. A `ref` makes the dependency real.
const authToken = ref<string | null>(null)
export function setAuthToken(token: string | null): void {
  authToken.value = token
}
export function getAuthToken(): string | null {
  return authToken.value
}

/**
 * `fetch` wrapper that adds the bearer token when present (native) and keeps every caller's
 * `credentials: 'include'` cookie path intact (web). Callers' own headers win over the injected one.
 */
// RFC-120 (#2009): fired when the API returns 401. The handler (registered in main.ts) decides —
// it only acts when we BELIEVED we were signed in (auth.isAuthenticated), so an anonymous 401 (a
// normal login-first response) is a no-op and can't cause a redirect loop.
let onUnauthorized: (() => void) | null = null
export function setOnUnauthorized(fn: (() => void) | null): void {
  onUnauthorized = fn
}

async function apiFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  // Fast-fail policy split by method (2026-09-15 RCA of #2034):
  //  - WRITES fast-fail on the full offline signal, so a mutation routes to the outbox for replay
  //    instead of silently succeeding on a live network under the forced-offline Config switch.
  //  - READS (GET) fast-fail ONLY on the explicit Config switch, NEVER on the auto-detected signal.
  //    #2034 gated reads on `isOffline()` here (and in getJSON); on device that signal false-
  //    negatives (WKWebView / cellular / VPN / cold-start), so a CONNECTED app had every read short-
  //    circuited — profile blank, collections empty, "couldn't reach the server", and the tell-tale
  //    "a bit works then a bit not". Before #2034 reads had no gate at all; this restores that, while
  //    keeping the explicit switch working. Real offline still rejects at `fetch` and the caller
  //    handles it (timeout + retry + cached/stale UI).
  const method = (init.method ?? "GET").toUpperCase()
  const blocked = method === "GET" ? isForcedOffline() : isOffline()
  if (blocked) throw new ApiError(0, `${method} ${String(input)} → offline`)
  const headers = new Headers(init.headers)
  if (!headers.has("Authorization")) {
    // User session (native OAuth) wins; else the prod coming-soon gate's Basic-auth fallback so open
    // reads reach the gated API pre-launch (services/tier.ts :: resolveGateAuthHeader). Both use the
    // `Authorization` header, so they're mutually exclusive — acceptable until native login lands.
    if (authToken.value) headers.set("Authorization", `Bearer ${authToken.value}`)
    else {
      const gate = resolveGateAuthHeader()
      if (gate) headers.set("Authorization", gate)
    }
  }
  // Every request is a probe of whether the SERVER works — the only place that can observe it.
  // A transport failure (refused / DNS / timeout) and a 503 both mean "not usable right now"; a 503
  // specifically is how the API now reports that it cannot authenticate anyone (a lost signing
  // secret, incident 2026-09-16) rather than lying that the caller's credential is bad.
  let resp: Response
  try {
    resp = await fetch(input, { ...init, headers })
  } catch (err) {
    reportServerReachable(false)
    throw err
  }
  reportServerReachable(resp.status !== 503)
  if (resp.status === 401 && onUnauthorized) onUnauthorized()
  return resp
}

/**
 * Safety timeout for a read the app is waiting on (F1.1a/F1.4). Generous — it is a BACKSTOP for the
 * "online but the server never answers" case, not the primary offline path (`isOffline()` fails
 * fast below). Long enough that a slow-but-legit response on a bad connection still lands; short
 * enough that a page resolves to its graceful "can't load" state instead of a spinner forever.
 * Callers that pass their own `signal` (e.g. getMe) opt out and own their bound.
 */
const READ_SAFETY_MS = 15000

async function getJSON<T>(
  path: string,
  params?: Record<string, string | number | undefined>,
  init?: RequestInit
): Promise<T> {
  // Fast-fail ONLY on the explicit Config offline switch — never on the auto-detected signal (RCA of
  // #2034, 2026-09-15). That signal false-negatives on device (WKWebView / cellular / VPN / a slow
  // cold-start network report), and gating reads on it made a CONNECTED app look broken — profile
  // blank, collections empty, "couldn't reach the server", and the tell-tale "a bit works then a bit
  // not". Before #2034 reads had no gate at all; this restores that. A read now always attempts and
  // falls to its cache / graceful error via the real error path (the timeout backstop below + the
  // store's cached/stale UI). status 0 is a transport failure, never a 401.
  if (isForcedOffline()) throw new ApiError(0, `GET ${path} → offline (forced)`)
  const url = new URL(`${BASE}${path}`, window.location.origin)
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, String(v))
    }
  }
  const resp = await apiFetch(url.toString(), {
    credentials: "include",
    headers: { Accept: "application/json" },
    ...init,
    // Backstop the online-but-unreachable case; a caller-supplied signal wins. Last, so `...init`
    // cannot drop it.
    signal: init?.signal ?? AbortSignal.timeout(READ_SAFETY_MS),
  })
  if (!resp.ok) {
    throw new ApiError(resp.status, `GET ${path} → ${resp.status}`)
  }
  return (await resp.json()) as T
}

/**
 * `getMe` is on the app's first-paint path (the router guard awaits it via auth.ensureLoaded). Bound
 * it with a timeout so a no-snapshot OFFLINE cold-start fails fast instead of hanging until the OS
 * connection timeout — which would leave even the static landing page blank. apiFetch has no global
 * timeout by design (a blanket one could abort slow legit requests); this scopes it to /me only.
 */
const ME_TIMEOUT_MS = 8000

/**
 * Absolutise the avatar URL for the native shell.
 *
 * The server returns `image` as a RELATIVE path (`/api/app/profile/<id>/avatar?v=…`). On the web
 * that is correct; inside the Capacitor WebView the document origin is `capacitor://localhost`, so
 * it resolves there, 404s, and `ProfileAvatar` quietly falls back to initials — an uploaded photo
 * simply never appeared (operator 2026-09-16).
 *
 * This is the SAME defect class as the artwork and audio URLs that motivated the device test tier;
 * `getAudioSource` already does exactly this for `media_url`. Done here, in one place, rather than
 * at each of the avatar's render sites.
 */
function withAbsoluteAvatar(me: Me): Me {
  return me.image ? { ...me, image: resolveMediaUrl(me.image) ?? me.image } : me
}

/** Signed-in user, or `null` when not authenticated (401). */
export async function getMe(): Promise<Me | null> {
  try {
    const me = await getJSON<Me>("/me", undefined, { signal: AbortSignal.timeout(ME_TIMEOUT_MS) })
    return withAbsoluteAvatar(me)
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return null
    // A timeout / transport abort is NOT a signed-out signal — rethrow so refresh() keeps the device
    // snapshot rather than clearing it (auth.ts: only a 401/403 may destroy cached auth state).
    throw err
  }
}

/** Catalog: episodes across the corpus, newest-first (paginated). */
export function listEpisodes(params: ListEpisodesParams = {}): Promise<EpisodesPage> {
  return getJSON<EpisodesPage>("/episodes", {
    page: params.page,
    page_size: params.pageSize,
    status: params.status,
    feed_id: params.feedId,
  })
}

/** Catalog: one podcast's episodes, newest-first (paginated). */
export function listPodcastEpisodes(
  feedId: string,
  params: Omit<ListEpisodesParams, "feedId"> = {}
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

/** At most this many slugs per batch request — the server's cap (`_BATCH_MAX`). */
const EPISODE_BATCH_MAX = 100

/**
 * Several episode details in one request each 100 — the queue, recently played and the other
 * lists of saved slugs. Returns a map by slug; unknown slugs are simply absent.
 *
 * A server that predates `/episodes/batch` reads "batch" as a slug and answers 404: the 1.0.3 app
 * is published before the deploy, so for a while it talks to that server. On 404 this falls back to
 * one `getEpisode` per slug, which is what every screen did before.
 */
export async function getEpisodesBatch(slugs: string[]): Promise<Record<string, EpisodeDetail>> {
  const unique = [...new Set(slugs.filter(Boolean))]
  const out: Record<string, EpisodeDetail> = {}
  for (let i = 0; i < unique.length; i += EPISODE_BATCH_MAX) {
    const chunk = unique.slice(i, i + EPISODE_BATCH_MAX)
    const query = chunk.map((s) => `slugs=${encodeURIComponent(s)}`).join("&")
    try {
      const body = await getJSON<{ items: EpisodeDetail[]; missing: string[] }>(
        `/episodes/batch?${query}`
      )
      for (const d of body.items) out[d.slug] = d
    } catch (err) {
      if (!(err instanceof ApiError) || err.status !== 404) throw err
      const each = await Promise.all(chunk.map((s) => getEpisode(s).catch(() => null)))
      for (const d of each) if (d) out[d.slug] = d
    }
  }
  return out
}

/** Transcript segments for the sync engine.
 *
 *  `lang` selects the ALTERNATIVE rendering, it does not pick the default: omit it and the
 *  resolver serves English whenever an English render exists (D-38). Pass the episode's source
 *  tag to read the original against the same audio.
 */
export function getSegments(slug: string, lang?: string | null): Promise<SegmentsResponse> {
  const path = `/episodes/${encodeURIComponent(slug)}/segments`
  return getJSON<SegmentsResponse>(lang ? `${path}?lang=${encodeURIComponent(lang)}` : path)
}

/** Origin audio descriptor — the client plays `url` directly (bridge, never rehost). */
export async function getAudioSource(slug: string, validate = false): Promise<AudioSource> {
  // `validate` HEADs the origin (an extra network call) and is what populates `content_length`.
  // The download path asks for it so a transfer that cannot fit is refused BEFORE it starts;
  // playback does not, because it would put a round trip in front of every play.
  const src = await getJSON<AudioSource>(
    `/episodes/${encodeURIComponent(slug)}/audio-source${validate ? "?validate=true" : ""}`
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

/** Post-episode recap (RFC-122 #2038) — summary key points + top insights + a signature quote. */
export function getEpisodeRecap(slug: string, limit = 3): Promise<EpisodeRecap> {
  return getJSON<EpisodeRecap>(`/episodes/${encodeURIComponent(slug)}/recap`, { limit })
}

/** KG entities (persons/orgs/topics) for an episode (empty when no KG artifact).
 *  Person photos absolutised like every people-carrying fetcher: the Episode notes panel shows the
 *  host and guests with their photos, and a relative route would fail on device in silence
 *  (see __checks__/person-photo-absolutised.test.ts). */
export async function getEntities(slug: string): Promise<EntitiesResponse> {
  const resp = await getJSON<EntitiesResponse>(`/episodes/${encodeURIComponent(slug)}/entities`)
  return {
    ...resp,
    persons: (resp.persons ?? []).map((p) =>
      p.image_url ? { ...p, image_url: resolveMediaUrl(p.image_url) ?? p.image_url } : p
    ),
  }
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

/**
 * Absolutise the entity image URLs a card carries — person photo, org logo, and the `image_url`
 * on every person in a related/top-voices list.
 *
 * EXACTLY what `withAbsoluteAvatar` does for `/me`, and for the same reason: the server returns
 * these as RELATIVE paths (`/api/app/persons/<id>/photo`). On the web that is correct; inside the
 * Capacitor WebView the document origin is `capacitor://localhost`, so a relative path resolves
 * THERE, 404s, and `ProfileAvatar` quietly falls back to initials — the photo simply never
 * appears. Same defect class as the artwork and audio URLs behind the device test tier.
 *
 * Done here at the API boundary, in one place, rather than at each render site — the person photo
 * had been left raw at one `:src` in PersonCardContent while every other image in the app went
 * through `resolveMediaUrl`, which is how it stayed broken.
 */
function withAbsoluteEntityImages<T>(card: T): T {
  const c = card as Record<string, unknown>
  const abs = (u: unknown) => (typeof u === "string" ? resolveMediaUrl(u) ?? u : u)

  const web = c.web as Record<string, unknown> | null | undefined
  if (web && typeof web === "object") {
    const next: Record<string, unknown> = { ...web }
    if (next.image_url) next.image_url = abs(next.image_url)
    if (next.logo_url) next.logo_url = abs(next.logo_url)
    c.web = next
  }
  for (const key of ["related_people", "people", "top_voices", "key_voices"]) {
    const list = c[key]
    if (Array.isArray(list)) {
      c[key] = list.map((p) =>
        p && typeof p === "object" && (p as Record<string, unknown>).image_url
          ? { ...p, image_url: abs((p as Record<string, unknown>).image_url) }
          : p
      )
    }
  }
  return c as T
}

/**
 * One page of an entity card's episodes (server paging, 2026-10-08). The cards returned every
 * episode — a storyline 99, 1.29 MB on prod — and the client showed five. Omit for the whole list.
 */
export interface EpisodePage {
  limit?: number
  offset?: number
  /** Person card: leave out the shows they host (their back-catalogue), on the server. */
  excludeHostShows?: boolean
}
function pageParams(p?: EpisodePage): Record<string, string | number | undefined> {
  return {
    episodes_limit: p?.limit,
    episodes_offset: p?.offset || undefined,
    exclude_host_shows: p?.excludeHostShows ? "true" : undefined,
  }
}

/** Person profile card — appears-in episodes + related people/topics (KG co-occurrence). */
export async function getPersonCard(id: string, scope?: "all" | "mine", page?: EpisodePage): Promise<PersonCard> {
  // scope='mine' = the guest across the episodes the signed-in user has heard (P3 #1122).
  return withAbsoluteEntityImages(
    await getJSON<PersonCard>(`/persons/${encodeURIComponent(id)}`, { scope, ...pageParams(page) })
  )
}

/**
 * Theme card — the member topics, their MERGED episodes, and the people across them.
 *
 * No `scope` parameter: the topic card has one, but a personally-filtered union answers a different
 * question than "what is this grouping, across the corpus".
 */
export async function getThemeCard(id: string, page?: EpisodePage): Promise<ClusterCard> {
  return withAbsoluteEntityImages(
    await getJSON<ClusterCard>(`/themes/${encodeURIComponent(id)}`, pageParams(page))
  )
}

/**
 * Storyline card — same shape, membership decided by co-occurrence instead of similarity.
 *
 * Takes the `thc:` id OR an anchor topic id, because `/storyline/:id` routes by anchor topic. This
 * replaces deriving the page from the anchor's TOPIC card, whose `episodes` are the anchor's alone:
 * the page said "Discussed in 30 episodes" for a storyline spanning 40.
 */
export async function getStorylineCard(id: string, page?: EpisodePage): Promise<ClusterCard> {
  return withAbsoluteEntityImages(
    await getJSON<ClusterCard>(`/storylines/${encodeURIComponent(id)}`, pageParams(page))
  )
}

/** Topic card — episodes-about + cluster siblings + related people (KG-grounded). */
export async function getTopicCard(id: string, scope?: "all" | "mine", page?: EpisodePage): Promise<TopicCard> {
  return withAbsoluteEntityImages(
    await getJSON<TopicCard>(`/topics/${encodeURIComponent(id)}`, { scope, ...pageParams(page) })
  )
}

/** Organization card (#2031) — mentioned-in episodes + co-occurring people/orgs/topics. */
export async function getOrgCard(id: string, page?: EpisodePage): Promise<OrgCard> {
  return withAbsoluteEntityImages(
    await getJSON<OrgCard>(`/organizations/${encodeURIComponent(id)}`, pageParams(page))
  )
}

/**
 * Topic perspectives — each speaker's grounded insights on the topic (#1146).
 *
 * Person photos ABSOLUTISED, same as every other people-carrying endpoint. `build_topic_perspectives`
 * hydrates `image_url` server-side and `TopicPerspectives.vue` renders it into a `ProfileAvatar`,
 * but the server returns it relative — which resolves against `capacitor://localhost` in the native
 * shell, 404s, and falls back to initials without an error anyone can see.
 *
 * Found by sweeping every endpoint that can carry a person photo after the same bug turned up on
 * the key-voices rail (operator 2026-09-27). This was the last one still raw.
 */
/**
 * A page of perspectives (server paging, 2026-10-08: a storyline's were 160 KB, every speaker with
 * every take, for a section showing three speakers with two takes). Omit for the full response.
 */
export interface PerspectivesPage {
  /** At most this many takes per speaker; `insight_count` stays their total. */
  perSpeaker?: number
  offset?: number
  limit?: number
}
function perspectivesParams(p?: PerspectivesPage): Record<string, string | number | undefined> {
  return {
    insights_per_speaker: p?.perSpeaker,
    speakers_offset: p?.offset || undefined,
    speakers_limit: p?.limit,
  }
}

export async function getTopicPerspectives(
  id: string,
  scope?: "all" | "mine",
  page?: PerspectivesPage
): Promise<TopicPerspectivesResponse> {
  return perspectivesFrom(`/topics/${encodeURIComponent(id)}/perspectives`, {
    scope,
    ...perspectivesParams(page),
  })
}

/**
 * The same, for a GROUPING — a theme or a storyline.
 *
 * Separate endpoints rather than one with a kind parameter, because the server resolves the id
 * differently for each: `/storyline/:id` routes by ANCHOR TOPIC, so a bare `topic:` id is a valid
 * storyline argument, and the same topic is usually a member of a theme too. The path is what
 * disambiguates.
 *
 * No `scope`, matching the grouping card routes: a theme page asks what the grouping is across the
 * corpus, and a personally-filtered union would quietly answer a different question.
 */
export function getThemePerspectives(id: string, page?: PerspectivesPage): Promise<TopicPerspectivesResponse> {
  return perspectivesFrom(`/themes/${encodeURIComponent(id)}/perspectives`, perspectivesParams(page))
}

export function getStorylinePerspectives(id: string, page?: PerspectivesPage): Promise<TopicPerspectivesResponse> {
  return perspectivesFrom(`/storylines/${encodeURIComponent(id)}/perspectives`, perspectivesParams(page))
}

/**
 * Shared projection for all three. The photo rewrite is the load-bearing part: the server returns
 * `image_url` relative, which resolves against `capacitor://localhost` in the native shell, 404s,
 * and silently falls back to initials. Keeping it in ONE place means a new perspectives surface
 * cannot reintroduce that bug by forgetting it.
 */
async function perspectivesFrom(
  path: string,
  params?: Record<string, string | number | undefined>
): Promise<TopicPerspectivesResponse> {
  const resp = await getJSON<TopicPerspectivesResponse>(path, params)
  return {
    ...resp,
    perspectives: (resp.perspectives ?? []).map((p) =>
      p.image_url ? { ...p, image_url: resolveMediaUrl(p.image_url) ?? p.image_url } : p
    ),
  }
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
    _corpusEnrichment = getJSON<{ signals: CorpusEnrichmentSignals }>("/corpus/enrichment")
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
    _trendingTopics = getJSON<TrendingTopicsResponse>("/corpus/trending-topics").catch((err) => {
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
  kind: "person" | "topic",
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
  scope?: "all" | "mine",
  enrichResults?: boolean
): Promise<SearchResponse> {
  // scope='mine' = grounded recall over the signed-in user's heard∪captured corpus (P3 #1120).
  // enrichResults=true asks the server to decorate hits with
  //   metadata.query_enrichments.related_topics (RFC-088, #1261-1). Chain failures on
  //   the server are swallowed — the client should tolerate hits without the field.
  return getJSON<SearchResponse>("/search", {
    q,
    top_k: topK,
    scope,
    enrich_results: enrichResults ? "true" : undefined,
  })
}

/** Resolve a query to a person/topic card (exact/near-exact); `entity: null` when none. */
export function resolveEntity(q: string): Promise<EntitySearchResponse> {
  return getJSON<EntitySearchResponse>("/entities/search", { q })
}

/** Home discovery feed — interest-ranked when enabled + signed-in, else recency (the default). */
export function getDiscover(limit = 8): Promise<EpisodesPage> {
  return getJSON<EpisodesPage>("/discover", { limit })
}

/** Home's What's new (operator 2026-10-07): newest from what the listener follows, or — with
 *  nothing followed or nothing matching — newest across every show. `scope` says which. */
export function getWhatsNew(limit = 5): Promise<WhatsNewResponse> {
  return getJSON<WhatsNewResponse>("/whats-new", { limit })
}

/** Home's Recommended without listening history (operator 2026-10-07): episodes carrying what the
 *  listener follows or their listening implies, minus what they played. `basis: "none"` = nothing to
 *  base it on, and no items. */
export function getRecommended(limit = 12): Promise<{ items: EpisodeSummary[]; basis: "interests" | "none" }> {
  return getJSON("/recommended", { limit })
}

/** Fire-and-forget: log a click on a discovery-feed episode (its shown rank position) for
 *  ranking telemetry (#11). Silent no-op when signed out or on any network error. */
export function recordDiscoverClick(slug: string, position: number): void {
  void apiFetch(`${BASE}/discover/click`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ slug, position }),
  }).catch(() => {})
}

/** Followables of one kind whose label contains `q`, best match first — an Interests section's
 *  search box. Substring, not exact: `resolveEntity` answers "does this name one thing". */
export async function searchInterests(
  kind: InterestHit["kind"],
  q: string,
  limit = 20
): Promise<InterestHit[]> {
  return (await getJSON<{ items: InterestHit[] }>("/interests/search", { kind, q, limit })).items
}

/** Top interest clusters for the picker, by corpus prevalence. */
export async function getTopClusters(limit = 12): Promise<InterestCluster[]> {
  return (await getJSON<{ items: InterestCluster[] }>("/themes", { limit })).items
}

/** Top storylines (theme clusters — topics discussed together) for the Home rail + picker. */
export async function getStorylines(limit = 12): Promise<Storyline[]> {
  return (await getJSON<{ items: Storyline[] }>("/storylines", { limit })).items
}

/** Trending entities of a kind (RFC-103 momentum), corpus-wide or the signed-in user's ('mine'). */
/** Trend window presets (RFC-103 R2) — the recent bucket the velocity is measured over. */
export type TrendWindow = "1m" | "3m" | "6m" | "1y"

// Shared per (kind, scope, limit, window) so concurrent callers share one request — the Home
// storylines rail and StorylineView both ask for `storyline` momentum on the same navigation
// (advisor N3). Cleared on failure so a transient error can retry.
//
// HOW LONG an answer is shared depends on whose it is (2026-10-09). This used to keep every answer
// for the whole session on the reasoning that "trending is corpus-wide" — true until `scope: 'mine'`
// existed. "Your trends" then stayed as first loaded however much you followed or listened (the
// cross-surface e2e spec caught it), and since the key has no user in it, a second account on the
// same device was handed the first account's. Now:
//   - mine: shared only while IN FLIGHT; every later call asks again (it is yours and it moves).
//   - corpus: shared for 5 minutes, the server's own cache window for the same answer.
const TRENDING_CORPUS_TTL_MS = 5 * 60_000
const _trending = new Map<string, { p: Promise<TrendingEntity[]>; at: number; settled: boolean }>()
export function getTrending(
  kind: string,
  scope: "corpus" | "mine" = "corpus",
  limit = 12,
  window: TrendWindow = "3m"
): Promise<TrendingEntity[]> {
  const key = `${kind}:${scope}:${limit}:${window}`
  const hit = _trending.get(key)
  const reusable =
    hit && (!hit.settled || (scope === "corpus" && Date.now() - hit.at < TRENDING_CORPUS_TTL_MS))
  if (hit && reusable) return hit.p
  // Person photos ABSOLUTISED (2026-09-30). `/trending` hydrates `image_url` for people as a
  // RELATIVE route; left raw, it resolved against capacitor://localhost on device, 404'd, and
  // every Trends → People row fell back to initials while the same person's card showed the photo.
  const entry = {
    at: Date.now(),
    settled: false,
    p: getJSON<{ items: TrendingEntity[] }>("/trending", { kind, scope, limit, window })
      .then((r) =>
        r.items.map((e) =>
          e.image_url ? { ...e, image_url: resolveMediaUrl(e.image_url) ?? e.image_url } : e
        )
      )
      .catch((err) => {
        if (_trending.get(key) === entry) _trending.delete(key)
        throw err
      })
      .finally(() => {
        entry.settled = true
      }),
  }
  _trending.set(key, entry)
  return entry.p
}

/** Test seam: forget every shared trending answer. */
export function _resetTrendingForTests(): void {
  _trending.clear()
}

/** The signed-in user's interest cluster ids; `[]` when signed out (401). Auth-gated. */
export async function getUserInterests(): Promise<string[]> {
  try {
    return (await getJSON<{ items: string[] }>("/interests")).items
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return []
    throw err
  }
}

/** The user's favorites. A 401 THROWS (same correction as getLibrary, #2004 #3): the store falls
 *  back to its cache instead of persisting an empty list as truth. Sole caller: stores/favorites. */
export async function getFavorites(): Promise<FavoritesResponse> {
  return await getJSON<FavoritesResponse>("/favorites")
}

/** Query for one page of the Saved list (`GET /favorites` with `limit`). */
export interface FavoritesPageQuery {
  kind?: FavoriteKind
  q?: string
  color?: string | null
  sort?: "recent" | "title"
  offset?: number
  limit: number
}

/**
 * One page of one kind of favourite, filtered and sorted on the server.
 *
 * A server that predates paging ignores the parameters and returns every favourite with no
 * `total`; this then does the same filtering and slicing here, so the screen behaves the same on
 * either server.
 */
export async function getFavoritesPage(query: FavoritesPageQuery): Promise<FavoritesResponse> {
  const resp = await getJSON<FavoritesResponse>("/favorites", {
    kind: query.kind,
    q: query.q?.trim() || undefined,
    color: query.color ?? undefined,
    sort: query.sort,
    offset: query.offset ?? 0,
    limit: query.limit,
  })
  return resp.total === undefined ? pageFavoritesLocally(resp, query) : resp
}

export function pageFavoritesLocally(all: FavoritesResponse, query: FavoritesPageQuery): FavoritesResponse {
  const needle = (query.q ?? "").trim().toLocaleLowerCase()
  const has = (...texts: (string | null | undefined)[]) =>
    !needle || texts.some((t) => (t ?? "").toLocaleLowerCase().includes(needle))
  const colourOk = (c?: string | null) => !query.color || c === query.color
  const eps = all.episodes.filter((e) => colourOk(e.color) && has(e.title, e.podcast_title))
  const ents = (all.entities ?? []).filter((e) => colourOk(e.color) && has(e.label))
  const counts: Partial<Record<FavoriteKind, number>> = { episode: eps.length }
  for (const e of ents) counts[e.kind] = (counts[e.kind] ?? 0) + 1
  const byTitle = query.sort === "title"
  const start = query.offset ?? 0
  const end = start + query.limit
  if (query.kind === "episode") {
    const list = byTitle ? [...eps].sort((a, b) => a.title.localeCompare(b.title)) : eps
    return { episodes: list.slice(start, end), entities: [], total: list.length, counts }
  }
  const list = query.kind ? ents.filter((e) => e.kind === query.kind) : ents
  const sorted = byTitle ? [...list].sort((a, b) => a.label.localeCompare(b.label)) : list
  return { episodes: [], entities: sorted.slice(start, end), total: sorted.length, counts }
}

/**
 * Which items are saved — identity only, for the hearts. Falls back to deriving it from the full
 * list on a server that predates `/favorites/refs` (404).
 */
export async function getFavoriteRefs(): Promise<FavoriteRef[]> {
  try {
    return (await getJSON<{ items: FavoriteRef[] }>("/favorites/refs")).items
  } catch (err) {
    if (!(err instanceof ApiError) || err.status !== 404) throw err
    return favoriteRefsOf(await getFavorites())
  }
}

/** The identities in a full favourites response (what every write still answers with). */
export function favoriteRefsOf(f: FavoritesResponse): FavoriteRef[] {
  return [
    ...f.episodes.map((e) => ({ kind: "episode" as const, ref: e.slug, color: e.color ?? null })),
    ...(f.entities ?? []).map((e) => ({ kind: e.kind, ref: e.ref, color: e.color ?? null })),
  ]
}

/** Save an item (auth-gated); returns the updated favorites. */
export async function addFavorite(item: FavoriteAdd): Promise<FavoritesResponse> {
  const resp = await apiFetch(`${BASE}/favorites`, {
    method: "PUT",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(item),
  })
  if (!resp.ok) throw new ApiError(resp.status, `PUT /favorites → ${resp.status}`)
  return (await resp.json()) as FavoritesResponse
}

/** Remove a saved item by kind+ref (auth-gated); returns the updated favorites. */
export async function removeFavorite(kind: string, ref: string): Promise<FavoritesResponse> {
  const resp = await apiFetch(
    `${BASE}/favorites/${encodeURIComponent(kind)}/${encodeURIComponent(ref)}`,
    { method: "DELETE", credentials: "include" }
  )
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /favorites → ${resp.status}`)
  return (await resp.json()) as FavoritesResponse
}

/** Set (token) or clear (null) a saved item's colour by kind+ref (RFC-121 ph. 4). 404 when the
 *  favorite is absent — colour is set on something already saved. Returns updated favorites. */
export async function setFavoriteColor(
  kind: string,
  ref: string,
  color: string | null
): Promise<FavoritesResponse> {
  const resp = await apiFetch(
    `${BASE}/favorites/${encodeURIComponent(kind)}/${encodeURIComponent(ref)}`,
    {
      method: "PATCH",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ color }),
    }
  )
  if (!resp.ok) throw new ApiError(resp.status, `PATCH /favorites → ${resp.status}`)
  return (await resp.json()) as FavoritesResponse
}

/** Follow one interest token — cluster (`tc:`), topic (`topic:`) or person (`person:`). Auth-gated. */
/** What the listener's own listening says they are into — people and topics from the episodes they
 *  heard or captured from, ranked (GET /interests/derived). */
export interface DerivedInterest {
  token: string
  kind: "person" | "topic"
  label: string
  count: number
  weight?: number
}
export async function getDerivedInterests(): Promise<DerivedInterest[]> {
  return (await getJSON<{ items: DerivedInterest[] }>("/interests/derived")).items
}

export async function addInterest(token: string): Promise<string[]> {
  const resp = await apiFetch(`${BASE}/interests/${encodeURIComponent(token)}`, {
    method: "POST",
    credentials: "include",
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /interests → ${resp.status}`)
  return ((await resp.json()) as { items: string[] }).items
}

/** Unfollow one interest token (auth-gated); returns the remaining list. */
export async function removeInterest(token: string): Promise<string[]> {
  const resp = await apiFetch(`${BASE}/interests/${encodeURIComponent(token)}`, {
    method: "DELETE",
    credentials: "include",
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /interests → ${resp.status}`)
  return ((await resp.json()) as { items: string[] }).items
}

/** Replace the user's interest cluster ids (auth-gated); returns the stored list. */
export async function putUserInterests(clusterIds: string[]): Promise<string[]> {
  const resp = await apiFetch(`${BASE}/interests`, {
    method: "PUT",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items: clusterIds }),
  })
  if (!resp.ok) {
    throw new ApiError(resp.status, `PUT /interests → ${resp.status}`)
  }
  return ((await resp.json()) as { items: string[] }).items
}

/** Distinct shows in the corpus (public, not per-user). */
/**
 * The whole show catalogue (~80 KB on prod). Shared and kept for a minute (2026-10-08): seven
 * surfaces ask for it, and Library → Following alone fetched it three times in one open. It is the
 * same for every listener and changes only when a show is added, so one request serves them all; a
 * failed request is not kept, so the next caller retries.
 */
let podcastsCache: { at: number; promise: Promise<Podcast[]> } | null = null
const PODCASTS_TTL_MS = 60_000
export function getPodcasts(): Promise<Podcast[]> {
  if (podcastsCache && Date.now() - podcastsCache.at < PODCASTS_TTL_MS) return podcastsCache.promise
  const promise = getJSON<{ items: Podcast[] }>("/podcasts").then((r) => r.items)
  podcastsCache = { at: Date.now(), promise }
  promise.catch(() => {
    if (podcastsCache?.promise === promise) podcastsCache = null
  })
  return promise
}

/** One page of the show catalogue (`GET /podcasts` with `limit`, 1.0.3). */
export interface PodcastsPage {
  items: Podcast[]
  total: number
  /** Every category in the catalogue (the filter's options). */
  categories: string[]
}

export interface PodcastsQuery {
  q?: string
  category?: string
  sort?: "newest" | "oldest" | "az" | "za" | "trending"
  offset?: number
  limit: number
  /** Only these shows — a lookup by id. */
  feedIds?: string[]
  /** Leave out descriptions (a list of names). */
  compact?: boolean
}

/**
 * A page of the catalogue, filtered and sorted on the server. Against an older server (no `total`)
 * the full list is cut here; "trending" then reads A-Z, as it does when no velocity is known.
 */
export async function getPodcastsPage(query: PodcastsQuery): Promise<PodcastsPage> {
  const params = new URLSearchParams({ limit: String(query.limit), offset: String(query.offset ?? 0) })
  if (query.q?.trim()) params.set("q", query.q.trim())
  if (query.category) params.set("category", query.category)
  if (query.sort) params.set("sort", query.sort)
  if (query.compact) params.set("compact", "true")
  for (const id of query.feedIds ?? []) params.append("feed_ids", id)
  const resp = await getJSON<{ items: Podcast[]; total?: number; categories?: string[] }>(
    `/podcasts?${params}`,
  )
  if (resp.total !== undefined) return resp as PodcastsPage
  return pagePodcastsLocally(resp.items, query)
}

export function pagePodcastsLocally(all: Podcast[], query: PodcastsQuery): PodcastsPage {
  const ids = new Set(query.feedIds ?? [])
  const words = (query.q ?? "").toLowerCase().split(/\s+/).filter(Boolean)
  const title = (p: Podcast) => (p.title ?? p.feed_id).toLowerCase()
  const hay = (p: Podcast) => [title(p), ...(p.authors ?? [])].join(" ").toLowerCase()
  let list = all.filter(
    (p) =>
      p.feed_id &&
      (!ids.size || ids.has(p.feed_id)) &&
      (!query.category || p.category === query.category) &&
      words.every((w) => hay(p).includes(w)),
  )
  const byTitle = (a: Podcast, b: Podcast) => title(a).localeCompare(title(b))
  if (query.sort === "za") list = [...list].sort((a, b) => byTitle(b, a))
  else if (query.sort === "az" || query.sort === "trending") list = [...list].sort(byTitle)
  else {
    const dated = list.filter((p) => p.last_updated)
    const undated = list.filter((p) => !p.last_updated).sort(byTitle)
    dated.sort((a, b) => {
      const c = (a.last_updated ?? "").localeCompare(b.last_updated ?? "") || byTitle(a, b)
      return query.sort === "oldest" ? c : -c
    })
    list = [...dated, ...undated]
  }
  const start = query.offset ?? 0
  return {
    items: list.slice(start, start + query.limit),
    total: list.length,
    categories: [...new Set(all.map((p) => p.category).filter((c): c is string => !!c))].sort(),
  }
}

/** These shows, by id — what a board, a show page or a followed list needs, not the catalogue. */
export async function getPodcastsByIds(ids: string[]): Promise<Podcast[]> {
  const unique = [...new Set(ids.filter(Boolean))]
  const out: Podcast[] = []
  for (let i = 0; i < unique.length; i += 200) {
    const chunk = unique.slice(i, i + 200)
    out.push(...(await getPodcastsPage({ feedIds: chunk, limit: chunk.length, sort: "az" })).items)
  }
  return out
}

/** Test seam: forget the shared show catalogue. */
export function __resetPodcastsCache(): void {
  podcastsCache = null
}

/**
 * Shows for the guided start's "follow a few shows" step (operator 2026-10-08): active in the last
 * month first, then the most loved, lifted by the listener's interests, minus shows they follow.
 */
export async function getSuggestedShows(limit = 8): Promise<Podcast[]> {
  return (await getJSON<{ items: Podcast[] }>(`/podcasts/suggested?limit=${limit}`)).items
}

// --- Feed subscriptions ("follow a show") — the library the Your Week digest reads for its
// "new in your follows" section. NOT the same store as interests (topic:/person: tokens), which
// feed "Recommended for you".

/**
 * The user's followed shows (auth-gated).
 *
 * A 401 THROWS. It used to return `[]`, and the store took that as fresh truth: `loaded = true`,
 * `stale = false`, and `writeCached('library', [])` — so an expired session did not just render
 * "you're not following any shows yet", it PERSISTED that answer into the per-user cache, where
 * the offline fallback would keep repeating it. "We could not ask" and "you follow nothing" are
 * different facts and the second is the one that makes a user think their follows were lost.
 *
 * Same correction, same reasoning, as `getCollections` (#2004 item 13). This has exactly one
 * caller — `stores/library.ts` — so the swallow protected nothing else.
 */
export async function getLibrary(): Promise<LibraryItem[]> {
  return (await getJSON<{ items: LibraryItem[] }>("/library")).items
}

/** Follow a show (idempotent on feed_id, auth-gated); returns the updated library. */
export async function followShow(
  feedId: string,
  meta: { feedUrl?: string | null; title?: string | null } = {}
): Promise<LibraryItem[]> {
  const resp = await apiFetch(`${BASE}/library`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
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
    method: "DELETE",
    credentials: "include",
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /library → ${resp.status}`)
  return ((await resp.json()) as { items: LibraryItem[] }).items
}

/** Show-level signals for a show page: topics/themes it's about, who's on it, what's trending. */
export function getPodcastSignals(feedId: string, topK?: number): Promise<PodcastSignals> {
  const q = topK != null ? `?top_k=${topK}` : ""
  return getJSON<PodcastSignals>(`/podcasts/${encodeURIComponent(feedId)}/signals${q}`)
}

/** Saved playback positions, newest-first (Home "Continue"); `[]` when signed out. */
/** Which positions to read (`GET /playback` with `limit`, 1.0.3) — none = every position. */
export interface PlaybackQuery {
  /** Started and not finished — Home's Continue listening. */
  inProgress?: boolean
  /** Only these episodes. */
  slugs?: string[]
  limit: number
}

export async function getPlaybackList(query?: PlaybackQuery): Promise<PlaybackPosition[]> {
  try {
    if (!query) return (await getJSON<{ items: PlaybackPosition[] }>("/playback")).items
    const params = new URLSearchParams({ limit: String(query.limit) })
    if (query.inProgress) params.set("in_progress", "true")
    for (const s of query.slugs ?? []) params.append("slugs", s)
    const resp = await getJSON<{ items: PlaybackPosition[]; total?: number }>(`/playback?${params}`)
    if (resp.total !== undefined) return resp.items
    // An older server ignores the parameters and answers with everything: filter here.
    const wanted = new Set(query.slugs ?? [])
    return resp.items
      .filter((p) => !wanted.size || wanted.has(p.slug))
      .filter((p) => !query.inProgress || (p.position_seconds > 1 && !p.finished))
      .slice(0, query.limit)
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
    method: "PUT",
    credentials: "include",
    keepalive: true,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      position_seconds: positionSeconds,
      finished,
      ...(clientTs ? { client_ts: clientTs } : {}),
      // The listener's offset, so listening time buckets by THEIR day (#1914). `getTimezoneOffset`
      // returns minutes WEST of UTC, i.e. the opposite sign to how offsets are written, so it is
      // negated here rather than at the server where the convention would be invisible.
      //
      // Sent per save on purpose: it is also the right answer for DST and for travel, since a save
      // belongs to the day it happened in the offset then in effect. Read at call time, not at
      // module load, so a device that crosses a boundary mid-session reports the new one.
      tz_offset_minutes: -new Date().getTimezoneOffset(),
    }),
  })
  // 401 THROWS like any other refusal (advisor 1.1). Swallowing it reported success, so the
  // flush cleared the pending flag and the live persister marked the position `synced` — with
  // nothing written server-side. A signed-out tick still costs nothing: the caller swallows.
  if (!resp.ok) {
    throw new ApiError(resp.status, `PUT /playback → ${resp.status}`)
  }
}

/**
 * The signed-in user's listening recap for one window (#1914).
 *
 * Sends the listener's UTC offset so the window is cut on the same day boundaries the RECORDING
 * used — otherwise a Sunday evening falls outside the week it belongs to.
 */
export async function getRecap(window: RecapWindow): Promise<RecapResponse | null> {
  try {
    const tz = -new Date().getTimezoneOffset()
    return await getJSON<RecapResponse>(`/me/recap?window=${window}&tz_offset_minutes=${tz}`)
  } catch (err) {
    // Signed out, or offline. A recap is a nice-to-have panel; it must never break the page it
    // sits on, and the caller renders nothing rather than an error.
    if (err instanceof ApiError && err.status === 401) return null
    return null
  }
}

/** The user's play queue (ordered slugs). A 401 THROWS (#2004 #3): the store falls back to cache
 *  rather than persisting an empty queue as truth. Sole caller: stores/queue. Auth-gated. */
export async function getQueue(): Promise<string[]> {
  return (await getJSON<{ items: string[] }>("/queue")).items
}

/** Replace the play queue (auth-gated). A 401 THROWS (#2004 #11): swallowing it reported a dead
 *  session as success while nothing was persisted; _persist reverts the optimistic move on the throw
 *  and tells the caller it did not take. */
export async function putQueue(items: string[]): Promise<void> {
  const resp = await apiFetch(`${BASE}/queue`, {
    method: "PUT",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items }),
  })
  if (!resp.ok) {
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
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ slug, after: after ?? null }),
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /queue/items → ${resp.status}`)
  return ((await resp.json()) as { items: string[] }).items
}

/** Remove ONE episode from the queue. Idempotent, so a replay cannot fail. */
export async function removeQueueItem(slug: string): Promise<string[]> {
  const resp = await apiFetch(`${BASE}/queue/items/${encodeURIComponent(slug)}`, {
    method: "DELETE",
    credentials: "include",
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /queue/items → ${resp.status}`)
  return ((await resp.json()) as { items: string[] }).items
}

/** Episodes the user has marked played. A 401 THROWS (#2004 #3): the store falls back to cache
 *  rather than persisting an empty set as truth. Sole caller: stores/completed. */
export async function getCompleted(): Promise<string[]> {
  return (await getJSON<{ slugs: string[] }>("/completed")).slugs
}

/** Mark one episode played (idempotent); returns the stored slug list. */
export async function markCompleted(slug: string): Promise<string[]> {
  const resp = await apiFetch(`${BASE}/completed/${encodeURIComponent(slug)}`, {
    method: "PUT",
    credentials: "include",
  })
  if (!resp.ok) throw new ApiError(resp.status, `PUT /completed → ${resp.status}`)
  return ((await resp.json()) as { slugs: string[] }).slugs
}

/** Clear the played mark for one episode; returns the stored slug list. */
export async function unmarkCompleted(slug: string): Promise<string[]> {
  const resp = await apiFetch(`${BASE}/completed/${encodeURIComponent(slug)}`, {
    method: "DELETE",
    credentials: "include",
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /completed → ${resp.status}`)
  return ((await resp.json()) as { slugs: string[] }).slugs
}

/**
 * Record that the user STARTED an episode (listen-event log). Best-effort; ignores 401.
 *
 * Returns whether it landed, so the caller can queue it for a later flush (#1924) — it used to
 * swallow every failure indistinguishably, which is why offline listening vanished. `clientTs`
 * carries when the listen actually happened for events flushed after the fact; the server clamps
 * it, so a wrong device clock cannot write into the far past or the future.
 */
/** One record of why the native app or its WebView ended (#2279); see services/lifecycle.ts. */
export interface AppExitEntry {
  source: 'metrickit' | 'android_exit_info' | 'webview_terminated' | 'memory_warning'
  reason: string
  count: number
  at?: string
}

/**
 * Forward the device's exit records to `/api/app/app-exits` (#2279). True only when the server
 * accepted them, because the caller clears the device log on true — a false must leave the records
 * for the next launch rather than losing them.
 */
export async function postAppExits(body: {
  platform: 'ios' | 'android'
  app_version: string
  entries: AppExitEntry[]
}): Promise<boolean> {
  try {
    const resp = await apiFetch(`${BASE}/app-exits`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
    return resp.ok
  } catch {
    return false
  }
}

export async function logListen(slug: string, clientTs?: number): Promise<boolean> {
  try {
    const resp = await apiFetch(`${BASE}/listen/${encodeURIComponent(slug)}`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(clientTs ? { client_ts: clientTs } : {}),
    })
    // ANY response is an answer — except 401/403, which is an answer about the CREDENTIAL and is
    // repaired by signing in again (advisor 1.1). Reporting it as delivered discarded a week of
    // offline listening the moment a cookie expired. A 404 means the episode is gone and does not
    // improve by retrying; only 408/429 and 5xx are worth another attempt.
    if (resp.ok) return true
    if (resp.status === 401 || resp.status === 403) return false
    if (resp.status === 408 || resp.status === 429 || resp.status >= 500) return false
    return true
  } catch {
    return false
  }
}

/**
 * Record that the user reached 25 / 50 / 75 / 95 percent of an episode (#2266). Best-effort.
 *
 * An OPEN is not a listen: `logListen` says the episode was opened, this says it was actually
 * heard, which is what makes completion rate and the beta's active-day metrics mean anything.
 *
 * The retry classification is IDENTICAL to `logListen`'s, and deliberately so — the two travel in
 * the same offline queue, so a milestone that "fails" differently from an open would make the
 * queue's stop-at-first-failure behaviour depend on which kind of event happened to be next.
 */
export async function logPlaybackProgress(
  slug: string,
  milestone: number,
  clientTs?: number,
): Promise<boolean> {
  try {
    const resp = await apiFetch(`${BASE}/playback-progress/${encodeURIComponent(slug)}`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(clientTs ? { milestone, client_ts: clientTs } : { milestone }),
    })
    if (resp.ok) return true
    if (resp.status === 401 || resp.status === 403) return false
    if (resp.status === 408 || resp.status === 429 || resp.status >= 500) return false
    return true
  } catch {
    return false
  }
}

/** The signed-in user's own listening analytics; `null` when signed out (401). Auth-gated. */
export async function getMyStats(): Promise<UserStats | null> {
  try {
    return await getJSON<UserStats>("/me/stats")
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
export function loginUrl(
  as?: string,
  native = false,
  returnTo?: string,
  provider?: string,
): string {
  const params = new URLSearchParams()
  if (as) params.set("as", as)
  // Which configured provider (#2275) — `apple`; absent means the server's primary (Google).
  if (provider) params.set("provider", provider)
  // Native (#1310): tells the backend to return the signed token via the app's deep link instead of
  // setting a cookie (which an external OAuth browser can't hand back to the WebView).
  if (native) params.set("platform", "native")
  // Web full-page OAuth redirect discards the SPA's `?redirect`; carry it as `return_to` so the
  // backend (guarded by _safe_return_to) bounces back to the deep link after callback (RFC-120 #2009).
  if (returnTo) params.set("return_to", returnTo)
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
/**
 * The sign-in providers the server has configured (#2275), from `/auth/status`.
 *
 * Not from `/api/health`: the public edge answers that with the coming-soon page and the player's
 * nginx never proxies it, so a signed-out phone parsed HTML, got nothing, and the Apple button
 * never appeared in production. `/auth/status` is public and proxied — reachable before sign-in.
 */
export async function getAuthProviders(): Promise<string[]> {
  try {
    const res = await apiFetch(`${BASE}/auth/status`, { credentials: "include" })
    if (!res.ok) return []
    const body = (await res.json()) as { providers?: unknown }
    return Array.isArray(body.providers) ? body.providers.filter((p): p is string => typeof p === "string") : []
  } catch {
    return []
  }
}

export async function getDevUsers(): Promise<{ enabled: boolean; users: DevUser[] }> {
  try {
    const res = await apiFetch(`${BASE}/auth/dev-users`, { credentials: "include" })
    if (!res.ok) return { enabled: false, users: [] }
    const body = (await res.json()) as { enabled?: boolean; users?: DevUser[] }
    return { enabled: body.enabled === true, users: Array.isArray(body.users) ? body.users : [] }
  } catch {
    return { enabled: false, users: [] }
  }
}

/**
 * Ask for an email sign-in link (#2272).
 *
 * Resolves the SAME WAY whatever the address is — known, unknown, or not on the allowlist. That is
 * the server's contract and the UI must not undo it: showing "no account with that address" here
 * would rebuild the account-existence oracle the uniform response exists to prevent.
 *
 * Never throws. A network failure resolves `false` so the caller can say "something went wrong"
 * without claiming anything about the address.
 */
export async function requestMagicLink(email: string, returnTo?: string): Promise<boolean> {
  try {
    const res = await apiFetch(`${BASE}/auth/email/request`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email,
        // The native shell cannot receive a cookie set in an external browser, so it takes the
        // session by deep link instead — the same split the OAuth login route makes.
        platform: isNativeShell() ? "native" : undefined,
        return_to: returnTo,
      }),
    })
    return res.ok
  } catch {
    return false
  }
}

/** Clear the session server-side (deletes the cookie). Best-effort; resolves on 204. */
export async function logout(): Promise<void> {
  await apiFetch(`${BASE}/auth/logout`, { method: "POST", credentials: "include" })
}

/** Upload a profile avatar (Area E) — multipart to the narrow endpoint; returns the served URL.
 *  No Content-Type header: the browser sets the multipart boundary. */
export async function uploadAvatar(file: Blob): Promise<{ image: string }> {
  const form = new FormData()
  // Accept a Blob (the crop modal emits a cropped PNG Blob) or a File; give the part a filename so
  // the multipart upload always carries one.
  form.append("file", file, file instanceof File ? file.name : "avatar.png")
  const resp = await apiFetch(`${BASE}/profile/avatar`, {
    method: "POST",
    credentials: "include",
    body: form,
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /profile/avatar → ${resp.status}`)
  const body = (await resp.json()) as { image: string }
  // Absolutised for the same reason as `getMe` — the upload response carries the same relative path,
  // and a caller that renders it directly would hit `capacitor://localhost` on native.
  return { ...body, image: resolveMediaUrl(body.image) ?? body.image }
}

/** Set the signed-in user's display name. Resolves to the name as stored (whitespace collapsed);
 *  throws ApiError 400 for a name the server refuses (empty, too long, control characters). */
/**
 * Clear the signed-in person's listening history (#2273): positions, finished marks, listening time
 * and the event logs. The account, library, queue, highlights, notes and collections stay.
 */
export async function clearListeningHistory(): Promise<void> {
  const resp = await apiFetch(`${BASE}/me/history`, { method: "DELETE", credentials: "include" })
  if (resp.status !== 204) throw new ApiError(resp.status, `DELETE /me/history → ${resp.status}`)
}

/**
 * Delete the signed-in account, irreversibly (#2273). `confirm` must be the literal "DELETE" — the
 * server checks it too, so a stray call cannot delete anyone. Throws ApiError on anything but 204.
 */
export async function deleteAccount(confirm: string): Promise<void> {
  const resp = await apiFetch(`${BASE}/me`, {
    method: "DELETE",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirm }),
  })
  if (resp.status !== 204) throw new ApiError(resp.status, `DELETE /me → ${resp.status}`)
}

export async function setProfileName(name: string): Promise<{ name: string }> {
  const resp = await apiFetch(`${BASE}/profile/name`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /profile/name → ${resp.status}`)
  return (await resp.json()) as { name: string }
}

// --- P2 Capture: highlights + notes (PRD-040 / RFC-098 §7) ---

/**
 * A note's target → the `capture_created` enum (#2267).
 *
 * The app's note targets and the spec's four do not line up exactly, and anything unmapped folds
 * into `episode` rather than becoming a silent new category in the dashboards. If that fold ever
 * hides something worth separating, widen the enum deliberately — do not let an unmapped value
 * leak through.
 */
function noteTargetKind(
  target: string,
): "episode" | "topic" | "person" | "insight" | "show" | "storyline" {
  if (
    target === "topic" ||
    target === "person" ||
    target === "insight" ||
    target === "show" ||
    target === "storyline"
  ) {
    return target
  }
  // `highlight` and `episode` both mean "a note against this episode's content" — a note on a
  // highlight is a note on a moment of the episode, so folding them is honest. Anything genuinely
  // new would land here too, which is why the union above is explicit rather than a cast.
  return "episode"
}

/** The user's highlights, optionally scoped to one episode. A 401 THROWS (#2004 #3): the store
 *  falls back to cache rather than telling a user with highlights they have none. Sole caller:
 *  stores/capture. */
export async function getHighlights(episode?: string): Promise<Highlight[]> {
  return (await getJSON<{ items: Highlight[] }>("/highlights", { episode })).items
}

/** One page of the Saved highlights — whole EPISODES, each with its first few highlights. */
export interface HighlightsPage {
  items: Highlight[]
  /** Matching highlights across all pages. */
  total: number
  /** Episodes with a match, across all pages. */
  episode_total: number
  /** Matching highlights per episode on this page (`items` holds at most `perEpisode` each). */
  episode_counts: Record<string, number>
  /** The notes on the highlights in `items`. */
  notes: Note[]
}

export interface HighlightsPageQuery {
  q?: string
  color?: string | null
  muted?: boolean
  sort?: "recent" | "title"
  /** Episodes to skip. */
  offset?: number
  /** Episodes per page. */
  limit: number
  perEpisode?: number
  /** One episode only — its next highlights. */
  episode?: string
}

/**
 * The Saved highlights a page of EPISODES at a time, filtered on the server (`GET /highlights`
 * with `limit`). Against a server that predates paging (no `total` in its answer) the full list is
 * grouped and cut here instead, the way the Saved tab used to — same result either way.
 */
export async function getHighlightsPage(query: HighlightsPageQuery): Promise<HighlightsPage> {
  const resp = await getJSON<Partial<HighlightsPage> & { items: Highlight[] }>("/highlights", {
    episode: query.episode,
    q: query.q?.trim() || undefined,
    color: query.color ?? undefined,
    muted: query.muted ? "true" : undefined,
    sort: query.sort,
    offset: query.offset ?? 0,
    limit: query.limit,
    per_episode: query.perEpisode ?? 5,
  })
  if (resp.total !== undefined) return resp as HighlightsPage
  return pageHighlightsLocally(resp.items, await getNotes("highlight").catch(() => []), query)
}

export function pageHighlightsLocally(
  all: Highlight[],
  notes: Note[],
  query: HighlightsPageQuery,
): HighlightsPage {
  const needle = (query.q ?? "").trim().toLocaleLowerCase()
  const has = (...texts: (string | null | undefined)[]) =>
    !needle || texts.some((t) => (t ?? "").toLocaleLowerCase().includes(needle))
  const groups = new Map<string, Highlight[]>()
  let total = 0
  for (const h of all) {
    if (query.episode && h.episode_slug !== query.episode) continue
    if (query.color && h.color !== query.color) continue
    if (query.muted && !h.retired) continue
    if (!has(h.quote_text, h.speaker)) continue
    total++
    groups.set(h.episode_slug, [...(groups.get(h.episode_slug) ?? []), h])
  }
  const newest = (hs: Highlight[]) => Math.max(...hs.map((h) => h.created_at ?? 0))
  for (const hs of groups.values()) hs.sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0))
  // An older server has no titles to sort by here; A–Z falls back to the slug.
  const order = [...groups.keys()].sort((a, b) =>
    query.sort === "title" ? a.localeCompare(b) : newest(groups.get(b)!) - newest(groups.get(a)!),
  )
  const start = query.offset ?? 0
  const page = order.slice(start, start + query.limit)
  const items = page.flatMap((slug) => groups.get(slug)!.slice(0, query.perEpisode ?? 5))
  const ids = new Set(items.map((h) => h.id))
  return {
    items,
    total,
    episode_total: order.length,
    episode_counts: Object.fromEntries(page.map((slug) => [slug, groups.get(slug)!.length])),
    notes: notes.filter((n) => n.target === "highlight" && ids.has(n.target_id)),
  }
}

/** One page of notes. */
export interface NotesPage {
  items: Note[]
  total: number
  /** Matches per target kind under the same q, ignoring the kind filter. */
  counts: Record<string, number>
  /** The highlights the page's highlight-notes are on (their links need the episode). */
  highlights: Highlight[]
}

export interface NotesPageQuery {
  q?: string
  /** Only these target kinds; none = all. */
  kinds?: string[]
  /** Every word of `q`, in any order (Search) — rather than `q` as one phrase. */
  words?: boolean
  offset?: number
  limit: number
}

/** Notes newest first, a page at a time (`GET /notes` with `limit`); an older server pages here. */
export async function getNotesPage(query: NotesPageQuery): Promise<NotesPage> {
  const params = new URLSearchParams()
  if (query.q?.trim()) params.set("q", query.q.trim())
  for (const k of query.kinds ?? []) params.append("kinds", k)
  if (query.words) params.set("match", "words")
  params.set("offset", String(query.offset ?? 0))
  params.set("limit", String(query.limit))
  const resp = await getJSON<Partial<NotesPage> & { items: Note[] }>(`/notes?${params}`)
  if (resp.total !== undefined) return resp as NotesPage
  return pageNotesLocally(resp.items, await getHighlights().catch(() => []), query)
}

export function pageNotesLocally(all: Note[], highlights: Highlight[], query: NotesPageQuery): NotesPage {
  const needle = (query.q ?? "").trim().toLocaleLowerCase()
  const words = query.words ? needle.split(/\s+/).filter(Boolean) : [needle]
  const hits = [...all]
    .sort((a, b) => b.created_at - a.created_at)
    .filter((n) => words.every((w) => !w || n.text.toLocaleLowerCase().includes(w)))
  const counts: Record<string, number> = {}
  for (const n of hits) counts[n.target] = (counts[n.target] ?? 0) + 1
  const kinds = query.kinds ?? []
  const selected = kinds.length ? hits.filter((n) => kinds.includes(n.target)) : hits
  const start = query.offset ?? 0
  const items = selected.slice(start, start + query.limit)
  const on = new Set(items.filter((n) => n.target === "highlight").map((n) => n.target_id))
  return { items, total: selected.length, counts, highlights: highlights.filter((h) => on.has(h.id)) }
}

/** Capture a highlight (auth-gated); returns the created record. */
export async function createHighlight(body: HighlightCreate): Promise<Highlight> {
  // #2267 — one of the spec's four Umami goals, and part of "learning actions per active day".
  // NEVER the highlighted text, only that a capture happened and against what kind of thing. A
  // highlight is always episode-scoped; notes are the ones that can hang off a topic or person.
  track("capture_created", { kind: "highlight", target_kind: "episode" })
  const resp = await apiFetch(`${BASE}/highlights`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /highlights → ${resp.status}`)
  return (await resp.json()) as Highlight
}

/** Edit a highlight's colour / captured text (auth-gated); returns the updated record. */
export async function patchHighlight(id: string, body: HighlightUpdate): Promise<Highlight> {
  const resp = await apiFetch(`${BASE}/highlights/${encodeURIComponent(id)}`, {
    method: "PATCH",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!resp.ok) throw new ApiError(resp.status, `PATCH /highlights → ${resp.status}`)
  return (await resp.json()) as Highlight
}

/** Remove a highlight by id (auth-gated); returns the remaining list. */
export async function deleteHighlight(id: string): Promise<Highlight[]> {
  const resp = await apiFetch(`${BASE}/highlights/${encodeURIComponent(id)}`, {
    method: "DELETE",
    credentials: "include",
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /highlights → ${resp.status}`)
  return ((await resp.json()) as { items: Highlight[] }).items
}

/** The user's notes, optionally scoped to one target. A 401 THROWS (#2004 #3): the store falls back
 *  to cache rather than persisting an empty list as truth. Sole caller: stores/capture. */
export async function getNotes(target?: string, targetId?: string): Promise<Note[]> {
  return (await getJSON<{ items: Note[] }>("/notes", { target, target_id: targetId })).items
}

/** Attach a free-text note to a highlight / insight / episode (auth-gated). */
export async function createNote(body: NoteCreate): Promise<Note> {
  // Same rule as createHighlight: the TARGET KIND, never the note's text.
  track("capture_created", { kind: "note", target_kind: noteTargetKind(body.target) })
  const resp = await apiFetch(`${BASE}/notes`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /notes → ${resp.status}`)
  return (await resp.json()) as Note
}

/** Edit a note's text (auth-gated); returns the updated record. */
export async function patchNote(id: string, text: string): Promise<Note> {
  const body: NoteUpdate = { text }
  const resp = await apiFetch(`${BASE}/notes/${encodeURIComponent(id)}`, {
    method: "PATCH",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!resp.ok) throw new ApiError(resp.status, `PATCH /notes → ${resp.status}`)
  return (await resp.json()) as Note
}

/** Remove a note by id (auth-gated); returns the remaining list. */
export async function deleteNote(id: string): Promise<Note[]> {
  const resp = await apiFetch(`${BASE}/notes/${encodeURIComponent(id)}`, {
    method: "DELETE",
    credentials: "include",
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /notes → ${resp.status}`)
  return ((await resp.json()) as { items: Note[] }).items
}

/** The URL for the Markdown export of highlights (a download link / new tab). With `color`, the
 *  export obeys the Saved surface's colour filter — only highlights of that colour (#2042). */
function exportQuery(color?: string | null, opts?: { mutedOnly?: boolean; q?: string }): string {
  const p = new URLSearchParams()
  if (color) p.set("color", color)
  if (opts?.mutedOnly) p.set("muted_only", "true")
  const q = opts?.q?.trim()
  if (q) p.set("q", q)
  return p.toString()
}

/**
 * The whole episode as notes — summary, key points, topics, everything said, and the user's own
 * captures on it. A different document from the highlights export, which is capture-scoped across
 * every episode (operator 2026-09-18).
 */
export function episodeNotesUrl(slug: string, ext: "md" | "html"): string {
  return `${BASE}/episodes/${encodeURIComponent(slug)}/notes.${ext}`
}

/** The kinds the server draws a share card for (`/og/{kind}/{id}.png`, server/og/build.py). */
export type ShareCardKind =
  | "episode"
  | "show"
  | "topic"
  | "person"
  | "storyline"
  | "theme"
  | "organization"

/**
 * The share card as a PNG — drawn by the SERVER (`server/og/card.py`), the one card design for every
 * kind (operator 2026-10-05). The same image a shared link unfurls as. Fetched through `apiFetch`, so
 * on native it reaches the live server with the shell's credentials like every other request.
 */
export async function fetchShareCard(kind: ShareCardKind, id: string): Promise<Blob> {
  const path = `/og/${kind}/${encodeURIComponent(id)}.png`
  const resp = await apiFetch(resolveMediaUrl(path) ?? path, { credentials: "include" })
  if (!resp.ok) throw new ApiError(resp.status, `GET /og/${kind} → ${resp.status}`)
  return resp.blob()
}

/**
 * One of the user's highlights as a quote card — the same server renderer as every other card,
 * served signed-in because a highlight is private (`/api/app/highlights/{id}/card.png`).
 */
export async function fetchHighlightCard(id: string): Promise<Blob> {
  const resp = await apiFetch(`${BASE}/highlights/${encodeURIComponent(id)}/card.png`, {
    credentials: "include",
  })
  if (!resp.ok) throw new ApiError(resp.status, `GET /highlights/{id}/card.png → ${resp.status}`)
  return resp.blob()
}

/** The same export, print-styled, for the browser's Save-as-PDF (operator 2026-09-18). */
export function highlightsPrintUrl(
  color?: string | null,
  opts?: { mutedOnly?: boolean; q?: string },
): string {
  const qs = exportQuery(color, opts)
  return qs ? `${BASE}/highlights/export.html?${qs}` : `${BASE}/highlights/export.html`
}

export function highlightsExportUrl(
  color?: string | null,
  opts?: { mutedOnly?: boolean; q?: string },
): string {
  // Export mirrors the Saved filters, all of them. Colour alone was passed, so narrowing by search
  // or by muted and then pressing Export handed back a file that disagreed with the screen that
  // produced it (operator 2026-09-18).
  const qs = exportQuery(color, opts)
  return qs ? `${BASE}/highlights/export.md?${qs}` : `${BASE}/highlights/export.md`
}

/**
 * Fetch the highlights Markdown export as text — used by the native shell, where `<a download>`
 * can't save (WKWebView) so we write+share the bytes instead (#1310). Web keeps the link. Honours
 * the active colour filter when one is passed.
 */
export async function fetchHighlightsExport(
  color?: string | null,
  opts?: { mutedOnly?: boolean; q?: string },
  format: "md" | "html" = "md",
): Promise<string> {
  const url =
    format === "html" ? highlightsPrintUrl(color, opts) : highlightsExportUrl(color, opts)
  const resp = await apiFetch(url, { credentials: "include" })
  if (!resp.ok) throw new Error(`highlights export failed: ${resp.status}`)
  return resp.text()
}

/**
 * The episode-notes Markdown as TEXT — the native shell's path, where `<a download>` saves nothing.
 * Web keeps the plain download link.
 */
export async function fetchEpisodeNotes(slug: string, ext: "md" | "html" = "md"): Promise<string> {
  const resp = await apiFetch(episodeNotesUrl(slug, ext), { credentials: "include" })
  if (!resp.ok) throw new Error(`episode notes export failed: ${resp.status}`)
  return resp.text()
}

export interface ObsidianExportResult {
  mode: "full" | "incremental"
  revision: number
  /** The server's vault identity. Store it beside `revision` and send both back — a bare
   *  revision cannot identify a snapshot across a server-side state reset (#41). */
  epoch: string
  written: number
  removed: number
  /** The vault archive. Hand it to `native.deliverFile` — this layer does not touch the DOM. */
  zip: Blob
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
  const q = new URLSearchParams({ format: "obsidian", since: String(since) })
  if (epoch) q.set("epoch", epoch)
  const resp = await apiFetch(`${BASE}/export?${q}`, {
    credentials: "include",
  })
  if (!resp.ok) throw new ApiError(resp.status, `GET /export → ${resp.status}`)
  // The bytes are RETURNED, not delivered. This module used to hard-code `<a download>`, which
  // does nothing in WKWebView — so the native build hid the Obsidian button rather than fix the
  // delivery, and the export vanished on the phone (operator 2026-09-18). A transport module
  // deciding how bytes reach a human is how that divergence happened. It cannot simply call
  // `native.deliverFile` either: native.ts imports `setAuthToken` from here, so that would close an
  // import cycle. Returning the blob breaks the knot rather than tying it tighter.
  return {
    zip: await resp.blob(),
    mode: (resp.headers.get("X-Export-Mode") as "full" | "incremental") ?? "full",
    revision: Number(resp.headers.get("X-Export-Revision") ?? "0"),
    epoch: resp.headers.get("X-Export-Epoch") ?? "",
    written: Number(resp.headers.get("X-Export-Written") ?? "0"),
    removed: Number(resp.headers.get("X-Export-Removed") ?? "0"),
  }
}

// --- P3 Consolidation: spaced resurfacing (RFC-101 §5) ---

/** Highlights due to resurface (+ reflection prompt + paused flag); empty signed out (401). */
/**
 * What is due, a page of EPISODES at a time (`GET /resurfacing` with `limit`). Against a server
 * that predates paging (no `total` in its answer) the full list is grouped and cut here.
 */
export async function getResurfacingPage(query: {
  offset?: number
  limit: number
  perEpisode?: number
}): Promise<Required<ResurfacingResponse>> {
  let resp: ResurfacingResponse
  try {
    resp = await getJSON<ResurfacingResponse>("/resurfacing", {
      offset: query.offset ?? 0,
      limit: query.limit,
      per_episode: query.perEpisode ?? 100,
    })
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) resp = { items: [], paused: false }
    else throw err
  }
  if (resp.total !== undefined) return resp as Required<ResurfacingResponse>
  return pageResurfacingLocally(resp, query)
}

export function pageResurfacingLocally(
  resp: ResurfacingResponse,
  query: { offset?: number; limit: number; perEpisode?: number },
): Required<ResurfacingResponse> {
  const groups = new Map<string, ResurfacingItem[]>()
  for (const it of resp.items) {
    const slug = it.highlight.episode_slug
    groups.set(slug, [...(groups.get(slug) ?? []), it])
  }
  const order = [...groups.keys()]
  const start = query.offset ?? 0
  const page = order.slice(start, start + query.limit)
  return {
    items: page.flatMap((slug) => groups.get(slug)!.slice(0, query.perEpisode ?? 100)),
    paused: resp.paused,
    total: resp.items.length,
    episode_total: order.length,
    episode_counts: Object.fromEntries(page.map((slug) => [slug, groups.get(slug)!.length])),
  }
}

export async function getResurfacing(): Promise<ResurfacingResponse> {
  try {
    return await getJSON<ResurfacingResponse>("/resurfacing")
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return { items: [], paused: false }
    throw err
  }
}

/**
 * Stop resurfacing one highlight — it stays in Saved (operator 2026-09-18).
 *
 * The ladder's only exit. Reviewing tops out at a 90-day rung and repeats for ever; ignoring leaves
 * an item permanently overdue at the top of the list. Without this the only way to stop either was
 * deleting the capture, which answers a different question.
 */
/** Resume resurfacing a retired highlight — the undo, reachable only from Saved. */
export async function unretireHighlight(id: string): Promise<void> {
  const resp = await apiFetch(`${BASE}/resurfacing/${encodeURIComponent(id)}/retire`, {
    method: "DELETE",
    credentials: "include",
  })
  if (!resp.ok && resp.status !== 401) {
    throw new ApiError(resp.status, `DELETE /resurfacing/retire → ${resp.status}`)
  }
}

export async function retireHighlight(id: string): Promise<void> {
  const resp = await apiFetch(`${BASE}/resurfacing/${encodeURIComponent(id)}/retire`, {
    method: "POST",
    credentials: "include",
  })
  if (!resp.ok && resp.status !== 401) {
    throw new ApiError(resp.status, `POST /resurfacing/retire → ${resp.status}`)
  }
}

/** Record that the user has seen a resurfaced highlight (advances its ladder). Best-effort. */
export async function markSurfaced(id: string): Promise<void> {
  const resp = await apiFetch(`${BASE}/resurfacing/${encodeURIComponent(id)}/surfaced`, {
    method: "POST",
    credentials: "include",
  })
  if (!resp.ok && resp.status !== 401) {
    throw new ApiError(resp.status, `POST /resurfacing/surfaced → ${resp.status}`)
  }
}

/** Update resurfacing pacing (pause/resume); returns the stored settings. */
export async function putResurfacingSettings(paused: boolean): Promise<ResurfacingSettings> {
  const resp = await apiFetch(`${BASE}/resurfacing/settings`, {
    method: "PUT",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paused }),
  })
  if (!resp.ok) throw new ApiError(resp.status, `PUT /resurfacing/settings → ${resp.status}`)
  return (await resp.json()) as ResurfacingSettings
}

// --- Delivery consent: per-TYPE × per-CHANNEL notification matrix (#1414 → wave-I) ---

const COMMS_CHANNELS_DEFAULT = { email: false, push: false, in_app: true }
const COMMS_DEFAULTS: CommsSettings = {
  types: {
    digest: { ...COMMS_CHANNELS_DEFAULT },
    daily_recap: { ...COMMS_CHANNELS_DEFAULT },
    new_episodes: { ...COMMS_CHANNELS_DEFAULT },
    product: { ...COMMS_CHANNELS_DEFAULT },
  },
  digest_schedule: { cadence: "weekly", day_of_week: 6, hour: 13, paused: false },
  email_verified: false,
  timezone: "",
  unsubscribe_ref: null,
}

export async function getComms(): Promise<CommsSettings> {
  try {
    return await getJSON<CommsSettings>("/comms")
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
    return await getJSON<YourWeekResponse>("/your-week")
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) {
      return { sections: [], period_label: "", generated_at: "" }
    }
    throw err
  }
}

export async function putComms(update: CommsUpdate): Promise<CommsSettings> {
  const resp = await apiFetch(`${BASE}/comms`, {
    method: "PUT",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(update),
  })
  if (!resp.ok) throw new ApiError(resp.status, `PUT /comms → ${resp.status}`)
  return (await resp.json()) as CommsSettings
}

/**
 * Persist the browser's IANA timezone so digests land at the user's local hour (#2041). Called
 * once on boot for a signed-in user; a no-op-ish PUT the server merges. Best-effort — a failure
 * just leaves the stored tz as-is (UTC fallback), so it never blocks boot.
 */
export function detectTimezone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || ""
  } catch {
    return ""
  }
}

export async function putTimezone(timezone: string): Promise<void> {
  if (!timezone) return
  await putComms({ timezone })
}

/** The public VAPID key the browser needs to subscribe (throws 503 when push isn't configured). */
export async function getVapidKey(): Promise<string> {
  const resp = await getJSON<{ key: string }>("/push/vapid-key")
  return resp.key
}

/** Register a browser push subscription (endpoint only; per-type push consent is a matrix toggle). */
export async function subscribePush(subscription: unknown): Promise<{ count: number }> {
  const resp = await apiFetch(`${BASE}/push/subscribe`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(subscription),
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /push/subscribe → ${resp.status}`)
  return (await resp.json()) as { count: number }
}

/** Remove a browser push subscription (disables the channel when the last one goes). */
export async function unsubscribePush(endpoint: string): Promise<{ count: number }> {
  const resp = await apiFetch(`${BASE}/push/subscribe`, {
    method: "DELETE",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ endpoint }),
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /push/subscribe → ${resp.status}`)
  return (await resp.json()) as { count: number }
}

// --- In-app notification inbox (wave-I, the in_app channel) ---

/**
 * The user's inbox + unread count. A 401 returns an empty inbox rather than throwing — the bell
 * is a passive surface that simply shows nothing when signed out (no sign-in prompt to trigger).
 */
export async function getNotifications(): Promise<NotificationsResponse> {
  try {
    return await getJSON<NotificationsResponse>("/notifications")
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return { items: [], unread: 0 }
    throw err
  }
}

/** Mark one notification read; returns the fresh unread count. */
export async function markNotificationRead(id: string): Promise<{ unread: number }> {
  const resp = await apiFetch(`${BASE}/notifications/${encodeURIComponent(id)}/read`, {
    method: "POST",
    credentials: "include",
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /notifications/${id}/read → ${resp.status}`)
  return (await resp.json()) as { unread: number }
}

/** Mark every notification read; returns the fresh unread count (0). */
export async function markAllNotificationsRead(): Promise<{ unread: number }> {
  const resp = await apiFetch(`${BASE}/notifications/read-all`, {
    method: "POST",
    credentials: "include",
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /notifications/read-all → ${resp.status}`)
  return (await resp.json()) as { unread: number }
}

// --- Key voices (wave-G, per-user) ---

/**
 * The signed-in user's key voices (most-present people in their own corpus). A 401 returns an
 * empty rail rather than throwing — the rail is a passive surface that hides when there's nothing.
 */
export async function getKeyVoices(limit = 8): Promise<KeyVoicesResponse> {
  try {
    /*
     * ABSOLUTISED, like every other surface carrying a person photo (operator 2026-09-27: "no
     * people images here on home page").
     *
     * The server sets `image_url` correctly and `KeyVoicesRail` renders it into `ProfileAvatar`.
     * But the server returns it RELATIVE (`/api/app/persons/<id>/photo`), and inside the Capacitor
     * WebView the document origin is `capacitor://localhost` — so a relative path resolves THERE,
     * 404s, and `ProfileAvatar` falls back to initials. On the web, same server and same user, it
     * works, because the origins match. Silent either way: a 404 on an `<img>` is not an error
     * anyone sees, and initials are a legitimate-looking state.
     *
     * Same defect the artwork, audio and profile-avatar URLs each hit in turn, and the reason
     * `withAbsoluteEntityImages` exists. This rail was the one people-carrying endpoint never
     * routed through it. Mapped directly rather than reusing that helper because the payload has
     * no `web` / `related_people` shape — it is a flat `voices` list.
     */
    const resp = await getJSON<KeyVoicesResponse>(`/key-voices?limit=${limit}`)
    return {
      ...resp,
      voices: (resp.voices ?? []).map((v) =>
        v.image_url ? { ...v, image_url: resolveMediaUrl(v.image_url) ?? v.image_url } : v,
      ),
    }
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return { voices: [] }
    throw err
  }
}

// --- Health / version (wave-I.6 update check) ---

/**
 * The server's health facts the player acts on. Returns null on any failure — the update check and
 * the sign-out guard are best-effort and must never throw into a boot path. `player_version` is the
 * released player-app version (same scale as the baked `__APP_VERSION__`); null when unset.
 *
 * Read from `/auth/status`, not `/api/health`: in production the public edge answers `/api/health`
 * with the coming-soon page and the player nginx never proxies it, so a phone parsed HTML, got null,
 * and the update prompt never fired while the "is the server unwell?" guard ran blind. The server
 * computes these fields once and serves them on both routes (`player_client_health`).
 */
export async function getHealth(): Promise<HealthInfo | null> {
  try {
    const resp = await apiFetch(`${BASE}/auth/status`, { credentials: "include" })
    if (!resp.ok) return null
    const body = (await resp.json()) as {
      providers?: unknown
      player_version?: string | null
      auth_ready?: boolean
      auth_epoch?: string | null
    }
    return {
      player_version: body.player_version ?? null,
      auth_ready: body.auth_ready,
      auth_epoch: body.auth_epoch ?? null,
      auth_providers: Array.isArray(body.providers)
        ? body.providers.filter((p): p is string => typeof p === "string")
        : [],
    }
  } catch {
    return null
  }
}

// --- Collections / boards (PRD-046 FR4 / #1417) ---

export async function getCollections(): Promise<Collection[]> {
  /**
   * A 401 used to be swallowed into an empty list (#2004 item 13).
   *
   * "You have no collections" and "we could not ask" are different statements, and rendering the
   * second as the first is how a user comes to believe their collections were never saved: the
   * create succeeds, the row appears in the local popup state, and every later read reports empty.
   * Both surfaces that show collections call THIS function, so the lie was told in one place and
   * believed in two.
   *
   * A 401 is now a real error rather than an empty list. `/library` is `requiresAuth`, so a 401
   * there means an expired session rather than "signed out".
   *
   * Callers currently surface it as a retryable error, NOT as a sign-in prompt — which is the right
   * end state (the app has `gated()` for exactly that) and is not built yet. Recorded on #2004 so
   * the gap is visible rather than implied by this comment.
   */
  return withAbsoluteCovers((await getJSON<{ items: Collection[] }>("/collections")).items)
}

/**
 * Which collections already hold an item.
 *
 * Its own call rather than a flag on the list, because membership is a question about the ITEM.
 * `checked` distinguishes "we looked and it is in none of them" from "we could not look" — an
 * empty `ids` under `checked: false` must never render as a confident "not added", since that is
 * the answer a user acts on by saving the thing twice.
 */
export async function getCollectionsContaining(
  item: CollectionItemRef
): Promise<{ ids: string[]; checked: boolean }> {
  const q = `?kind=${encodeURIComponent(item.kind)}&ref=${encodeURIComponent(item.ref)}`
  return await getJSON<{ ids: string[]; checked: boolean }>(`/collections/containing${q}`)
}

/**
 * Absolutise each board's `cover_url` for the native shell.
 *
 * The API returns it RELATIVE (`/api/app/artwork?ref=…`). On web that is correct — the app and the
 * API share an origin. On native the document origin is `capacitor://localhost`, so the same string
 * resolves into the app bundle and the thumbnail renders as a broken image (operator 2026-09-17).
 * `fetch` was never affected because `apiFetch` prefixes an absolute base itself; `<img src>` has
 * nothing doing that for it — the identical trap the avatar hit.
 */
function withAbsoluteCovers(items: Collection[]): Collection[] {
  return items.map(withAbsoluteCover)
}

/**
 * The same for ONE board — every call that returns a board goes through this (2026-10-09).
 *
 * Only the list calls did. Create / add-item / remove-item / the board read returned the cover
 * RELATIVE, which was harmless while only the Boards view's own copy held them; once those answers
 * reached the shared store (so Library and Home's teaser show a board the moment it changes), a
 * board just added to painted a broken image on the device (operator 2026-10-09).
 */
function withAbsoluteCover(c: Collection): Collection {
  return c.cover_url ? { ...c, cover_url: resolveMediaUrl(c.cover_url) ?? c.cover_url } : c
}

export async function getCollection(id: string): Promise<CollectionDetail> {
  const detail = await getJSON<CollectionDetail>(`/collections/${encodeURIComponent(id)}`)
  return { ...detail, collection: withAbsoluteCover(detail.collection) }
}

/**
 * One page of a board's items (`GET /collections/{id}` with `limit`, 1.0.3), optionally of one
 * kind. Against an older server (no `total`) the whole board is cut here.
 */
export async function getCollectionPage(
  id: string,
  query: { limit: number; offset?: number; kind?: string },
): Promise<Required<CollectionDetail>> {
  const resp = await getJSON<CollectionDetail>(`/collections/${encodeURIComponent(id)}`, {
    limit: query.limit,
    offset: query.offset ?? 0,
    kind: query.kind,
  })
  const page = { ...resp, collection: withAbsoluteCover(resp.collection) }
  if (page.total !== undefined) return page as Required<CollectionDetail>
  return pageCollectionLocally(page, query)
}

export function pageCollectionLocally(
  resp: CollectionDetail,
  query: { limit: number; offset?: number; kind?: string },
): Required<CollectionDetail> {
  const kindCounts: Record<string, number> = {}
  for (const it of resp.items) kindCounts[it.kind] = (kindCounts[it.kind] ?? 0) + 1
  const selected = query.kind ? resp.items.filter((i) => i.kind === query.kind) : resp.items
  const start = query.offset ?? 0
  return {
    collection: resp.collection,
    items: selected.slice(start, start + query.limit),
    total: selected.length,
    kind_counts: kindCounts,
  }
}

// `clientId` (a `col_…` id minted offline) makes the create idempotent: a replay returns the
// existing row (#2004), so a pin queued against that same id still lands instead of 404ing.
export async function createCollection(name: string, clientId?: string): Promise<Collection> {
  const resp = await apiFetch(`${BASE}/collections`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(clientId ? { name, client_id: clientId } : { name }),
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /collections → ${resp.status}`)
  return withAbsoluteCover((await resp.json()) as Collection)
}

export async function deleteCollection(id: string): Promise<Collection[]> {
  const resp = await apiFetch(`${BASE}/collections/${encodeURIComponent(id)}`, {
    method: "DELETE",
    credentials: "include",
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /collections/${id} → ${resp.status}`)
  return withAbsoluteCovers(((await resp.json()) as { items: Collection[] }).items)
}

/** Persist the manual board order (CO.7). Returns the server's full list, not the optimistic one. */
export async function reorderCollections(order: string[]): Promise<Collection[]> {
  const resp = await apiFetch(`${BASE}/collections/order`, {
    method: "PATCH",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ order }),
    // The ONE write the user starts and then immediately walks away from: a drag ends with a drop,
    // and the next thing they do is leave the screen. Nothing awaits this call (the drop handler
    // fires it with `void`), so without `keepalive` a navigation or a backgrounded tab ABORTS the
    // request mid-flight and the arrangement they just made is gone. The outbox catches that case
    // now, but only on the NEXT boot — the order flashes back to the old one in between.
    // `keepalive` lets the request finish on its own after the page goes away, which is what makes
    // the drop durable rather than merely recoverable. The body is a list of ids, far under the
    // 64 KB keepalive cap.
    keepalive: true,
  })
  if (!resp.ok) throw new ApiError(resp.status, `PATCH /collections/order → ${resp.status}`)
  return withAbsoluteCovers(((await resp.json()) as { items: Collection[] }).items)
}

export async function addToCollection(id: string, item: CollectionItemRef): Promise<Collection> {
  // #2267. No props: the spec gives this event none, and the collection's name is user-typed, so
  // there is nothing here that could be reported without breaking the no-free-text rule.
  track("collection_add")
  const resp = await apiFetch(`${BASE}/collections/${encodeURIComponent(id)}/items`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(item),
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /collections/${id}/items → ${resp.status}`)
  return withAbsoluteCover((await resp.json()) as Collection)
}

export async function removeFromCollection(
  id: string,
  kind: string,
  ref: string
): Promise<Collection> {
  const q = `kind=${encodeURIComponent(kind)}&ref=${encodeURIComponent(ref)}`
  const resp = await apiFetch(`${BASE}/collections/${encodeURIComponent(id)}/items?${q}`, {
    method: "DELETE",
    credentials: "include",
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /collections/${id}/items → ${resp.status}`)
  return withAbsoluteCover((await resp.json()) as Collection)
}

// --- MCP "Connected agents" (RFC-112 §5): connector config + personal-access tokens ---

/** The connector wiring the Profile section shows (resource URL + OAuth status). mcp_access-gated. */
export async function getMcpConfig(): Promise<McpConnectionConfig> {
  const resp = await apiFetch(`${BASE}/mcp/config`, { credentials: "include" })
  if (!resp.ok) throw new ApiError(resp.status, `GET /mcp/config → ${resp.status}`)
  return (await resp.json()) as McpConnectionConfig
}

/** List the user's MCP tokens (metadata only — the secret is never returned after creation). */
export async function getMcpTokens(): Promise<McpTokenMeta[]> {
  const resp = await apiFetch(`${BASE}/mcp/tokens`, { credentials: "include" })
  if (!resp.ok) throw new ApiError(resp.status, `GET /mcp/tokens → ${resp.status}`)
  return ((await resp.json()).items ?? []) as McpTokenMeta[]
}

/** Mint a token; the plaintext is returned ONCE (copy-then-forget). */
export async function createMcpToken(label: string): Promise<McpTokenCreated> {
  const resp = await apiFetch(`${BASE}/mcp/tokens`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  })
  if (!resp.ok) throw new ApiError(resp.status, `POST /mcp/tokens → ${resp.status}`)
  return (await resp.json()) as McpTokenCreated
}

/** Revoke a token by id; returns the remaining tokens. */
export async function revokeMcpToken(id: string): Promise<McpTokenMeta[]> {
  const resp = await apiFetch(`${BASE}/mcp/tokens/${encodeURIComponent(id)}`, {
    method: "DELETE",
    credentials: "include",
  })
  if (!resp.ok) throw new ApiError(resp.status, `DELETE /mcp/tokens/${id} → ${resp.status}`)
  return ((await resp.json()).items ?? []) as McpTokenMeta[]
}

/** List the OAuth agents (claude.ai etc.) the user has connected. */
export async function getMcpConnections(): Promise<McpConnection[]> {
  const resp = await apiFetch(`${BASE}/mcp/connections`, { credentials: "include" })
  if (!resp.ok) throw new ApiError(resp.status, `GET /mcp/connections → ${resp.status}`)
  return ((await resp.json()).items ?? []) as McpConnection[]
}

/** Disconnect an OAuth agent (forget consent + drop its live tokens); returns the remaining. */
export async function revokeMcpConnection(clientId: string): Promise<McpConnection[]> {
  const resp = await apiFetch(`${BASE}/mcp/connections/${encodeURIComponent(clientId)}`, {
    method: "DELETE",
    credentials: "include",
  })
  if (!resp.ok)
    throw new ApiError(resp.status, `DELETE /mcp/connections/${clientId} → ${resp.status}`)
  return ((await resp.json()).items ?? []) as McpConnection[]
}
