/**
 * TypeScript mirrors of the `/api/app/*` response shapes (RFC-098/RFC-099). Kept in sync
 * with src/podcast_scraper/server/schemas.py. Only the shapes the client consumes today are
 * declared; extend as surfaces land.
 */

export interface Me {
  user_id: string
  email: string
  name: string
  /** Immutable public handle (@…), auto-derived at account creation (Area E). */
  username?: string
  /** Avatar URL — the OAuth provider picture captured at login, or a user upload (Area E). */
  image?: string | null
  /** RFC-112: holds the MCP entitlement — gates the "Connected agents" UI. */
  mcp_access?: boolean
}

/** One MCP personal-access token's metadata (GET /api/app/mcp/tokens — never the secret). */
export interface McpTokenMeta {
  id: string
  label: string
  created_at: number
  last_used_at: number | null
}

/** The freshly-minted token — the plaintext is shown ONCE (POST /api/app/mcp/tokens). */
export interface McpTokenCreated {
  token: string
  meta: McpTokenMeta
}

/** Connector wiring for the "Connected agents" section (GET /api/app/mcp/config). */
export interface McpConnectionConfig {
  connector_url: string | null
  authorization_server: string | null
  oauth_enabled: boolean
}

/** One connected OAuth agent (a remembered consent), revocable (GET /api/app/mcp/connections). */
export interface McpConnection {
  client_id: string
  client_name: string
  scopes: string[]
  connected_at: number
  /**
   * Unix time the agent last made an authenticated request, or `null` when it never has.
   *
   * `null` also covers every connection made before the server kept this log — the two are the
   * same fact (no evidence it ran), so the UI says "not used yet" rather than guessing a date.
   * Coalesced server-side to ~5 minutes.
   */
  last_used_at: number | null
}

export type EpisodeStatus = "ready" | "pending"

/** One catalog card (GET /api/app/episodes — AppEpisodeSummary). */
export interface EpisodeSummary {
  slug: string
  title: string
  feed_id: string
  podcast_title: string | null
  publish_date: string | null
  duration_seconds: number | null
  episode_image_url: string | null
  feed_image_url: string | null
  /** Preferred artwork (our locally-stored copy, thumb size) when present; else use image urls. */
  artwork_url: string | null
  status: EpisodeStatus
  /** Short, clean one-line lede for the card (NOT the bullets joined). */
  summary_preview: string | null
  /** Full prose summary, for the card's hover/expand preview (null when absent). */
  summary_text: string | null
  /** Summary bullets for the card (capped for size); count is `summary_bullet_count`. */
  summary_bullets: string[]
  /** TRUE key-point count, uncapped — `summary_bullets` is a capped preview, so its length pins
   *  at the cap (that was the "every episode has 8 key points" bug). */
  summary_bullet_count?: number
  topics: string[]
  has_transcript: boolean
  has_summary: boolean
  has_gi: boolean
  has_kg: boolean
  has_bridge: boolean
}

/** Paginated catalog list (AppEpisodesResponse). */
export interface EpisodesPage {
  items: EpisodeSummary[]
  page: number
  page_size: number
  total: number
  has_more: boolean
}

/** Episode detail (GET /api/app/episodes/{slug} — AppEpisodeDetail). */
export interface EpisodeDetail {
  slug: string
  title: string
  feed_id: string
  podcast_title: string | null
  publish_date: string | null
  duration_seconds: number | null
  episode_image_url: string | null
  feed_image_url: string | null
  /** Preferred artwork (our locally-stored copy, large size for the player) when present. */
  artwork_url: string | null
  summary_title: string | null
  summary_bullets: string[]
  summary_text: string | null
  has_transcript: boolean
  has_summary: boolean
  has_gi: boolean
  has_kg: boolean
  has_bridge: boolean
}

export interface ListEpisodesParams {
  page?: number
  pageSize?: number
  status?: EpisodeStatus
  feedId?: string
}

/** One transcript segment (segments.json contract). */
export interface Segment {
  id: string
  start: number
  end: number
  text: string
  speaker: string | null
}

export interface SegmentsResponse {
  version: string
  episode_slug: string
  segments: Segment[]
}

/** Origin audio descriptor (GET /api/app/episodes/{slug}/audio-source). */
export interface AudioSource {
  episode_slug: string
  url: string
  mime: string | null
  duration_seconds: number | null
  media_id: string | null
  strategy: string
  resolved_url: string | null
  verified: boolean | null
  content_length: number | null
}

/** Per-user saved playback position (auth-gated). */
export interface PlaybackPosition {
  slug: string
  position_seconds: number
  updated_at: number | null
  /** The listener reached the end — set on `ended` or at the completion threshold. Optional so a
   *  record written before the flag existed still parses; absent means unfinished. */
  finished?: boolean
}

/** A distinct show in the corpus (GET /api/app/podcasts) — public, not per-user. */
export interface Podcast {
  feed_id: string
  title: string | null
  artwork_url: string | null
  image_url: string | null
  description: string | null
  /** Podcast category/genre when known (BS.1); null when the feed carried none. */
  category?: string | null
  episode_count: number
}

/**
 * A feed the user is subscribed to (GET/POST/DELETE /api/app/library) — auth-gated per-user state,
 * distinct from the public corpus catalog above and from interest tokens (topic:/person:). This is
 * the store the "Your Week" digest reads for its "new in your follows" section.
 */
export interface LibraryItem {
  feed_id: string
  feed_url: string | null
  title: string | null
  added_at: number | null
}

/** Show-level signals for the consumer show page (GET /api/app/podcasts/{feed_id}/signals). */
export interface PodcastSignals {
  feed_id: string
  episode_count: number
  top_topics: Array<{
    topic_id: string
    label: string
    episode_count: number
    velocity: number | null
    /** Episodes in the whole corpus that mention the topic; null if unknown. */
    corpus_episode_count: number | null
    /** Episodes in the whole corpus — the denominator behind `lift`. */
    corpus_episode_total: number | null
    /**
     * Distinctiveness: the topic's share of this show's episodes over its share of the corpus.
     * 1.0 = exactly the corpus base rate (says nothing about this show in particular); >1 = the
     * show is unusually focused on it. null when the corpus base rate is unavailable.
     */
    lift: number | null
  }>
  key_people: Array<{ person_id: string; name: string; episode_count: number }>
  recurring_guests: Array<{ person_id: string; name: string; episode_count: number }>
  dominant_themes: Array<{
    theme_id: string
    label: string
    topic_count: number
    anchor_topic_id: string | null
  }>
  trending_topics: Array<{
    topic_id: string
    label: string
    velocity: number
    episode_count: number
  }>
}

/** A verbatim quote supporting an insight. */
export interface Quote {
  text: string
  speaker: string | null
  char_start: number | null
  char_end: number | null
  start_ms: number | null
  end_ms: number | null
}

/** A grounded GIL insight (with supporting quotes). */
export interface Insight {
  id: string
  text: string
  grounded: boolean
  insight_type: string | null
  confidence: number | null
  position_hint: string | null
  /** ADR-135/#1191 route-and-tag. Server returns insights already salience-sorted; the player
   *  shows `surface`-tagged only (null = pre-3.1 corpus, kept for back-compat). */
  salience?: number | null
  rank?: number | null
  routing_tag?: "surface" | "connect" | "drop" | null
  tier?: number | null
  /** #2032 — topic-perspective insights carry their source episode + the supporting quote's
   *  start moment, so the take links to `/episode/:slug?t=<start_ms/1000>`. Absent elsewhere. */
  episode_slug?: string | null
  start_ms?: number | null
  quotes: Quote[]
}

export interface InsightsResponse {
  episode_slug: string
  insights: Insight[]
}

/**
 * Post-episode recap (GET /api/app/episodes/{slug}/recap — AppEpisodeRecap, RFC-122 #2038).
 *
 * One reinforcement model the panel renders when an episode finishes: the summary key points, the
 * top salience-ranked insights, and the single strongest attributed quote (the anchor). "More like
 * this" is a separate call (`getRelated`), so this stays a pure read over one episode's artifacts.
 */
export interface EpisodeRecap {
  slug: string
  title: string | null
  podcast_title: string | null
  artwork_url: string | null
  /** Summary bullet points — the gist to consolidate. */
  key_points: string[]
  /** Full summary paragraph — a fallback lede when there are no bullets. */
  summary_text: string | null
  insights: Insight[]
  /** The memorable anchor; null when the episode has no grounded, quoted insight. */
  signature_quote: Quote | null
  has_gi: boolean
}

/** A KG person/org entity. */
export interface Entity {
  id: string
  name: string
  kind: "person" | "org"
  /** Speaker role in the episode KG (host / guest / mentioned); null for orgs / older data. */
  role?: string | null
  /** Served hosted-photo route when the web enricher has a photo for this person, else null.
   *  Surfaced on people-list surfaces (Top voices, related people) that opt to show an avatar. */
  image_url?: string | null
}

/** A KG topic. Cluster fields (RFC-102) drive cluster-first grouping; null/0 = singleton/no artifact.
 *  `cluster_*` = semantic ("Similar"); `theme_cluster_*` = co-occurrence ("Theme"). */
export interface Topic {
  id: string
  label: string
  cluster_id: string | null
  cluster_label: string | null
  cluster_size: number
  theme_cluster_id?: string | null
  theme_cluster_label?: string | null
  theme_cluster_size?: number
}

export interface EntitiesResponse {
  episode_slug: string
  persons: Entity[]
  orgs: Entity[]
  topics: Topic[]
}

/**
 * Saveable favorite kinds. `insight` is NOT one — an insight is a capture, saved via the
 * highlights path, never a favorite (RFC-121 / #1593).
 */
export type FavoriteKind = "episode" | "person" | "topic" | "show" | "storyline"

/** Body for PUT /api/app/favorites — denormalized so the Library renders without re-fetching. */
export interface FavoriteAdd {
  kind: FavoriteKind
  ref: string
  label?: string
  sublabel?: string
  slug?: string
}

/** A saved non-episode favorite (show / topic / person / storyline). */
export interface FavoriteEntity {
  kind: "person" | "topic" | "show" | "storyline"
  ref: string
  label: string
  sublabel?: string | null
}

/** The user's favorites (GET/PUT/DELETE /api/app/favorites). */
export interface FavoritesResponse {
  episodes: EpisodeSummary[]
  entities?: FavoriteEntity[]
}

// --- P2 Capture: highlights + notes (PRD-040 / RFC-098 §7) ---

export type HighlightKind = "span" | "moment" | "insight"

/** A captured highlight (GET/POST/PATCH/DELETE /api/app/highlights — the Highlight schema). */
export interface Highlight {
  id: string
  episode_slug: string
  kind: HighlightKind
  start_ms: number | null
  end_ms: number | null
  char_start: number | null
  char_end: number | null
  segment_ids: string[]
  quote_text: string | null
  speaker: string | null
  source_insight_id: string | null
  color: string | null
  created_at: number
  /** 'anchored' | 'drifted' after a re-anchor on re-scrape; null until then. */
  anchor_status: string | null
  /** Canonical person/topic refs (#1419) — the highlight as a graph node. Optional: absent on
   *  pre-#1419 highlights and when the episode has no KG, so callers must guard (`?? []`). */
  graph_refs?: EntityRef[]
}

/** Body for POST /api/app/highlights. */
export interface HighlightCreate {
  episode_slug: string
  kind: HighlightKind
  start_ms?: number | null
  end_ms?: number | null
  char_start?: number | null
  char_end?: number | null
  segment_ids?: string[]
  quote_text?: string | null
  speaker?: string | null
  source_insight_id?: string | null
  color?: string | null
  /**
   * Client-minted id (#1925). Supplying one makes the POST idempotent, which is what lets a
   * capture made offline sit in the outbox and be replayed: without a key, a retry whose response
   * was lost creates a duplicate. Re-posting an existing one returns the stored row unchanged.
   */
  client_id?: string
}

/** Body for PATCH /api/app/highlights/{id} — edit colour / captured text. */
export interface HighlightUpdate {
  color?: string | null
  quote_text?: string | null
}

export interface HighlightsResponse {
  items: Highlight[]
}

export type NoteTarget =
  | "highlight"
  | "insight"
  | "episode"
  | "show"
  | "topic"
  | "person"
  | "storyline"

/** A free-text note (GET/POST/PATCH/DELETE /api/app/notes — the Note schema). */
export interface Note {
  id: string
  target: NoteTarget
  target_id: string
  text: string
  created_at: number
  updated_at: number
}

/** Body for POST /api/app/notes. */
export interface NoteCreate {
  target: NoteTarget
  target_id: string
  text: string
  /** Client-minted id — see `HighlightCreate.client_id` (#1925). */
  client_id?: string
}

/** Body for PATCH /api/app/notes/{id}. */
export interface NoteUpdate {
  text: string
}

export interface NotesResponse {
  items: Note[]
}

// --- P3 Consolidation: spaced resurfacing (RFC-101 §5) ---

export interface ResurfacingItem {
  highlight: Highlight
  reflection_prompt: string
}

export interface ResurfacingResponse {
  items: ResurfacingItem[]
  paused: boolean
}

export interface ResurfacingSettings {
  paused: boolean
}

// --- Collections / boards — curation layer (PRD-046 FR4 / #1417) ---

export interface Collection {
  id: string
  name: string
  created_at: number
  /** Unix time of the last membership change (last-modified). */
  updated_at?: number
  count: number
  /** Derived cover thumbnail (CO.6): first episode/highlight member's artwork; null when none. */
  cover_url?: string | null
}

/** A pinnable kind (RFC-119). */
export type CollectionItemKind =
  | "highlight"
  | "episode"
  | "show"
  | "search"
  | "topic"
  | "person"
  | "link"

/** A typed reference to add to a collection. */
export interface CollectionItemRef {
  kind: CollectionItemKind
  ref: string
  scope?: string | null
  title?: string | null
}

/** A resolved collection item (server fills display fields best-effort). */
export interface CollectionItem {
  kind: CollectionItemKind
  ref: string
  title?: string | null
  subtitle?: string | null
  artwork_url?: string | null
  deep_link?: string | null
  scope?: string | null
}

export interface CollectionDetail {
  collection: Collection
  items: CollectionItem[]
}

// --- Delivery consent: per-TYPE × per-CHANNEL notification matrix (#1414 → wave-I) ---

/** The notification types a user tunes independently per channel. */
export type CommsType = "digest" | "new_episodes" | "product"
/** email/push are outbound (opt-in); in_app is the in-app inbox (default on). */
export type CommsChannel = "email" | "push" | "in_app"

export type CommsChannels = Record<CommsChannel, boolean>

/** The full matrix: each type's per-channel toggles. */
export type CommsMatrix = Record<CommsType, CommsChannels>

/** The digest email cadence — not per-channel, so it sits outside the matrix. */
export interface CommsSchedule {
  cadence: "weekly" | "daily"
  day_of_week: number
  hour: number
  paused: boolean
}

export interface CommsSettings {
  types: CommsMatrix
  digest_schedule: CommsSchedule
  email_verified: boolean
  unsubscribe_ref: string | null
}

/**
 * PUT /api/app/comms body. Send the FULL `types` matrix you want in effect — the server merges
 * known type/channel keys, so a partial matrix silently resets the omitted cells. The client
 * holds current state and PUTs it whole.
 */
export interface CommsUpdate {
  types?: CommsMatrix
  digest_schedule?: CommsSchedule
}

// --- In-app notification inbox (wave-I, the in_app channel) ---

export interface NotificationItem {
  id: string
  /** Notification type — usually a {@link CommsType} (digest / new_episodes / product), but the
   *  server sends a free string, so treat it as opaque (don't switch on it exhaustively). */
  type: string
  title: string
  body?: string | null
  deep_link?: string | null
  read: boolean
  created_at: number
}

/** GET /api/app/notifications — the inbox + unread count for the bell badge. */
export interface NotificationsResponse {
  items: NotificationItem[]
  unread: number
}

/** One person in the user's key-voices rail (wave-G). */
export interface KeyVoice {
  id: string
  kind: "person"
  label: string
  episode_count: number
  /** Served hosted-photo route when the web enricher has a photo for this person, else null. */
  image_url?: string | null
}

/** GET /api/app/key-voices — the signed-in user's most-present people. */
export interface KeyVoicesResponse {
  voices: KeyVoice[]
}

/** GET /api/health (subset the client cares about). */
export interface HealthInfo {
  /** Backend package version — NOT comparable to the client app version. */
  code_version: string
  /** Released player-app version (same scale as `__APP_VERSION__`); null when the deploy is unset. */
  player_version: string | null
}

/** A graph entity referenced by a Your Week item (person/topic) — GET /api/app/your-week. */
export interface YourWeekGraphRef {
  id: string
  kind: string
  label: string
}

/** One item in a Your Week section. Shapes vary by section kind, so most fields are optional;
 *  `episode_slug` + `episode_title` are always present, `quote`/`t_ms` only on `revisit`. */
export interface YourWeekItem {
  episode_slug: string
  /** Route-backfilled from the catalog; absent only when a slug no longer resolves (card falls
   *  back to the lead graph label). */
  episode_title?: string
  deep_link: string
  quote?: string
  t_ms?: number
  graph_refs?: YourWeekGraphRef[]
  source?: string
  /** The user's own highlight behind this item — present only for `source: 'user'` captures.
   *  Carried into the player as `?revisit=` so arriving advances its spaced ladder (#35);
   *  auto-picks have no ladder and so no id. */
  highlight_id?: string
  /** Episode/show artwork used as the card backdrop (in-app enrichment; absent → flat card). */
  image_url?: string | null
}

export type YourWeekSectionKind =
  | "revisit"
  | "new_in_follows"
  | "new_in_interests"
  | "trending_in_your_corpus"

export interface YourWeekSection {
  kind: YourWeekSectionKind
  items: YourWeekItem[]
}

/** GET /api/app/your-week — the same rollup the email digest sends, decoupled from email consent. */
export interface YourWeekResponse {
  sections: YourWeekSection[]
  period_label: string
  generated_at: string
}

/** One selectable interest cluster (GET /api/app/clusters — AppInterestCluster). */
export interface InterestCluster {
  id: string
  label: string
  size: number
}

/** One storyline — a THEME cluster (topics discussed together). GET /api/app/theme-clusters.
 *  `id` is the `thc:` interest token; `anchor_topic_id` is the representative topic card to open. */
export interface Storyline {
  id: string
  label: string
  size: number
  anchor_topic_id: string
}

/** One trending entity (RFC-103 momentum). GET /api/app/trending?kind=… */
export interface TrendingEntity {
  entity_id: string
  kind: string
  label: string
  velocity: number
  volume: number
  heating_up: boolean
  total: number
  series: number[]
  /** Headline speaker role (host/guest/mentioned) for person entities; null otherwise. */
  role?: string | null
  /** Trend window this row was ranked under (1m|3m|6m|1y). RFC-103 R2. */
  window?: string
  /** Served hosted-photo route for a person entity with a hosted photo; null otherwise. */
  image_url?: string | null
}

/** A resolved person/topic reference (GET /api/app/entities/search — AppEntityRef). */
export interface EntityRef {
  id: string
  kind: "person" | "topic" | "organization"
  label: string
}

/** Entity-in-search resolution (AppEntitySearchResponse) — at most one exact/near-exact match. */
export interface EntitySearchResponse {
  query: string
  entity: EntityRef | null
}

/** One show a person appears in, with their role there (AppPersonShow). Per-show, not global. */
export interface PersonShow {
  feed_id: string
  title: string
  /** Aggregate role within this show (host / guest / mentioned); null when unknown. */
  role?: string | null
  episode_count: number
}

/** Optional external bio + attribution (person_web enricher, wave-G). */
export interface PersonWeb {
  bio: string
  /** One-line "who is this" descriptor (e.g. "American financier and politician") — a glanceable
   *  subtitle under the name. Null when the source carried none. */
  description?: string | null
  source: string
  source_url?: string | null
  /** OUR served photo route (/api/app/persons/{id}/photo) when a photo is self-hosted; else null. */
  image_url?: string | null
  license?: string | null
  /** The PHOTO's own license + credit (distinct from the bio text's `license`). */
  image_license?: string | null
  image_artist?: string | null
}

/** Person profile card (GET /api/app/persons/{id} — AppPersonCard). KG co-occurrence. */
export interface PersonCard {
  id: string
  label: string
  /** Headline speaker role across the corpus (host / guest / mentioned); null when unknown. */
  role?: string | null
  /** Per-show role breakdown — hosts of one show can be guests on another. Hosted shows first. */
  shows?: PersonShow[]
  episode_count: number
  episodes: EpisodeSummary[]
  related_people: Entity[]
  related_topics: Topic[]
  /** Optional external bio + attribution; absent unless the person_web enricher matched. */
  web?: PersonWeb | null
}

/** Organization card (GET /api/app/organizations/{id} — AppOrgCard; #2031). KG-grounded over
 *  MENTIONS_ORG. Leaner than the person card — no web bio/photo; a name, where it's mentioned,
 *  and who/what it co-occurs with. */
/** External web enrichment for an org (org_web enricher, #2035): description + logo + facts +
 *  attribution. Absent unless the enricher ran and matched. Logos are often absent (non-free). */
export interface OrgWeb {
  description?: string | null
  summary?: string | null
  source: string
  source_url?: string | null
  logo_url?: string | null
  logo_license?: string | null
  founded?: string | null
  industry?: string | null
  website?: string | null
}

export interface OrgCard {
  id: string
  label: string
  episode_count: number
  episodes: EpisodeSummary[]
  related_people: Entity[]
  related_orgs: Entity[]
  related_topics: Topic[]
  web?: OrgWeb | null
}

/** Topic card (GET /api/app/topics/{id} — AppTopicCard). Episodes-about + cluster siblings. */
export interface TopicCard {
  id: string
  label: string
  cluster_id: string | null
  cluster_label: string | null
  cluster_size: number
  sibling_topics: Topic[]
  theme_cluster_id?: string | null
  theme_cluster_label?: string | null
  theme_cluster_size?: number
  theme_sibling_topics?: Topic[]
  episode_count: number
  episodes: EpisodeSummary[]
  related_people: Entity[]
}

/** One speaker's take on a topic — their grounded insights (#1146). */
export interface TopicPerspective {
  person_id: string
  person_name: string
  /** Served hosted-photo route when the web enricher has a photo for this speaker, else null. */
  image_url?: string | null
  insight_count: number
  episode_count: number
  insights: Insight[]
}

/** Multi-perspective synthesis (GET /api/app/topics/{id}/perspectives → AppTopicPerspectivesResponse). */
export interface TopicPerspectivesResponse {
  topic_id: string
  topic_label: string
  perspective_count: number
  perspectives: TopicPerspective[]
}

/** One ISO-week bucket of a topic's conversation (volume + sentiment mix) — ADR-108. */
export interface TopicConversationArcWeek {
  week: string
  volume: number
  negative: number
  neutral: number
  positive: number
  avg_compound: number
}

export interface TopicConversationArcResponse {
  topic_id: string
  weeks: TopicConversationArcWeek[]
}

/** Corpus-scope enrichment signals (GET /api/app/corpus/enrichment → `signals`).
 *  Enricher id → its envelope `data`. Every field is optional/best-effort: an
 *  enricher that didn't run just doesn't appear. Only the fields the entity card
 *  consumes are typed. */
export interface CorpusEnrichmentSignals {
  /** #1927 — per-EPISODE since the per-Person metric was found to be a constant (1.0 for all
   *  689 people: an ungrounded insight has no supporting quote, therefore no speaker, therefore
   *  no person to attribute it to). No consumer surface renders this — per-episode QA is an
   *  operator concern — but the raw envelope is still reachable via `/api/app/corpus/enrichment`,
   *  so the shape is declared honestly rather than left as a stale `persons` array. */
  grounding_rate?: {
    episodes?: Array<{
      episode_id: string
      total_insights: number
      grounded_insights: number
      rate: number
    }>
    corpus_rate?: number
  }
  guest_coappearance?: {
    pairs?: Array<{
      person_a_id: string
      person_b_id: string
      person_a_name?: string
      person_b_name?: string
      episode_count: number
    }>
  }
  /** ADR-108 cross-person corroboration on a topic (embedding cosine + low NLI contradiction). */
  topic_consensus?: {
    consensus?: Array<{
      topic_id: string
      person_a_id: string
      person_b_id: string
      person_a_name?: string
      person_b_name?: string
      insight_a_text?: string
      insight_b_text?: string
    }>
  }
  temporal_velocity?: {
    /** Ordered YYYY-MM axis the monthly_counts are keyed on. */
    window_months?: string[]
    topics?: Array<{
      topic_id: string
      topic_label?: string
      velocity_last_over_6mo?: number
      total?: number
      monthly_counts?: Record<string, number>
    }>
  }
  topic_similarity?: {
    topics?: Array<{
      topic_id: string
      top_k?: Array<{ topic_id: string; topic_label?: string; similarity: number }>
    }>
  }
  topic_cooccurrence_corpus?: {
    pairs?: Array<{
      topic_a_id: string
      topic_b_id: string
      topic_a_label?: string
      topic_b_label?: string
      episode_count: number
      lift?: number
    }>
  }
  /** Co-occurrence theme clusters ("storylines"); used to mark which topics belong to a theme. */
  topic_theme_clusters?: {
    clusters?: Array<{
      graph_compound_parent_id?: string
      canonical_label?: string
      members?: Array<{ topic_id: string }>
    }>
  }
}

/** Top-N rising topics for the Home trending rail (GET /api/app/corpus/trending-topics).
 *  A server-side projection of `temporal_velocity` — already filtered (rising), sorted (velocity
 *  by trend_score desc) and trimmed, so the client renders it directly instead of downloading the whole corpus.
 *  `has_velocity_data` separates "no enricher → render nothing" from "ran, nothing rising → quiet". */
export interface TrendingTopicsResponse {
  has_velocity_data: boolean
  window_months: string[]
  topics: Array<{
    topic_id: string
    topic_label?: string | null
    velocity_last_over_6mo: number
    /** #1931 — volume-with-recency. This is what the server ORDERS by; velocity is a ratio.
     *  NULL on an artifact written before #1931 — the server deliberately does not coerce a
     *  missing score to 0.0, because 0.0 is a real value and the fallback below needs to
     *  tell the two apart. */
    trend_score?: number | null
    total: number
    monthly_counts: Record<string, number>
  }>
  theme_clusters: Array<{
    graph_compound_parent_id?: string | null
    canonical_label?: string | null
    members: Array<{ topic_id: string }>
  }>
}

/** Per-episode enrichment signals (GET /api/app/episodes/{slug}/enrichment → `signals`).
 *  Only the fields the player consumes are typed. */
export interface EpisodeEnrichmentSignals {
  insight_density?: {
    counts?: { early: number; mid: number; late: number; unknown?: number }
    total_insights?: number
    duration_seconds?: number
    has_timing?: boolean
  }
}

/** One grounded search hit (loosely typed — metadata/lifted vary by tier). */
export interface SearchHit {
  doc_id: string
  score: number
  text: string
  metadata: Record<string, unknown>
  source_tier: string
  supporting_quotes?: Record<string, unknown>[] | null
  lifted?: Record<string, unknown> | null
}

export interface SearchResponse {
  query: string
  results: SearchHit[]
  error: string | null
}

/** One day bucket of a listening sparkline (UXS-014). */
export interface StatPoint {
  date: string
  count: number
}

/** The signed-in user's own listening analytics (GET /api/app/me/stats). */
export interface UserStats {
  episodes: number
  shows: number
  listening_seconds: number
  active_days: number
  day_streak: number
  daily: StatPoint[]
}

/** Cross-user reach for one episode (GET /api/app/episodes/{slug}/stats). */
export interface EpisodeStats {
  slug: string
  /**
   * NULL below the k-anonymity floor (#1923). This endpoint is public, and with a handful of users
   * an exact count re-identifies. Null means "not enough people to say", never "nobody" — render
   * nothing rather than a zero.
   */
  listeners: number | null
  opens: number | null
  insights: number
  daily: StatPoint[]
}

/** A topic or person that recurred across a recap window's episodes (#1914). */
export interface RecapTheme {
  token: string
  label: string
  episodes: number
  /**
   * Change against the previous window of the same length. The point of the exposure log: the
   * same three labels every week say nothing, "up two" says what changed. Zero when there is
   * nothing to compare against (history predating the log).
   */
  delta: number
  /** Absent from the previous window entirely — reads as "new", not as "+3". */
  is_new: boolean
}

/** An episode ranked by engagement strength (RFC-114), not by play count. */
export interface RecapStrongEpisode {
  slug: string
  strength: number
}

/** A verbatim line the listener saved — the one part of a recap that is an artifact. */
export interface RecapLine {
  quote_text: string
  episode_slug: string | null
  /** Anchor, so the UI can open the episode AT the moment rather than at the start. */
  start_ms: number | null
  created_at: number
}

/**
 * A listening recap for one window (GET /api/app/me/recap).
 *
 * `days_recorded` / `days_in_window` are NOT decoration: listening time has only been recorded
 * since #1914 Phase 0, so a "year" asked for today covers days. Rendering the total without
 * saying so states something untrue.
 */
export interface RecapResponse {
  window: RecapWindow
  from_day: string
  to_day: string
  listening_seconds: number
  by_day: Record<string, number>
  episodes_started: number
  distinct_episodes: number
  top_episodes: { slug: string; starts: number }[]
  episodes_finished: number
  topics: RecapTheme[]
  people: RecapTheme[]
  top_by_strength: RecapStrongEpisode[]
  best_line: RecapLine | null
  days_recorded: number
  days_in_window: number
  coverage_from: string | null
  first_listened_at: number | null
}

export type RecapWindow = "week" | "month" | "year" | "ytd"
