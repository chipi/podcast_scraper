/**
 * TypeScript mirrors of the `/api/app/*` response shapes (RFC-098/RFC-099). Kept in sync
 * with src/podcast_scraper/server/schemas.py. Only the shapes the client consumes today are
 * declared; extend as surfaces land.
 */

export interface Me {
  user_id: string
  email: string
  name: string
}

export type EpisodeStatus = 'ready' | 'pending'

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
  /** Full summary bullets, surfaced via the card's expand-on-demand insights view. */
  summary_bullets: string[]
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
}

/** A show in the user's library (Home "Your shows"). */
export interface Podcast {
  feed_id: string
  title: string | null
  artwork_url: string | null
  image_url: string | null
  description: string | null
  episode_count: number
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
  }>
  key_people: Array<{ person_id: string; name: string; episode_count: number }>
  recurring_guests: Array<{ person_id: string; name: string; episode_count: number }>
  dominant_themes: Array<{
    theme_id: string
    label: string
    topic_count: number
    anchor_topic_id: string | null
  }>
  trending_topics: Array<{ topic_id: string; label: string; velocity: number; episode_count: number }>
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
  quotes: Quote[]
}

export interface InsightsResponse {
  episode_slug: string
  insights: Insight[]
}

/** A KG person/org entity. */
export interface Entity {
  id: string
  name: string
  kind: 'person' | 'org'
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

/** Saveable kinds for the polymorphic favorites store. */
export type FavoriteKind = 'episode' | 'insight' | 'person' | 'topic'

/** Body for PUT /api/app/favorites — denormalized so the Library renders without re-fetching. */
export interface FavoriteAdd {
  kind: FavoriteKind
  ref: string
  label?: string
  sublabel?: string
  slug?: string
  start_ms?: number
}

/** A saved insight (AppFavoriteInsight) — snapshot, since insights have no global detail route. */
export interface FavoriteInsight {
  ref: string
  text: string
  episode_slug: string | null
  podcast_title: string | null
  start_ms: number | null
}

/** The user's favorites, grouped by kind (GET/PUT/DELETE /api/app/favorites). */
export interface FavoritesResponse {
  episodes: EpisodeSummary[]
  insights: FavoriteInsight[]
}

// --- P2 Capture: highlights + notes (PRD-040 / RFC-098 §7) ---

export type HighlightKind = 'span' | 'moment' | 'insight'

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
}

/** Body for PATCH /api/app/highlights/{id} — edit colour / captured text. */
export interface HighlightUpdate {
  color?: string | null
  quote_text?: string | null
}

export interface HighlightsResponse {
  items: Highlight[]
}

export type NoteTarget = 'highlight' | 'insight' | 'episode'

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
}

/** A resolved person/topic reference (GET /api/app/entities/search — AppEntityRef). */
export interface EntityRef {
  id: string
  kind: 'person' | 'topic'
  label: string
}

/** Entity-in-search resolution (AppEntitySearchResponse) — at most one exact/near-exact match. */
export interface EntitySearchResponse {
  query: string
  entity: EntityRef | null
}

/** Person profile card (GET /api/app/persons/{id} — AppPersonCard). KG co-occurrence. */
export interface PersonCard {
  id: string
  label: string
  episode_count: number
  episodes: EpisodeSummary[]
  related_people: Entity[]
  related_topics: Topic[]
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
  grounding_rate?: {
    persons?: Array<{
      person_id: string
      person_name?: string
      total_insights: number
      grounded_insights: number
      rate: number
    }>
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
  listeners: number
  opens: number
  insights: number
  daily: StatPoint[]
}
