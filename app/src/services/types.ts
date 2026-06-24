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
  summary_preview: string | null
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

/** A KG topic. */
export interface Topic {
  id: string
  label: string
}

export interface EntitiesResponse {
  episode_slug: string
  persons: Entity[]
  orgs: Entity[]
  topics: Topic[]
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
