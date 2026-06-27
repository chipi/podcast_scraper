import type { EpisodeDetail, EpisodeSummary } from '../services/types'

/** Anything carrying the episode artwork fallback chain (EpisodeSummary or EpisodeDetail). */
type WithEpisodeArt = Pick<EpisodeSummary, 'artwork_url' | 'episode_image_url' | 'feed_image_url'>

/**
 * Preferred episode artwork: our locally-stored copy, then the episode image, then the feed image.
 * ONE place for the fallback order so it can't drift across cards, the player and panels.
 */
export function episodeArtwork(e: WithEpisodeArt): string | null {
  return e.artwork_url || e.episode_image_url || e.feed_image_url
}

/** Preferred show artwork: our stored copy, then the remote feed image. */
export function showArtwork(p: { artwork_url: string | null; image_url: string | null }): string | null {
  return p.artwork_url || p.image_url
}

/** A short, clean one-line lede from a prose summary: the first sentence, else a capped excerpt. */
function ledeFrom(text: string | null, maxLen = 160): string | null {
  if (!text) return null
  const match = text.match(/^[\s\S]*?[.!?](?=\s|$)/)
  const first = (match ? match[0] : text).trim()
  return first.length <= maxLen ? first : `${text.slice(0, maxLen).trimEnd()}…`
}

/**
 * Adapt a hydrated {@link EpisodeDetail} to the {@link EpisodeSummary} shape the shared
 * `<EpisodeCard>` consumes, so Queue / Recent / Saved all showcase an episode identically
 * (UXS-014 — one card, every surface). Detail has no catalog-style short lede, so we derive a
 * one-line lede from the prose summary (not the full text, which the card's lede slot would clamp)
 * and leave topics empty.
 */
export function summaryFromDetail(d: EpisodeDetail): EpisodeSummary {
  return {
    slug: d.slug,
    title: d.title,
    feed_id: d.feed_id,
    podcast_title: d.podcast_title,
    publish_date: d.publish_date,
    duration_seconds: d.duration_seconds,
    episode_image_url: d.episode_image_url,
    feed_image_url: d.feed_image_url,
    artwork_url: d.artwork_url,
    status: 'ready',
    summary_preview: ledeFrom(d.summary_text),
    summary_text: d.summary_text,
    summary_bullets: d.summary_bullets,
    topics: [],
    has_transcript: d.has_transcript,
    has_summary: d.has_summary,
    has_gi: d.has_gi,
    has_kg: d.has_kg,
    has_bridge: d.has_bridge,
  }
}
