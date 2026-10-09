import { resolveMediaUrl } from '../services/tier'
import type { EpisodeDetail, EpisodeSummary } from '../services/types'

/** Anything carrying the episode artwork fallback chain (EpisodeSummary or EpisodeDetail). */
type WithEpisodeArt = Pick<EpisodeSummary, 'artwork_url' | 'episode_image_url' | 'feed_image_url'> & {
  artwork_thumb_url?: string | null
}

/**
 * Episode artwork for a CARD, TILE or ROW: our stored thumb, then the episode image, then the feed
 * image. ONE place for the fallback order so it can't drift across surfaces.
 *
 * A detail carries two sizes and this takes the thumb. It used to take the detail's `artwork_url`,
 * which was the original — up to 3000px, ~36 MB decoded — in 116px tiles on Home, Queue, Recent,
 * Saved and boards (measured on a Pixel 8 emulator against prod, 2026-10-08).
 */
export function episodeArtwork(e: WithEpisodeArt): string | null {
  // Absolutised because the API returns these relative, which breaks every image on native
  // (document origin is capacitor://localhost, not the API).
  return resolveMediaUrl(e.artwork_thumb_url || e.artwork_url || e.episode_image_url || e.feed_image_url)
}

/**
 * Episode artwork at PLAYER size (≤1024px): the player hero, the lock screen and the offline copy.
 * Only a detail carries it; anything else falls back to the card chain.
 */
export function episodePlayerArtwork(e: WithEpisodeArt): string | null {
  return resolveMediaUrl(e.artwork_url || e.episode_image_url || e.feed_image_url)
}

/**
 * The thumb of one of OUR artwork URLs, for a small slot fed by a player-size one (the 36px mini
 * player, the accent sampler). Anything else — a remote feed image, an offline file — is returned
 * unchanged: only our own route can be asked for another size.
 */
export function artworkThumb(url: string | null | undefined): string | null {
  if (!url) return null
  return /\/api\/app\/artwork\?/.test(url) ? url.replace(/([?&]size=)(?:medium|large)\b/, '$1thumb') : url
}

/** Preferred show artwork: our stored copy, then the remote feed image. */
export function showArtwork(p: { artwork_url: string | null; image_url: string | null }): string | null {
  return resolveMediaUrl(p.artwork_url || p.image_url)
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
    // A summary's `artwork_url` is card-sized everywhere else, so it gets the thumb here too.
    artwork_url: d.artwork_thumb_url ?? d.artwork_url,
    status: 'ready',
    // `summary_title`, like the server (#2004 item 4). This used to be `ledeFrom(d.summary_text)` —
    // the first-sentence-of-prose shape the server rewrite removed — so Queue and Recent rendered a
    // DIFFERENT shape in the same slot as every other surface, which is the inconsistency that fix
    // was about. `EpisodeDetail` already carries the title, so no extra fetch.
    summary_preview: d.summary_title,
    summary_text: d.summary_text,
    description: d.description ?? null,
    summary_bullets: d.summary_bullets,
    topics: [],
    has_transcript: d.has_transcript,
    has_summary: d.has_summary,
    has_gi: d.has_gi,
    has_kg: d.has_kg,
    has_bridge: d.has_bridge,
    // Carried, not dropped: Queue, Recent, Revisit and Saved all render through this adapter, and
    // without it none of them could show the language badge (V2-C.1).
    language: d.language ?? null,
  }
}
