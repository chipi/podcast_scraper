import type { CorpusDigestTopicHit } from '../api/digestApi'
import { formatDurationSeconds } from './formatDuration'

export type TopicHitHoverContext = {
  feedDisplayLabel: string
  showFeedLine: boolean
  /** Same string as list card recap (`digestRowSummaryPreview`). */
  summaryPreview: string
}

/** Compact hover payload for digest topic-band rows. */
export type DigestTopicHitHoverPanel = {
  /** Raw publish day for ``Published:`` label (YYYY-MM-DD from catalog). */
  publishDateValue: string | null
  /** ``E#`` and duration joined with middots when present (no ``Published`` prefix). */
  timingExtras: string | null
  similarityScore: string | null
  feedLine: string | null
  summaryPreview: string | null
  aboutFeed: string | null
  rssLine: string | null
}

export function topicHitFirstRowHasContent(p: DigestTopicHitHoverPanel): boolean {
  return Boolean(p.publishDateValue || p.timingExtras || p.similarityScore)
}

export function topicHitHoverPanelIsEmpty(p: DigestTopicHitHoverPanel): boolean {
  return !(
    topicHitFirstRowHasContent(p) ||
    p.feedLine ||
    p.summaryPreview ||
    p.aboutFeed ||
    p.rssLine
  )
}

/**
 * Structured metadata for topic-band hit hover (Digest top section).
 * First UI row pairs timing (``Published:`` + date + optional ``E#`` / duration) with score on the right.
 */
export function buildTopicHitHoverPanel(
  h: CorpusDigestTopicHit,
  ctx: TopicHitHoverContext,
): DigestTopicHitHoverPanel {
  const publishDateValue = h.publish_date?.trim() || null

  const extras: string[] = []
  if (h.episode_number != null) {
    extras.push(`E${h.episode_number}`)
  }
  const dur = formatDurationSeconds(h.duration_seconds)
  if (dur) {
    extras.push(dur)
  }
  const timingExtras = extras.length ? extras.join(' · ') : null

  const similarityScore = h.score != null ? h.score.toFixed(3) : null

  const feedLine =
    ctx.showFeedLine && ctx.feedDisplayLabel.trim() ? ctx.feedDisplayLabel.trim() : null

  const summaryRaw = ctx.summaryPreview.trim()
  const summaryPreview = summaryRaw ? summaryRaw : null

  const desc = h.feed_description?.trim()
  const aboutFeed = desc ? (desc.length > 220 ? `${desc.slice(0, 217)}…` : desc) : null

  const rssLine = h.feed_rss_url?.trim() || null

  return {
    publishDateValue,
    timingExtras,
    similarityScore,
    feedLine,
    summaryPreview,
    aboutFeed,
    rssLine,
  }
}
