/** Shared digest card copy (Digest tab recent rows; Library episode list uses `libraryEpisodeSummaryLine`). */

import { normalizeFeedIdForViewer } from './feedId'

export type DigestRowSummaryFields = {
  summary_preview?: string | null
  summary_title?: string | null
  summary_bullets_preview?: string[]
}

export type DigestRowFeedFields = {
  feed_display_title?: string | null
  feed_id?: string
}

export function digestRowSummaryPreview(row: DigestRowSummaryFields): string {
  const p = row.summary_preview?.trim()
  if (p) {
    return p
  }
  const t = row.summary_title?.trim()
  const bullets = (row.summary_bullets_preview ?? []).map((b) => b.trim()).filter(Boolean)
  if (t && bullets.length) {
    return `${t} — ${bullets.join(' · ')}`
  }
  if (t) {
    return t
  }
  if (bullets.length) {
    return bullets.join(' · ')
  }
  return ''
}

/** Library list row: same recap as digest, with ``topics`` as bullet fallback (older APIs). */
export type LibraryEpisodeSummaryInput = DigestRowSummaryFields & {
  topics?: string[]
}

export function libraryEpisodeSummaryLine(e: LibraryEpisodeSummaryInput): string {
  const fromPreview = e.summary_bullets_preview?.filter((b) => String(b).trim())
  const bullets =
    fromPreview?.length ? fromPreview : (e.topics ?? []).filter((t) => String(t).trim())
  return digestRowSummaryPreview({
    summary_preview: e.summary_preview,
    summary_title: e.summary_title,
    summary_bullets_preview: bullets,
  })
}

export function digestRowFeedLabel(row: DigestRowFeedFields): string {
  const t = row.feed_display_title?.trim()
  if (t) {
    return t
  }
  const id = row.feed_id?.trim()
  if (id) {
    return id
  }
  return 'Unknown feed'
}

/**
 * Prefer ``GET /api/corpus/feeds`` ``display_title`` for ``feed_id`` (Library sidebar source),
 * then row-level ``feed_display_title`` / ``feed_id`` — aligns Digest with Library feed names.
 */
/** Tunable: strong semantic similarity tier (Digest topic-band hits). */
export const DIGEST_SIMILARITY_STRONG_MIN = 0.85
/** Tunable: good tier floor (inclusive). */
export const DIGEST_SIMILARITY_GOOD_MIN = 0.7

export type DigestTopicSimilarityDisplay = {
  label: string
  /** Tailwind text colour classes for the label (no border / pill). */
  labelClass: string
  /** Native ``title`` for the label element (raw numeric score). */
  rawTitle: string
}

/**
 * Semantic strength label for a topic-band vector score (Digest).
 * Raw value is surfaced via ``rawTitle`` on the label, not in the visible string.
 */
export function digestTopicHitSimilarityDisplay(score: number): DigestTopicSimilarityDisplay {
  const rawTitle = `Similarity: ${score.toFixed(3)}`
  if (score >= DIGEST_SIMILARITY_STRONG_MIN) {
    return { label: 'Strong match', labelClass: 'text-gi', rawTitle }
  }
  if (score >= DIGEST_SIMILARITY_GOOD_MIN) {
    return { label: 'Good match', labelClass: 'text-muted', rawTitle }
  }
  return { label: 'Weak match', labelClass: 'text-disabled', rawTitle }
}

export function digestRowFeedLabelWithCatalog(
  row: DigestRowFeedFields,
  titleByFeedId: Readonly<Record<string, string>>,
): string {
  const fid = normalizeFeedIdForViewer(row.feed_id)
  if (fid) {
    const fromCatalog = titleByFeedId[fid]?.trim()
    if (fromCatalog) {
      return fromCatalog
    }
  }
  return digestRowFeedLabel(row)
}
