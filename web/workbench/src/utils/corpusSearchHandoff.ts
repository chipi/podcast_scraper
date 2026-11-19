/**
 * Build semantic-search query text for Library → Search handoff.
 * Uses the same **field order** as ``build_similarity_query`` in ``corpus_similar.py``
 * (summary title → bullets → episode title) and does **not** use prose ``summary_text``.
 *
 * Metadata often stores a full recap in ``summary.title`` or a single huge bullet; we **clip**
 * each segment and cap the total so the Search box stays short (server Similar episodes may
 * still use longer strings — different UX budget).
 */

/** Final query length cap (characters). */
const HANDOFF_TOTAL_MAX = 480
/** Max characters from ``summary.title`` (may contain paragraph-style text). */
const HANDOFF_TITLE_MAX = 140
/** Max characters per bullet. */
const HANDOFF_BULLET_MAX = 220
/** Max characters from episode title when used as fallback. */
const HANDOFF_EPISODE_TITLE_MAX = 200
/** At most this many bullets contribute (after clipping). */
const HANDOFF_MAX_BULLETS = 3

function clipSegment(s: string, maxChars: number): string {
  const t = s.trim()
  if (!t) {
    return ''
  }
  if (t.length <= maxChars) {
    return t
  }
  const cut = t.slice(0, maxChars)
  const lastSpace = cut.lastIndexOf(' ')
  if (lastSpace > maxChars * 0.55) {
    return cut.slice(0, lastSpace).trim()
  }
  return cut.trim()
}

function truncateQuery(q: string, maxChars: number): string {
  if (q.length <= maxChars) {
    return q
  }
  const cut = q.slice(0, maxChars)
  const lastSpace = cut.lastIndexOf(' ')
  if (lastSpace > 0) {
    return cut.slice(0, lastSpace).trim()
  }
  return cut.trim()
}

export function buildLibrarySearchHandoffQuery(
  detail: {
    summary_title: string | null
    summary_bullets: string[]
    episode_title: string
  },
  maxChars: number = HANDOFF_TOTAL_MAX,
): string {
  const parts: string[] = []
  const st = detail.summary_title?.trim()
  if (st) {
    const c = clipSegment(st, HANDOFF_TITLE_MAX)
    if (c) {
      parts.push(c)
    }
  }
  let nBullets = 0
  for (const b of detail.summary_bullets ?? []) {
    if (nBullets >= HANDOFF_MAX_BULLETS) {
      break
    }
    const t = b?.trim()
    if (!t) {
      continue
    }
    const c = clipSegment(t, HANDOFF_BULLET_MAX)
    if (c) {
      parts.push(c)
      nBullets += 1
    }
  }
  if (parts.length === 0) {
    const et = detail.episode_title?.trim()
    if (et) {
      const c = clipSegment(et, HANDOFF_EPISODE_TITLE_MAX)
      if (c) {
        parts.push(c)
      }
    }
  }
  const q = parts.join(' ').trim()
  return truncateQuery(q, maxChars)
}
