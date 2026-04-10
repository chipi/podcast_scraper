/**
 * Build semantic-search query text for Library → Search handoff.
 * Prefers full ``summary_text`` when present; else mirrors ``build_similarity_query``
 * in ``corpus_similar.py`` (summary title + bullets, fallback episode title).
 */
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
    summary_text?: string | null
    summary_title: string | null
    summary_bullets: string[]
    episode_title: string
  },
  maxChars = 6000,
): string {
  const raw = detail.summary_text?.trim()
  if (raw) {
    return truncateQuery(raw, maxChars)
  }
  const parts: string[] = []
  const st = detail.summary_title?.trim()
  if (st) {
    parts.push(st)
  }
  for (const b of detail.summary_bullets) {
    const t = b?.trim()
    if (t) {
      parts.push(t)
    }
  }
  if (parts.length === 0) {
    const et = detail.episode_title?.trim()
    if (et) {
      parts.push(et)
    }
  }
  const q = parts.join(' ').trim()
  return truncateQuery(q, maxChars)
}
