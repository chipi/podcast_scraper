/**
 * Feed-row display helpers shared by Library, the graph Feed chip,
 * Search Feed chip, and any other surface that renders a row from
 * ``GET /api/corpus/feeds``.
 *
 * Extracted from ``components/library/LibraryView.vue`` (#669 / #658)
 * so the chip-bar refactor and the legacy Library row can reuse the
 * same hover/title/aria semantics.
 */
import type { CorpusFeedItem } from '../api/corpusLibraryApi'

/** Search-input visibility threshold for the feed picker. */
export const CORPUS_FEED_FILTER_SEARCH_THRESHOLD = 15

/** Visible label inside the row: prefer display title, fall back to feed_id. */
export function feedRowVisibleLabel(f: CorpusFeedItem): string {
  if (f.feed_id === '') {
    return '(No feed id)'
  }
  const t = f.display_title?.trim()
  if (t) {
    return t
  }
  return f.feed_id.trim()
}

/** Hover/native tooltip: title, id, RSS URL, description when present. */
export function feedRowTitleAttr(f: CorpusFeedItem): string {
  const parts: string[] = []
  if (f.feed_id === '') {
    parts.push('(No feed id)')
  } else {
    const t = f.display_title?.trim()
    const id = f.feed_id.trim()
    if (t && id) {
      parts.push(`${t} · ${id}`)
    } else {
      parts.push(id || t || '')
    }
  }
  if (f.rss_url?.trim()) {
    parts.push(`RSS: ${f.rss_url.trim()}`)
  }
  if (f.description?.trim()) {
    parts.push(f.description.trim())
  }
  return parts.filter(Boolean).join('\n')
}

/** Accessible name for screen readers: title, feed id, episode count. */
export function feedRowAccessibleName(f: CorpusFeedItem): string {
  if (f.feed_id === '') {
    return `(No feed id), ${f.episode_count} episodes`
  }
  const t = f.display_title?.trim()
  const id = f.feed_id.trim()
  if (t) {
    return `${t}, feed id ${id}, ${f.episode_count} episodes`
  }
  return `${id}, ${f.episode_count} episodes`
}

/** Filter the feed list by case-insensitive substring of display title or feed_id. */
export function filterFeedsByQuery(
  feeds: ReadonlyArray<CorpusFeedItem>,
  q: string,
): CorpusFeedItem[] {
  const needle = q.trim().toLowerCase()
  if (!needle) return feeds.slice()
  return feeds.filter((f) => {
    const label = feedRowVisibleLabel(f).toLowerCase()
    const id = f.feed_id.toLowerCase()
    return label.includes(needle) || id.includes(needle)
  })
}
