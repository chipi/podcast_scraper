/**
 * Group episode-cards by publish year for the mobile-friendly year-headers
 * variant of the timeline chart concept (#1261-7). Answers the listener
 * question "when was this discussed?" without shipping a chart that doesn't
 * read on a 375px viewport.
 *
 * Year source: ``publish_date`` (ISO string, e.g. ``"2024-03-01"``). Rows
 * without a resolvable year fall into an ``unknown`` bucket that renders at
 * the bottom (still visible — never silently dropped).
 *
 * Sort order:
 *   1. Known years, descending (newest first — matches every catalog list).
 *   2. Then the ``unknown`` bucket, if any.
 * Groups keep the incoming order inside each year (already score-sorted).
 */

export type YearKey = number | 'unknown'

export interface YearSection<T> {
  year: YearKey
  groups: T[]
}

function parseYear(dateStr: string | null | undefined): number | null {
  if (typeof dateStr !== 'string' || !dateStr.trim()) return null
  const m = dateStr.match(/^(\d{4})/)
  if (!m) return null
  const y = Number(m[1])
  return Number.isFinite(y) ? y : null
}

/**
 * Bucket an ordered list of episode groups by publish year.
 *
 * ``getDate`` reads the ISO date off each group; kept as a parameter so
 * this helper isn't coupled to any specific group shape.
 */
export function groupEpisodesByYear<T>(
  groups: readonly T[],
  getDate: (g: T) => string | null | undefined,
): YearSection<T>[] {
  const byYear = new Map<YearKey, T[]>()
  const order: YearKey[] = []
  for (const g of groups) {
    const year = parseYear(getDate(g)) ?? ('unknown' as const)
    let bucket = byYear.get(year)
    if (!bucket) {
      bucket = []
      byYear.set(year, bucket)
      order.push(year)
    }
    bucket.push(g)
  }
  const known = order.filter((k): k is number => k !== 'unknown').sort((a, b) => b - a)
  const hasUnknown = order.includes('unknown')
  const out: YearSection<T>[] = []
  for (const y of known) out.push({ year: y, groups: byYear.get(y)! })
  if (hasUnknown) out.push({ year: 'unknown', groups: byYear.get('unknown')! })
  return out
}
