/** Max GI+KG artifact rows to bucket client-side before showing a cap warning. */
export const MAX_ARTIFACT_ROWS_FOR_CLIENT_TIMELINE = 8000

export type DayBucket = { day: string; count: number }

/**
 * Parse `mtime_utc` ISO strings to UTC calendar day `YYYY-MM-DD`.
 * Returns empty string when unparseable.
 */
export function utcDayFromMtime(iso: string): string {
  const s = iso.trim()
  if (s.length >= 10 && s[4] === '-' && s[7] === '-') {
    return s.slice(0, 10)
  }
  try {
    const d = new Date(s)
    if (Number.isNaN(d.getTime())) {
      return ''
    }
    return d.toISOString().slice(0, 10)
  } catch {
    return ''
  }
}

/**
 * Sorted `YYYY-MM` keys with episode counts (from server histogram).
 */
export function sortedMonthHistogram(
  hist: Record<string, number>,
): { label: string; count: number }[] {
  const keys = Object.keys(hist).sort()
  return keys.map((label) => ({ label, count: hist[label] ?? 0 }))
}

/** Per UTC day counts of new artifact writes, per kind (not cumulative). */
export type DayKindBucket = { day: string; counts: Record<string, number> }

/**
 * Last ``windowDays`` UTC calendar days ending at ``endUtc`` (inclusive of that UTC date), with,
 * for each of ``kinds``, the count of artifacts of that kind whose ``mtime_utc`` falls on the day.
 */
export function dailyNewCountsLastDays(
  items: { kind: string; mtime_utc: string }[],
  kinds: string[],
  windowDays = 30,
  endUtc: Date = new Date(),
): DayKindBucket[] {
  const byKind = new Map<string, Map<string, number>>(kinds.map((k) => [k, new Map()]))
  for (const it of items) {
    const map = byKind.get(it.kind)
    if (!map) {
      continue
    }
    const day = utcDayFromMtime(it.mtime_utc)
    if (!day) {
      continue
    }
    map.set(day, (map.get(day) ?? 0) + 1)
  }
  const y = endUtc.getUTCFullYear()
  const m = endUtc.getUTCMonth()
  const d = endUtc.getUTCDate()
  const days: string[] = []
  for (let i = windowDays - 1; i >= 0; i -= 1) {
    const dt = new Date(Date.UTC(y, m, d - i))
    days.push(dt.toISOString().slice(0, 10))
  }
  return days.map((day) => ({
    day,
    counts: Object.fromEntries(kinds.map((k) => [k, byKind.get(k)!.get(day) ?? 0])),
  }))
}
