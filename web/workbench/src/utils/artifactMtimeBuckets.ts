import { localDay } from './formatting'

/** Max GI+KG artifact rows to bucket client-side before showing a cap warning. */
export const MAX_ARTIFACT_ROWS_FOR_CLIENT_TIMELINE = 8000

export type DayBucket = { day: string; count: number }

/**
 * Viewer-local calendar day `YYYY-MM-DD` of an `mtime_utc` timestamp (a naive value is UTC).
 * Returns empty string when unparseable.
 */
export function localDayFromMtime(iso: string): string {
  return localDay(iso) ?? ''
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

/** Per viewer-local day counts of new artifact writes, per kind (not cumulative). */
export type DayKindBucket = { day: string; counts: Record<string, number> }

/**
 * Last ``windowDays`` viewer-local calendar days ending at ``end`` (inclusive of its local date), with,
 * for each of ``kinds``, the count of artifacts of that kind whose ``mtime_utc`` falls on the day.
 */
export function dailyNewCountsLastDays(
  items: { kind: string; mtime_utc: string }[],
  kinds: string[],
  windowDays = 30,
  end: Date = new Date(),
): DayKindBucket[] {
  const byKind = new Map<string, Map<string, number>>(kinds.map((k) => [k, new Map()]))
  for (const it of items) {
    const map = byKind.get(it.kind)
    if (!map) {
      continue
    }
    const day = localDayFromMtime(it.mtime_utc)
    if (!day) {
      continue
    }
    map.set(day, (map.get(day) ?? 0) + 1)
  }
  const y = end.getFullYear()
  const m = end.getMonth()
  const d = end.getDate()
  const days: string[] = []
  for (let i = windowDays - 1; i >= 0; i -= 1) {
    const dt = new Date(y, m, d - i)
    days.push(localDay(dt.toISOString()) ?? '')
  }
  return days.map((day) => ({
    day,
    counts: Object.fromEntries(kinds.map((k) => [k, byKind.get(k)!.get(day) ?? 0])),
  }))
}
