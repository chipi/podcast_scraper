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
 * Bucket `.gi.json` artifact mtimes by UTC day; sort ascending by day.
 */
export function bucketGiMtimesByDay(
  items: { kind: string; mtime_utc: string }[],
): DayBucket[] {
  const map = new Map<string, number>()
  for (const it of items) {
    if (it.kind !== 'gi') {
      continue
    }
    const day = utcDayFromMtime(it.mtime_utc)
    if (!day) {
      continue
    }
    map.set(day, (map.get(day) ?? 0) + 1)
  }
  const keys = Array.from(map.keys()).sort()
  return keys.map((day) => ({ day, count: map.get(day) ?? 0 }))
}

/**
 * Bucket GI and KG artifact mtimes by UTC day (one count per file per day).
 * Use for “when did processing write files?” when runs may only emit KG.
 */
export function bucketGiKgMtimesByDay(
  items: { kind: string; mtime_utc: string }[],
): DayBucket[] {
  const map = new Map<string, number>()
  for (const it of items) {
    if (it.kind !== 'gi' && it.kind !== 'kg') {
      continue
    }
    const day = utcDayFromMtime(it.mtime_utc)
    if (!day) {
      continue
    }
    map.set(day, (map.get(day) ?? 0) + 1)
  }
  const keys = Array.from(map.keys()).sort()
  return keys.map((day) => ({ day, count: map.get(day) ?? 0 }))
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

/**
 * Per UTC day counts of GI vs KG artifact writes, then cumulative totals (for growth curves).
 */
/** Per UTC day counts of new GI vs KG writes (not cumulative). */
export type DayGiKgBucket = { day: string; gi: number; kg: number }

/**
 * Last ``windowDays`` UTC calendar days ending at ``endUtc`` (inclusive of that UTC date),
 * with counts of artifacts whose ``mtime_utc`` falls on each day.
 */
export function dailyGiKgNewCountsLastDays(
  items: { kind: string; mtime_utc: string }[],
  windowDays = 30,
  endUtc: Date = new Date(),
): DayGiKgBucket[] {
  const giDay = new Map<string, number>()
  const kgDay = new Map<string, number>()
  for (const it of items) {
    if (it.kind !== 'gi' && it.kind !== 'kg') {
      continue
    }
    const day = utcDayFromMtime(it.mtime_utc)
    if (!day) {
      continue
    }
    const map = it.kind === 'gi' ? giDay : kgDay
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
    gi: giDay.get(day) ?? 0,
    kg: kgDay.get(day) ?? 0,
  }))
}

export function cumulativeGiKgByDay(
  items: { kind: string; mtime_utc: string }[],
): { day: string; gi: number; kg: number }[] {
  const giDay = new Map<string, number>()
  const kgDay = new Map<string, number>()
  for (const it of items) {
    if (it.kind !== 'gi' && it.kind !== 'kg') {
      continue
    }
    const day = utcDayFromMtime(it.mtime_utc)
    if (!day) {
      continue
    }
    const map = it.kind === 'gi' ? giDay : kgDay
    map.set(day, (map.get(day) ?? 0) + 1)
  }
  const days = Array.from(new Set([...giDay.keys(), ...kgDay.keys()])).sort()
  let giC = 0
  let kgC = 0
  return days.map((day) => {
    giC += giDay.get(day) ?? 0
    kgC += kgDay.get(day) ?? 0
    return { day, gi: giC, kg: kgC }
  })
}
