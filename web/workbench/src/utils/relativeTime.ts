import { localZoneShortName, parseStoredInstant } from './formatting'

/**
 * Compact bidirectional relative time for the scheduled-jobs table (#709) —
 * e.g. "in 3h 12m" for a future `next_run_at`, "5m ago" for the past. Returns
 * "—" for null/blank/unparseable input so the cell stays tidy.
 */
export function formatRelativeShort(iso: string | null | undefined, nowMs: number): string {
  const raw = iso?.trim()
  if (!raw) return '—'
  const instant = parseStoredInstant(raw)
  if (!instant) return '—'
  const t = instant.getTime()

  const deltaMs = t - nowMs
  const future = deltaMs >= 0
  const sec = Math.floor(Math.abs(deltaMs) / 1000)
  if (sec < 30) return 'now'

  const min = Math.floor(sec / 60)
  const h = Math.floor(min / 60)
  const d = Math.floor(h / 24)

  let body: string
  if (min < 60) {
    body = `${min}m`
  } else if (h < 24) {
    const remMin = min % 60
    body = remMin ? `${h}h ${remMin}m` : `${h}h`
  } else if (d < 7) {
    const remH = h % 24
    body = remH ? `${d}d ${remH}h` : `${d}d`
  } else {
    body = `${d}d`
  }
  return future ? `in ${body}` : `${body} ago`
}

/** Absolute label in the viewer's local zone for the hover title (e.g. "2026-06-18 22:00 EDT"). */
export function formatAbsoluteLocal(iso: string | null | undefined): string {
  const raw = iso?.trim()
  if (!raw) return ''
  const d = parseStoredInstant(raw)
  if (!d) return raw
  const p = (n: number) => String(n).padStart(2, '0')
  const day = `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
  return `${day} ${p(d.getHours())}:${p(d.getMinutes())} ${localZoneShortName(d)}`
}
