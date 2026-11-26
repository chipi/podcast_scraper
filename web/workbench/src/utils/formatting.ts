/** Escape text for safe HTML interpolation. */
export function escapeHtml(s: string): string {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

export function truncate(s: string, max: number): string {
  const str = String(s)
  if (str.length <= max) {
    return str
  }
  return `${str.slice(0, max - 1)}…`
}

/**
 * Extract a short phrase from a long sentence.  Cuts at the first natural
 * break (comma, semicolon, em-dash, period) that falls within `max` chars,
 * otherwise hard-truncates with an ellipsis.
 */
export function shortPhrase(s: string, max = 40): string {
  const str = String(s).trim()
  if (str.length <= max) return str
  const breakRe = /[,;.—–]|\s-\s/
  const m = breakRe.exec(str.slice(0, max))
  if (m && m.index >= 8) {
    return str.slice(0, m.index).trimEnd()
  }
  const spaceIdx = str.lastIndexOf(' ', max - 1)
  if (spaceIdx >= 8) {
    return `${str.slice(0, spaceIdx)}…`
  }
  return `${str.slice(0, max - 1)}…`
}

export function humanizeSlug(slug: string): string {
  return String(slug)
    .split('-')
    .filter(Boolean)
    .map((w) => (w.length ? w.charAt(0).toUpperCase() + w.slice(1) : ''))
    .join(' ')
}

const BARE_DATE_RE = /^(\d{4})-(\d{2})-(\d{2})$/
const NAIVE_ISO_RE = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?$/
const ZONED_ISO_RE = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}(?::?\d{2})?)$/i

function isRealBareDate(s: string): boolean {
  const m = BARE_DATE_RE.exec(s)
  if (!m) return false
  const d = new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])))
  return d.getUTCMonth() === Number(m[2]) - 1 && d.getUTCDate() === Number(m[3])
}

/**
 * Read a stored timestamp as an instant. Everything we store is UTC, so an ISO datetime with no
 * offset is UTC (`new Date()` would read it as viewer-local). Z / offset values keep their zone.
 * A bare `YYYY-MM-DD` is a day, not an instant, so it (and blank or unparseable input) is null.
 * Numbers are epoch seconds, or milliseconds when too large to be seconds.
 */
export function parseStoredInstant(raw: string | number | null | undefined): Date | null {
  if (typeof raw === 'number') {
    if (!Number.isFinite(raw)) return null
    const d = new Date(Math.abs(raw) < 1e11 ? raw * 1000 : raw)
    return Number.isNaN(d.getTime()) ? null : d
  }
  const s = String(raw ?? '').trim()
  if (!s) return null
  let iso: string
  if (NAIVE_ISO_RE.test(s)) {
    iso = `${s.replace(' ', 'T')}Z`
  } else if (ZONED_ISO_RE.test(s)) {
    iso = s.replace(' ', 'T')
  } else {
    return null
  }
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? null : d
}

/**
 * The viewer-local `YYYY-MM-DD` of a stored instant. A bare date is returned unchanged (it is a day,
 * never shifted). Null for blank or unparseable input.
 */
export function localDay(raw: string | number | null | undefined): string | null {
  if (typeof raw === 'string') {
    const s = raw.trim()
    if (isRealBareDate(s)) return s
  }
  const d = parseStoredInstant(raw)
  if (!d) return null
  const mm = String(d.getMonth() + 1).padStart(2, '0')
  const dd = String(d.getDate()).padStart(2, '0')
  return `${String(d.getFullYear()).padStart(4, '0')}-${mm}-${dd}`
}

/** Short name of the viewer's zone at `d` (e.g. "EDT", "GMT+2"). */
export function localZoneShortName(d: Date): string {
  const part = new Intl.DateTimeFormat('en-US', { timeZoneName: 'short' })
    .formatToParts(d)
    .find((p) => p.type === 'timeZoneName')
  return part?.value ?? ''
}

/**
 * Calendar-style date for episode / timeline rails (no time-of-day), in the viewer's local zone.
 * A bare `YYYY-MM-DD` is shown as that day (formatted in UTC so it never shifts). Unparseable
 * input is truncated when very long, else returned as-is.
 */
export function formatCalendarDateForDisplay(raw: string | null | undefined): string {
  const s = String(raw ?? '').trim()
  if (!s) {
    return ''
  }
  const opts: Intl.DateTimeFormatOptions = { month: 'short', day: 'numeric', year: 'numeric' }
  if (isRealBareDate(s)) {
    return new Intl.DateTimeFormat('en-US', { ...opts, timeZone: 'UTC' }).format(new Date(`${s}T00:00:00Z`))
  }
  const d = parseStoredInstant(s)
  if (!d) {
    return s.length > 40 ? `${s.slice(0, 39)}…` : s
  }
  return new Intl.DateTimeFormat('en-US', opts).format(d)
}

/**
 * Pretty-print a stored instant in the viewer's local zone with the short zone name
 * (no sub-second noise). A bare date is shown as a calendar date; unparseable input is returned unchanged.
 */
export function formatDateTimeForDisplay(raw: string | null | undefined): string {
  const s = String(raw ?? '').trim()
  if (!s) {
    return ''
  }
  if (isRealBareDate(s)) {
    return formatCalendarDateForDisplay(s)
  }
  const d = parseStoredInstant(s)
  if (!d) {
    return s
  }
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
    timeZoneName: 'short',
  }).format(d)
}

/** Compact on-disk size for index bytes. */
export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return '—'
  if (n < 1024) return `${Math.round(n)} B`
  const kb = n / 1024
  if (kb < 1024) return `${kb.toFixed(1)} KB`
  return `${(kb / 1024).toFixed(1)} MB`
}
