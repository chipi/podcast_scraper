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

/**
 * Pretty-print an ISO-8601 instant in UTC for the digest window header (no sub-second noise).
 * Unparseable input is returned unchanged.
 */
export function formatUtcDateTimeForDisplay(iso: string): string {
  const raw = String(iso ?? '').trim()
  if (!raw) {
    return ''
  }
  const d = new Date(raw)
  if (Number.isNaN(d.getTime())) {
    return raw
  }
  return (
    new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
      timeZone: 'UTC',
    }).format(d) + ' UTC'
  )
}

/** Compact on-disk size for index bytes. */
export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return '—'
  if (n < 1024) return `${Math.round(n)} B`
  const kb = n / 1024
  if (kb < 1024) return `${kb.toFixed(1)} KB`
  return `${(kb / 1024).toFixed(1)} MB`
}
