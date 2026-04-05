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

export function humanizeSlug(slug: string): string {
  return String(slug)
    .split('-')
    .filter(Boolean)
    .map((w) => (w.length ? w.charAt(0).toUpperCase() + w.slice(1) : ''))
    .join(' ')
}

/** Compact on-disk size for index bytes. */
export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return '—'
  if (n < 1024) return `${Math.round(n)} B`
  const kb = n / 1024
  if (kb < 1024) return `${kb.toFixed(1)} KB`
  return `${(kb / 1024).toFixed(1)} MB`
}
