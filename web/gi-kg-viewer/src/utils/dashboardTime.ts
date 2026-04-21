/** Relative age for pipeline run timestamps (briefing card). */
export function formatRelativeRunAge(iso: string | null | undefined): string {
  const raw = iso?.trim()
  if (!raw) {
    return ''
  }
  const t = Date.parse(raw)
  if (Number.isNaN(t)) {
    return ''
  }
  const diffMs = Date.now() - t
  if (diffMs < 0) {
    return 'just now'
  }
  const sec = Math.floor(diffMs / 1000)
  if (sec < 45) {
    return 'just now'
  }
  const min = Math.floor(sec / 60)
  if (min < 60) {
    return `${min} minute${min === 1 ? '' : 's'} ago`
  }
  const h = Math.floor(min / 60)
  if (h < 24) {
    return `${h} hour${h === 1 ? '' : 's'} ago`
  }
  const d = Math.floor(h / 24)
  if (d === 1) {
    return 'yesterday'
  }
  if (d < 7) {
    return `${d} days ago`
  }
  if (d < 30) {
    return `${d} days ago`
  }
  return `${Math.floor(d / 30)} month${Math.floor(d / 30) === 1 ? '' : 's'} ago`
}

/** Last calendar day ``YYYY-MM-DD`` of a ``YYYY-MM`` month string. */
export function lastYmdOfMonth(ym: string): string {
  const m = ym.trim()
  if (!/^\d{4}-\d{2}$/.test(m)) {
    return ''
  }
  const [y, mo] = m.split('-').map((x) => Number(x))
  if (!y || !mo) {
    return ''
  }
  const last = new Date(Date.UTC(y, mo, 0))
  return last.toISOString().slice(0, 10)
}
