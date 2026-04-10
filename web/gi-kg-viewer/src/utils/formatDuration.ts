/** Human-readable duration from seconds (podcast metadata). */
export function formatDurationSeconds(total: number | null | undefined): string {
  if (total == null || total < 0 || !Number.isFinite(total)) {
    return ''
  }
  const s = Math.floor(total)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  if (h > 0) {
    return m > 0 ? `${h}h ${m}m` : `${h}h`
  }
  if (m > 0) {
    return `${m}m`
  }
  return `${s}s`
}
