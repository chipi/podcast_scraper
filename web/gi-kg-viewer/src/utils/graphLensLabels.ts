import { localYmdDaysAgo } from './localCalendarDate'

/** Which graph lens control matches ``sinceYmd`` (Digest preset pattern). */
export function graphLensActivePreset(
  sinceYmd: string,
): 'all' | '7' | '30' | '90' | 'custom' {
  const s = sinceYmd.trim()
  if (!s) return 'all'
  if (s === localYmdDaysAgo(7)) return '7'
  if (s === localYmdDaysAgo(30)) return '30'
  if (s === localYmdDaysAgo(90)) return '90'
  return 'custom'
}

/** Human-readable graph lens label for the status line. */
export function graphLensSummaryLabel(sinceYmd: string): string {
  const s = sinceYmd.trim()
  if (!s) {
    return 'all time'
  }
  if (s === localYmdDaysAgo(7)) return 'last 7 days'
  if (s === localYmdDaysAgo(30)) return 'last 30 days'
  if (s === localYmdDaysAgo(90)) return 'last 90 days'
  return `since ${s}`
}

export function formatGraphNodeCount(n: number): string {
  if (!Number.isFinite(n) || n < 0) return '0'
  if (n >= 1000) {
    const k = n / 1000
    const t = k >= 10 ? k.toFixed(0) : k.toFixed(1).replace(/\.0$/, '')
    return `${t}k`
  }
  return String(Math.floor(n))
}
