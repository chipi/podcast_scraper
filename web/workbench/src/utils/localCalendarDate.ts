/** UTC calendar YYYY-MM-DD of ``d``: the day the server's ``since`` filter compares with. */
export function formatUtcYmd(d: Date): string {
  return d.toISOString().slice(0, 10)
}

/** Today minus ``days`` in the UTC calendar, as YYYY-MM-DD (a ``since`` value). */
export function utcYmdDaysAgo(days: number): string {
  const d = new Date()
  d.setUTCDate(d.getUTCDate() - days)
  return formatUtcYmd(d)
}

export type CorpusLensPreset = 'all' | '7' | '30' | '90' | 'custom'

/** Map current ``sinceYmd`` to which Library/Digest preset it matches (if any). */
export function inferCorpusLensPreset(sinceYmd: string): CorpusLensPreset {
  const s = sinceYmd.trim()
  if (!s) {
    return 'all'
  }
  for (const n of [7, 30, 90] as const) {
    if (s === utcYmdDaysAgo(n)) {
      return String(n) as CorpusLensPreset
    }
  }
  return 'custom'
}
