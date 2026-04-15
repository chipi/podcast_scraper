/** Local calendar YYYY-MM-DD (same convention as Library ``since`` filter). */
export function formatLocalYmd(d: Date): string {
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}

/** Today minus ``days`` in the local calendar, as YYYY-MM-DD. */
export function localYmdDaysAgo(days: number): string {
  const d = new Date()
  d.setDate(d.getDate() - days)
  return formatLocalYmd(d)
}

export type CorpusLensPreset = 'all' | '7' | '30' | '90' | 'custom'

/** Map current ``sinceYmd`` to which Library/Digest preset it matches (if any). */
export function inferCorpusLensPreset(sinceYmd: string): CorpusLensPreset {
  const s = sinceYmd.trim()
  if (!s) {
    return 'all'
  }
  for (const n of [7, 30, 90] as const) {
    if (s === localYmdDaysAgo(n)) {
      return String(n) as CorpusLensPreset
    }
  }
  return 'custom'
}
