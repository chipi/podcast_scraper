import { describe, expect, it } from 'vitest'
import { formatLocalYmd, inferCorpusLensPreset, localYmdDaysAgo } from './localCalendarDate'

describe('localCalendarDate', () => {
  it('formatLocalYmd pads month and day', () => {
    expect(formatLocalYmd(new Date(2024, 0, 5))).toBe('2024-01-05')
    expect(formatLocalYmd(new Date(2024, 11, 31))).toBe('2024-12-31')
  })

  it('inferCorpusLensPreset returns all for empty', () => {
    expect(inferCorpusLensPreset('')).toBe('all')
    expect(inferCorpusLensPreset('  ')).toBe('all')
  })

  it('inferCorpusLensPreset matches rolling presets for same clock as localYmdDaysAgo', () => {
    const seven = localYmdDaysAgo(7)
    expect(inferCorpusLensPreset(seven)).toBe('7')
    expect(inferCorpusLensPreset('1999-01-01')).toBe('custom')
  })
})
