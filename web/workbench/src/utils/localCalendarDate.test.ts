import { describe, expect, it, vi } from 'vitest'
import { formatUtcYmd, inferCorpusLensPreset, utcYmdDaysAgo } from './localCalendarDate'

describe('localCalendarDate', () => {
  it('formatUtcYmd pads month and day', () => {
    expect(formatUtcYmd(new Date(Date.UTC(2024, 0, 5, 12)))).toBe('2024-01-05')
    expect(formatUtcYmd(new Date(Date.UTC(2024, 11, 31, 12)))).toBe('2024-12-31')
  })

  it('utcYmdDaysAgo counts UTC days, whatever the viewer zone', () => {
    const original = process.env.TZ
    vi.useFakeTimers()
    try {
      // 02:00 on 2026-09-17 in UTC+14 is 12:00Z on 2026-09-16: the server's day is the 16th.
      vi.setSystemTime(new Date('2026-09-16T12:00:00Z'))
      process.env.TZ = 'Pacific/Kiritimati'
      expect(utcYmdDaysAgo(0)).toBe('2026-09-16')
      expect(utcYmdDaysAgo(30)).toBe('2026-08-17')
      process.env.TZ = 'America/Los_Angeles'
      expect(utcYmdDaysAgo(0)).toBe('2026-09-16')
    } finally {
      vi.useRealTimers()
      if (original === undefined) delete process.env.TZ
      else process.env.TZ = original
    }
  })

  it('inferCorpusLensPreset returns all for empty', () => {
    expect(inferCorpusLensPreset('')).toBe('all')
    expect(inferCorpusLensPreset('  ')).toBe('all')
  })

  it('inferCorpusLensPreset matches rolling presets for same clock as utcYmdDaysAgo', () => {
    const seven = utcYmdDaysAgo(7)
    expect(inferCorpusLensPreset(seven)).toBe('7')
    expect(inferCorpusLensPreset('1999-01-01')).toBe('custom')
  })
})
