import { describe, expect, it } from 'vitest'
import { formatDuration, formatPublishDate } from './format'

describe('formatDuration', () => {
  it('returns minutes under an hour', () => {
    expect(formatDuration(48 * 60)).toBe('48 min')
  })
  it('returns h/m at or over an hour, zero-padded minutes', () => {
    expect(formatDuration(3600 + 3 * 60)).toBe('1h 03m')
  })
  it('returns null for null / zero / negative', () => {
    expect(formatDuration(null)).toBeNull()
    expect(formatDuration(0)).toBeNull()
    expect(formatDuration(-5)).toBeNull()
  })
})

describe('formatPublishDate', () => {
  it('formats a YYYY-MM-DD date for the locale', () => {
    // Locale formatting varies; assert it parsed (not the raw ISO) and includes the year.
    const out = formatPublishDate('2024-03-10', 'en')
    expect(out).toContain('2024')
    expect(out).not.toBe('2024-03-10')
  })
  it('passes through unparseable input and null', () => {
    expect(formatPublishDate(null)).toBeNull()
    expect(formatPublishDate('not-a-date')).toBe('not-a-date')
  })
})
