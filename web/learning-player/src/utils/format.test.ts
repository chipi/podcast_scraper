import { describe, expect, it } from 'vitest'
import { formatPlayedAt, formatDuration, formatPublishDate } from './format'

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

describe('formatPlayedAt', () => {
  // Recently played is ORDERED by this stamp, so it has to carry the clock: two sittings with the
  // same show on the same day are otherwise indistinguishable (operator 2026-09-23).
  it('returns date and time SEPARATELY, so the caller can stack them', () => {
    // One string ("Sep 24, 2026, 8:45 AM") was wider than the artwork it sits under and stretched
    // the whole left column of the card (operator 2026-09-23).
    const out = formatPlayedAt(1_760_000_000, 'en')
    expect(out).toBeTruthy()
    expect(out!.time).toMatch(/\d{1,2}:\d{2}/)
    expect(out!.date).not.toMatch(/\d{1,2}:\d{2}/)
  })

  it('returns null for a missing or unusable stamp rather than a fake date', () => {
    // `updated_at` is nullable on PlaybackPosition; rendering "1 Jan 1970" for it would be worse
    // than rendering nothing.
    expect(formatPlayedAt(null)).toBeNull()
    expect(formatPlayedAt(undefined)).toBeNull()
    expect(formatPlayedAt(Number.NaN)).toBeNull()
  })
})

