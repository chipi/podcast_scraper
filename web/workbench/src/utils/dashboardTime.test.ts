import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { formatRelativeRunAge, lastYmdOfMonth } from './dashboardTime'

const NOW = Date.parse('2026-06-15T12:00:00Z')

function ago(ms: number): string {
  return new Date(NOW - ms).toISOString()
}

const SEC = 1000
const MIN = 60 * SEC
const HOUR = 60 * MIN
const DAY = 24 * HOUR

describe('formatRelativeRunAge', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(NOW)
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it.each([null, undefined, '', '   '])('returns empty string for %j', (v) => {
    expect(formatRelativeRunAge(v)).toBe('')
  })

  it('returns empty string for an unparseable timestamp', () => {
    expect(formatRelativeRunAge('not-a-date')).toBe('')
  })

  it('treats a future timestamp as just now', () => {
    expect(formatRelativeRunAge(ago(-5 * MIN))).toBe('just now')
  })

  it('treats under 45 seconds as just now', () => {
    expect(formatRelativeRunAge(ago(44 * SEC))).toBe('just now')
  })

  it('pluralises minutes', () => {
    expect(formatRelativeRunAge(ago(60 * SEC))).toBe('1 minute ago')
    expect(formatRelativeRunAge(ago(5 * MIN))).toBe('5 minutes ago')
    expect(formatRelativeRunAge(ago(59 * MIN))).toBe('59 minutes ago')
  })

  it('pluralises hours', () => {
    expect(formatRelativeRunAge(ago(HOUR))).toBe('1 hour ago')
    expect(formatRelativeRunAge(ago(23 * HOUR))).toBe('23 hours ago')
  })

  it('says yesterday for exactly one day', () => {
    expect(formatRelativeRunAge(ago(DAY + HOUR))).toBe('yesterday')
  })

  it('counts days up to a month', () => {
    expect(formatRelativeRunAge(ago(3 * DAY))).toBe('3 days ago')
    expect(formatRelativeRunAge(ago(6 * DAY))).toBe('6 days ago')
    expect(formatRelativeRunAge(ago(7 * DAY))).toBe('7 days ago')
    expect(formatRelativeRunAge(ago(29 * DAY))).toBe('29 days ago')
  })

  it('counts months from 30 days, with singular for one', () => {
    expect(formatRelativeRunAge(ago(30 * DAY))).toBe('1 month ago')
    expect(formatRelativeRunAge(ago(65 * DAY))).toBe('2 months ago')
  })

  it('trims surrounding whitespace before parsing', () => {
    expect(formatRelativeRunAge(`  ${ago(2 * HOUR)}  `)).toBe('2 hours ago')
  })
})

describe('lastYmdOfMonth', () => {
  it('returns the last day of 30- and 31-day months', () => {
    expect(lastYmdOfMonth('2026-04')).toBe('2026-04-30')
    expect(lastYmdOfMonth('2026-01')).toBe('2026-01-31')
    expect(lastYmdOfMonth('2026-12')).toBe('2026-12-31')
  })

  it('handles February in leap and non-leap years', () => {
    expect(lastYmdOfMonth('2026-02')).toBe('2026-02-28')
    expect(lastYmdOfMonth('2024-02')).toBe('2024-02-29')
  })

  it('trims whitespace', () => {
    expect(lastYmdOfMonth(' 2026-03 ')).toBe('2026-03-31')
  })

  it.each(['', '2026', '2026-3', '2026-03-01', 'abcd-ef'])(
    'returns empty string for malformed %j',
    (v) => {
      expect(lastYmdOfMonth(v)).toBe('')
    },
  )

  it('returns empty string for month 00', () => {
    expect(lastYmdOfMonth('2026-00')).toBe('')
  })
})
