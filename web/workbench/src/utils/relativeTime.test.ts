import { afterEach, describe, expect, it } from 'vitest'
import { formatAbsoluteLocal, formatRelativeShort } from './relativeTime'

const ORIGINAL_TZ = process.env.TZ

afterEach(() => {
  if (ORIGINAL_TZ === undefined) delete process.env.TZ
  else process.env.TZ = ORIGINAL_TZ
})

const NOW = Date.parse('2026-06-19T12:00:00Z')

describe('formatRelativeShort', () => {
  it('returns — for empty / invalid', () => {
    expect(formatRelativeShort(null, NOW)).toBe('—')
    expect(formatRelativeShort('not a date', NOW)).toBe('—')
  })

  it('formats near-now as now', () => {
    expect(formatRelativeShort('2026-06-19T12:00:10Z', NOW)).toBe('now')
  })

  it('formats future hours+minutes', () => {
    expect(formatRelativeShort('2026-06-19T15:12:00Z', NOW)).toBe('in 3h 12m')
  })

  it('formats future minutes only', () => {
    expect(formatRelativeShort('2026-06-19T12:45:00Z', NOW)).toBe('in 45m')
  })

  it('formats future days+hours', () => {
    expect(formatRelativeShort('2026-06-21T18:00:00Z', NOW)).toBe('in 2d 6h')
  })

  it('formats the past with ago', () => {
    expect(formatRelativeShort('2026-06-19T11:15:00Z', NOW)).toBe('45m ago')
  })
})

describe('formatRelativeShort naive timestamps', () => {
  it('reads a naive ISO datetime as UTC whatever the viewer zone', () => {
    for (const tz of ['America/New_York', 'Europe/Belgrade']) {
      process.env.TZ = tz
      expect(formatRelativeShort('2026-06-19T15:12:00', NOW)).toBe('in 3h 12m')
      expect(formatRelativeShort('2026-06-19T11:15:00', NOW)).toBe('45m ago')
    }
  })

  it('applies an explicit offset', () => {
    expect(formatRelativeShort('2026-06-19T17:12:00+02:00', NOW)).toBe('in 3h 12m')
  })

  it('is — for a bare date', () => {
    expect(formatRelativeShort('2026-06-19', NOW)).toBe('—')
  })
})

describe('formatAbsoluteLocal', () => {
  it('renders a compact local label with the zone name', () => {
    process.env.TZ = 'America/New_York'
    expect(formatAbsoluteLocal('2026-06-19T02:00:00Z')).toBe('2026-06-18 22:00 EDT')
  })

  it('reads a naive datetime as UTC', () => {
    process.env.TZ = 'America/New_York'
    expect(formatAbsoluteLocal('2026-09-17T02:00:00')).toBe('2026-09-16 22:00 EDT')
    process.env.TZ = 'Europe/Belgrade'
    expect(formatAbsoluteLocal('2026-09-17T02:00:00')).toBe('2026-09-17 04:00 GMT+2')
  })

  it('applies an explicit offset', () => {
    process.env.TZ = 'America/New_York'
    expect(formatAbsoluteLocal('2026-09-17T02:00:00+02:00')).toBe('2026-09-16 20:00 EDT')
  })

  it('returns garbage and a bare date unchanged', () => {
    expect(formatAbsoluteLocal('not-a-date')).toBe('not-a-date')
    expect(formatAbsoluteLocal('2026-09-17')).toBe('2026-09-17')
  })

  it('is empty for blank input', () => {
    expect(formatAbsoluteLocal(null)).toBe('')
  })
})
