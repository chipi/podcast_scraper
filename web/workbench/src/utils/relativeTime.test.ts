import { describe, expect, it } from 'vitest'
import { formatAbsoluteUtc, formatRelativeShort } from './relativeTime'

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

describe('formatAbsoluteUtc', () => {
  it('renders a compact UTC label', () => {
    expect(formatAbsoluteUtc('2026-06-19T02:00:00Z')).toBe('2026-06-19 02:00 UTC')
  })

  it('is empty for blank input', () => {
    expect(formatAbsoluteUtc(null)).toBe('')
  })
})
