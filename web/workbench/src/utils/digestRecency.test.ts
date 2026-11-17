import { describe, expect, it, vi } from 'vitest'
import {
  isPublishDateWithin24hRolling,
  parsePublishYmdLocalMidnight,
  recencyDotHoverTitle,
} from './digestRecency'

describe('digestRecency', () => {
  it('parsePublishYmdLocalMidnight rejects invalid patterns', () => {
    expect(parsePublishYmdLocalMidnight('')).toBeNull()
    expect(parsePublishYmdLocalMidnight('04-19-2026')).toBeNull()
    expect(parsePublishYmdLocalMidnight('2026-13-40')).toBeNull()
  })

  it('isPublishDateWithin24hRolling respects mocked clock', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date(2026, 3, 19, 10, 0, 0))
    expect(isPublishDateWithin24hRolling('2026-04-19')).toBe(true)
    expect(isPublishDateWithin24hRolling('2026-04-18')).toBe(false)
    expect(isPublishDateWithin24hRolling('2026-04-17')).toBe(false)
    vi.useRealTimers()
  })

  it('recencyDotHoverTitle returns hours phrasing', () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date(2026, 3, 19, 15, 30, 0))
    expect(recencyDotHoverTitle('2026-04-19')).toBe('Published 15 hours ago')
    vi.setSystemTime(new Date(2026, 3, 19, 0, 20, 0))
    expect(recencyDotHoverTitle('2026-04-19')).toBe('Published less than 1 hour ago')
    vi.useRealTimers()
  })
})
