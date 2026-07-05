import { describe, expect, it } from 'vitest'
import { trendArrow, trendColor, trendDirection } from './trend'

describe('trendDirection', () => {
  it('classifies rising / cooling / steady with a neutral band around flat', () => {
    expect(trendDirection(2.0)).toBe('up')
    expect(trendDirection(1.15)).toBe('up') // inclusive lower bound
    expect(trendDirection(1.1)).toBe('steady')
    expect(trendDirection(1.0)).toBe('steady')
    expect(trendDirection(0.85)).toBe('down') // inclusive upper bound
    expect(trendDirection(0.4)).toBe('down')
  })
})

describe('trendColor', () => {
  it('maps direction to green / red / amber hex', () => {
    expect(trendColor(2.0)).toBe('#22c55e')
    expect(trendColor(0.4)).toBe('#ef4444')
    expect(trendColor(1.0)).toBe('#f59e0b')
  })
})

describe('trendArrow', () => {
  it('maps direction to up / down / steady glyphs', () => {
    expect(trendArrow(2.0)).toBe('↑')
    expect(trendArrow(0.4)).toBe('↓')
    expect(trendArrow(1.0)).toBe('→')
  })
})
