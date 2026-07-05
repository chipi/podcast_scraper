import { describe, expect, it } from 'vitest'
import { trendArrow, trendColor, trendDirection } from './trending'

describe('trend direction thresholds', () => {
  it('classifies clearly rising / cooling / steady', () => {
    expect(trendDirection(2.0)).toBe('up')
    expect(trendDirection(1.15)).toBe('up') // inclusive lower bound of "up"
    expect(trendDirection(0.5)).toBe('down')
    expect(trendDirection(0.85)).toBe('down') // inclusive upper bound of "down"
    expect(trendDirection(1.0)).toBe('steady')
    expect(trendDirection(1.1)).toBe('steady') // neutral band around flat
  })
})

describe('trendColor', () => {
  it('maps direction to green / red / amber', () => {
    expect(trendColor(2.0)).toBe('#22c55e')
    expect(trendColor(0.5)).toBe('#ef4444')
    expect(trendColor(1.0)).toBe('#f59e0b')
  })
})

describe('trendArrow', () => {
  it('maps direction to up / down / steady glyphs', () => {
    expect(trendArrow(2.0)).toBe('↑')
    expect(trendArrow(0.5)).toBe('↓')
    expect(trendArrow(1.0)).toBe('→')
  })
})
