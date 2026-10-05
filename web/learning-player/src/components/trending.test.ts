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
  it('maps direction to the --lp-trend-* tokens a direction can repaint', () => {
    expect(trendColor(2.0)).toBe('var(--lp-trend-rising)')
    expect(trendColor(0.5)).toBe('var(--lp-trend-cooling)')
    expect(trendColor(1.0)).toBe('var(--lp-trend-steady)')
  })
})

describe('trendArrow', () => {
  it('maps direction to up / down / steady glyphs', () => {
    expect(trendArrow(2.0)).toBe('↑')
    expect(trendArrow(0.5)).toBe('↓')
    expect(trendArrow(1.0)).toBe('→')
  })
})
