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
  it('maps direction to the success / danger / muted theme tokens', () => {
    expect(trendColor(2.0)).toBe('var(--ps-success)')
    expect(trendColor(0.4)).toBe('var(--ps-danger)')
    expect(trendColor(1.0)).toBe('var(--ps-muted)')
  })
})

describe('trendArrow', () => {
  it('maps direction to up / down / steady glyphs', () => {
    expect(trendArrow(2.0)).toBe('↑')
    expect(trendArrow(0.4)).toBe('↓')
    expect(trendArrow(1.0)).toBe('→')
  })
})
