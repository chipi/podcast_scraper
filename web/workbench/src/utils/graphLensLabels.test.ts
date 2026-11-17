import { describe, expect, it } from 'vitest'
import {
  formatGraphNodeCount,
  graphLensActivePreset,
  graphLensSummaryLabel,
} from './graphLensLabels'
import { localYmdDaysAgo } from './localCalendarDate'

describe('graphLensActivePreset', () => {
  it('detects all time', () => {
    expect(graphLensActivePreset('')).toBe('all')
    expect(graphLensActivePreset('   ')).toBe('all')
  })

  it('detects 7/30/90 presets', () => {
    expect(graphLensActivePreset(localYmdDaysAgo(7))).toBe('7')
    expect(graphLensActivePreset(localYmdDaysAgo(30))).toBe('30')
    expect(graphLensActivePreset(localYmdDaysAgo(90))).toBe('90')
  })

  it('trims surrounding whitespace before matching presets', () => {
    expect(graphLensActivePreset(`  ${localYmdDaysAgo(7)}  `)).toBe('7')
    expect(graphLensActivePreset(`\t${localYmdDaysAgo(30)}\n`)).toBe('30')
  })

  it('treats arbitrary YYYY-MM-DD as custom', () => {
    expect(graphLensActivePreset('2020-01-15')).toBe('custom')
  })

  it('treats a non-matching trimmed value as custom', () => {
    expect(graphLensActivePreset('  not-a-preset  ')).toBe('custom')
  })
})

describe('graphLensSummaryLabel', () => {
  it('labels empty/whitespace as all time', () => {
    expect(graphLensSummaryLabel('')).toBe('all time')
    expect(graphLensSummaryLabel('   ')).toBe('all time')
  })

  it('labels the 7/30/90 presets', () => {
    expect(graphLensSummaryLabel(localYmdDaysAgo(7))).toBe('last 7 days')
    expect(graphLensSummaryLabel(localYmdDaysAgo(30))).toBe('last 30 days')
    expect(graphLensSummaryLabel(localYmdDaysAgo(90))).toBe('last 90 days')
  })

  it('trims before matching presets', () => {
    expect(graphLensSummaryLabel(`  ${localYmdDaysAgo(90)}  `)).toBe(
      'last 90 days',
    )
  })

  it('falls back to "since <date>" for custom values', () => {
    expect(graphLensSummaryLabel('2020-01-15')).toBe('since 2020-01-15')
  })

  it('uses the trimmed value in the custom fallback', () => {
    expect(graphLensSummaryLabel('  2019-12-31  ')).toBe('since 2019-12-31')
  })
})

describe('formatGraphNodeCount', () => {
  it('returns "0" for non-finite and negative values', () => {
    expect(formatGraphNodeCount(Number.NaN)).toBe('0')
    expect(formatGraphNodeCount(Number.POSITIVE_INFINITY)).toBe('0')
    expect(formatGraphNodeCount(Number.NEGATIVE_INFINITY)).toBe('0')
    expect(formatGraphNodeCount(-1)).toBe('0')
    expect(formatGraphNodeCount(-1000)).toBe('0')
  })

  it('returns "0" for zero', () => {
    expect(formatGraphNodeCount(0)).toBe('0')
  })

  it('floors sub-1000 values', () => {
    expect(formatGraphNodeCount(1)).toBe('1')
    expect(formatGraphNodeCount(42)).toBe('42')
    expect(formatGraphNodeCount(999)).toBe('999')
    expect(formatGraphNodeCount(123.9)).toBe('123')
  })

  it('formats thousands with one decimal, stripping trailing .0', () => {
    expect(formatGraphNodeCount(1000)).toBe('1k')
    expect(formatGraphNodeCount(1500)).toBe('1.5k')
    expect(formatGraphNodeCount(1234)).toBe('1.2k')
    expect(formatGraphNodeCount(9999)).toBe('10k')
  })

  it('drops the decimal once at/above 10k', () => {
    expect(formatGraphNodeCount(10000)).toBe('10k')
    expect(formatGraphNodeCount(10500)).toBe('11k')
    expect(formatGraphNodeCount(12345)).toBe('12k')
    expect(formatGraphNodeCount(999999)).toBe('1000k')
  })
})
