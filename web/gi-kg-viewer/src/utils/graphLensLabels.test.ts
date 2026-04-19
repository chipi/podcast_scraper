import { describe, expect, it } from 'vitest'
import { graphLensActivePreset } from './graphLensLabels'
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

  it('treats arbitrary YYYY-MM-DD as custom', () => {
    expect(graphLensActivePreset('2020-01-15')).toBe('custom')
  })
})
