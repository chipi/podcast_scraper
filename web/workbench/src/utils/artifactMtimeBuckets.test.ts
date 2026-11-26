import { afterEach, describe, expect, it } from 'vitest'
import {
  dailyNewCountsLastDays,
  sortedMonthHistogram,
  localDayFromMtime,
} from './artifactMtimeBuckets'

describe('localDayFromMtime', () => {
  it('extracts date prefix when already YYYY-MM-DD', () => {
    expect(localDayFromMtime('2024-06-01T12:00:00Z')).toBe('2024-06-01')
  })

  it('parses full ISO', () => {
    expect(localDayFromMtime('2024-12-31T12:00:00.000Z')).toBe('2024-12-31')
  })

  it('returns empty for garbage', () => {
    expect(localDayFromMtime('')).toBe('')
    expect(localDayFromMtime('not-a-date')).toBe('')
  })
})

describe('dailyNewCountsLastDays', () => {
  it('fills a fixed local window with per-day new counts for the kinds asked', () => {
    const end = new Date(2024, 5, 10, 12)
    const rows = dailyNewCountsLastDays(
      [
        { kind: 'gi', mtime_utc: '2024-06-10T12:00:00Z' },
        { kind: 'kg', mtime_utc: '2024-06-09T12:00:00Z' },
        { kind: 'other', mtime_utc: '2024-06-09T12:00:00Z' },
        { kind: 'kg', mtime_utc: 'not a date' },
      ],
      ['gi', 'kg'],
      3,
      end,
    )
    expect(rows).toHaveLength(3)
    expect(rows[0]).toEqual({ day: '2024-06-08', counts: { gi: 0, kg: 0 } })
    expect(rows[1]).toEqual({ day: '2024-06-09', counts: { gi: 0, kg: 1 } })
    expect(rows[2]).toEqual({ day: '2024-06-10', counts: { gi: 1, kg: 0 } })
    // A kind not asked for is not counted.
    expect(dailyNewCountsLastDays([{ kind: 'gi', mtime_utc: '2024-06-10T12:00:00Z' }], ['kg'], 1, end)).toEqual([
      { day: '2024-06-10', counts: { kg: 0 } },
    ])
  })
})

describe('sortedMonthHistogram', () => {
  it('sorts keys lexically (YYYY-MM)', () => {
    const out = sortedMonthHistogram({ '2024-02': 1, '2023-11': 3, '2024-01': 2 })
    expect(out.map((x) => x.label)).toEqual(['2023-11', '2024-01', '2024-02'])
    expect(out.map((x) => x.count)).toEqual([3, 2, 1])
  })
})

describe('local-day bucketing', () => {
  const original = process.env.TZ
  afterEach(() => {
    if (original === undefined) delete process.env.TZ
    else process.env.TZ = original
  })

  it('files a late-UTC write under the previous day in New York', () => {
    process.env.TZ = 'America/New_York'
    expect(localDayFromMtime('2024-06-10T02:00:00Z')).toBe('2024-06-09')
    expect(localDayFromMtime('2024-06-10T02:00:00')).toBe('2024-06-09')
    const rows = dailyNewCountsLastDays(
      [{ kind: 'gi', mtime_utc: '2024-06-10T02:00:00Z' }],
      ['gi'],
      2,
      new Date(2024, 5, 10, 12),
    )
    expect(rows).toEqual([
      { day: '2024-06-09', counts: { gi: 1 } },
      { day: '2024-06-10', counts: { gi: 0 } },
    ])
  })
})
