import { describe, expect, it } from 'vitest'
import {
  bucketGiKgMtimesByDay,
  bucketGiMtimesByDay,
  cumulativeGiKgByDay,
  dailyGiKgNewCountsLastDays,
  sortedMonthHistogram,
  utcDayFromMtime,
} from './artifactMtimeBuckets'

describe('utcDayFromMtime', () => {
  it('extracts date prefix when already YYYY-MM-DD', () => {
    expect(utcDayFromMtime('2024-06-01T12:00:00Z')).toBe('2024-06-01')
  })

  it('parses full ISO', () => {
    expect(utcDayFromMtime('2024-12-31T23:59:59.000Z')).toBe('2024-12-31')
  })

  it('returns empty for garbage', () => {
    expect(utcDayFromMtime('')).toBe('')
    expect(utcDayFromMtime('not-a-date')).toBe('')
  })
})

describe('bucketGiMtimesByDay', () => {
  it('counts only gi kind and groups by day', () => {
    const rows = bucketGiMtimesByDay([
      { kind: 'gi', mtime_utc: '2024-01-02T10:00:00Z' },
      { kind: 'gi', mtime_utc: '2024-01-02T15:00:00Z' },
      { kind: 'kg', mtime_utc: '2024-01-03T10:00:00Z' },
      { kind: 'gi', mtime_utc: '2024-01-03T08:00:00Z' },
    ])
    expect(rows).toEqual([
      { day: '2024-01-02', count: 2 },
      { day: '2024-01-03', count: 1 },
    ])
  })
})

describe('bucketGiKgMtimesByDay', () => {
  it('counts gi and kg per day', () => {
    const rows = bucketGiKgMtimesByDay([
      { kind: 'gi', mtime_utc: '2024-01-01T00:00:00Z' },
      { kind: 'kg', mtime_utc: '2024-01-01T12:00:00Z' },
      { kind: 'gi', mtime_utc: '2024-01-02T00:00:00Z' },
    ])
    expect(rows).toEqual([
      { day: '2024-01-01', count: 2 },
      { day: '2024-01-02', count: 1 },
    ])
  })
})

describe('cumulativeGiKgByDay', () => {
  it('returns running totals per kind ordered by day', () => {
    const rows = cumulativeGiKgByDay([
      { kind: 'gi', mtime_utc: '2024-01-01T10:00:00Z' },
      { kind: 'kg', mtime_utc: '2024-01-01T12:00:00Z' },
      { kind: 'gi', mtime_utc: '2024-01-02T08:00:00Z' },
      { kind: 'kg', mtime_utc: '2024-01-02T09:00:00Z' },
    ])
    expect(rows).toEqual([
      { day: '2024-01-01', gi: 1, kg: 1 },
      { day: '2024-01-02', gi: 2, kg: 2 },
    ])
  })
})

describe('dailyGiKgNewCountsLastDays', () => {
  it('fills a fixed UTC window with per-day new counts', () => {
    const end = new Date(Date.UTC(2024, 5, 10))
    const rows = dailyGiKgNewCountsLastDays(
      [
        { kind: 'gi', mtime_utc: '2024-06-10T10:00:00Z' },
        { kind: 'kg', mtime_utc: '2024-06-09T12:00:00Z' },
      ],
      3,
      end,
    )
    expect(rows).toHaveLength(3)
    expect(rows[0]?.day).toBe('2024-06-08')
    expect(rows[1]).toEqual({ day: '2024-06-09', gi: 0, kg: 1 })
    expect(rows[2]).toEqual({ day: '2024-06-10', gi: 1, kg: 0 })
  })
})

describe('sortedMonthHistogram', () => {
  it('sorts keys lexically (YYYY-MM)', () => {
    const out = sortedMonthHistogram({ '2024-02': 1, '2023-11': 3, '2024-01': 2 })
    expect(out.map((x) => x.label)).toEqual(['2023-11', '2024-01', '2024-02'])
    expect(out.map((x) => x.count)).toEqual([3, 2, 1])
  })
})
