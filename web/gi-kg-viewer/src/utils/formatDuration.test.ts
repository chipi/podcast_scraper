import { describe, expect, it } from 'vitest'
import { formatDashboardRunDurationSeconds, formatDurationSeconds } from './formatDuration'

describe('formatDurationSeconds', () => {
  it('formats hours and minutes', () => {
    expect(formatDurationSeconds(3661)).toBe('1h 1m')
  })
  it('formats hours only when zero remaining minutes', () => {
    expect(formatDurationSeconds(3600)).toBe('1h')
    expect(formatDurationSeconds(7200)).toBe('2h')
  })
  it('drops the seconds component once past an hour', () => {
    // 1h 0m 59s -> seconds are not surfaced at the hour scale
    expect(formatDurationSeconds(3659)).toBe('1h')
  })
  it('formats minutes only', () => {
    expect(formatDurationSeconds(125)).toBe('2m')
  })
  it('drops trailing seconds at the minute scale', () => {
    expect(formatDurationSeconds(119)).toBe('1m')
    expect(formatDurationSeconds(60)).toBe('1m')
  })
  it('formats seconds when under one minute', () => {
    expect(formatDurationSeconds(45)).toBe('45s')
  })
  it('formats zero as 0s', () => {
    expect(formatDurationSeconds(0)).toBe('0s')
  })
  it('floors fractional seconds', () => {
    expect(formatDurationSeconds(45.9)).toBe('45s')
    expect(formatDurationSeconds(59.9999)).toBe('59s')
    expect(formatDurationSeconds(3661.7)).toBe('1h 1m')
  })
  it('handles large values', () => {
    // 100h 1m 1s
    expect(formatDurationSeconds(360061)).toBe('100h 1m')
  })
  it('returns empty for null', () => {
    expect(formatDurationSeconds(null)).toBe('')
  })
  it('returns empty for undefined', () => {
    expect(formatDurationSeconds(undefined)).toBe('')
  })
  it('returns empty for negative values', () => {
    expect(formatDurationSeconds(-1)).toBe('')
    expect(formatDurationSeconds(-3600)).toBe('')
  })
  it('returns empty for non-finite values', () => {
    expect(formatDurationSeconds(Number.POSITIVE_INFINITY)).toBe('')
    expect(formatDurationSeconds(Number.NEGATIVE_INFINITY)).toBe('')
    expect(formatDurationSeconds(Number.NaN)).toBe('')
  })
})

describe('formatDashboardRunDurationSeconds', () => {
  it('formats hours, minutes and seconds', () => {
    expect(formatDashboardRunDurationSeconds(3661)).toBe('1h 1m 1s')
  })
  it('formats hours and seconds when zero minutes', () => {
    expect(formatDashboardRunDurationSeconds(3601)).toBe('1h 1s')
    expect(formatDashboardRunDurationSeconds(3600)).toBe('1h 0s')
  })
  it('formats hours with both minutes and seconds present', () => {
    expect(formatDashboardRunDurationSeconds(7325)).toBe('2h 2m 5s')
  })
  it('formats minutes and seconds', () => {
    expect(formatDashboardRunDurationSeconds(204)).toBe('3m 24s')
  })
  it('formats minutes with zero seconds', () => {
    expect(formatDashboardRunDurationSeconds(120)).toBe('2m 0s')
  })
  it('formats seconds only when under one minute', () => {
    expect(formatDashboardRunDurationSeconds(45)).toBe('45s')
  })
  it('formats zero as 0s', () => {
    expect(formatDashboardRunDurationSeconds(0)).toBe('0s')
  })
  it('floors fractional seconds', () => {
    expect(formatDashboardRunDurationSeconds(204.99)).toBe('3m 24s')
    expect(formatDashboardRunDurationSeconds(3661.4)).toBe('1h 1m 1s')
  })
  it('handles large values', () => {
    // 100h 1m 1s
    expect(formatDashboardRunDurationSeconds(360061)).toBe('100h 1m 1s')
  })
  it('returns empty for null', () => {
    expect(formatDashboardRunDurationSeconds(null)).toBe('')
  })
  it('returns empty for undefined', () => {
    expect(formatDashboardRunDurationSeconds(undefined)).toBe('')
  })
  it('returns empty for negative values', () => {
    expect(formatDashboardRunDurationSeconds(-1)).toBe('')
  })
  it('returns empty for non-finite values', () => {
    expect(formatDashboardRunDurationSeconds(Number.POSITIVE_INFINITY)).toBe('')
    expect(formatDashboardRunDurationSeconds(Number.NaN)).toBe('')
  })
})
