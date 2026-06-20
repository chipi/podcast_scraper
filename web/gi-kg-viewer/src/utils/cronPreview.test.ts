import { describe, expect, it } from 'vitest'
import { isValidCron, nextCronRuns } from './cronPreview'

describe('isValidCron', () => {
  it('accepts a standard 5-field expression', () => {
    expect(isValidCron('0 2 * * *')).toBe(true)
  })
  it('rejects blank / garbage', () => {
    expect(isValidCron('')).toBe(false)
    expect(isValidCron('not a cron')).toBe(false)
    expect(isValidCron(null)).toBe(false)
  })
})

describe('nextCronRuns', () => {
  it('previews the next N fire times from a fixed currentDate (UTC)', () => {
    const runs = nextCronRuns('0 2 * * *', 3, {
      tz: 'UTC',
      currentDate: new Date('2026-06-19T00:00:00Z'),
    })
    expect(runs).toEqual([
      '2026-06-19T02:00:00.000Z',
      '2026-06-20T02:00:00.000Z',
      '2026-06-21T02:00:00.000Z',
    ])
  })

  it('returns null for an invalid expression', () => {
    expect(nextCronRuns('nope')).toBeNull()
  })
})
