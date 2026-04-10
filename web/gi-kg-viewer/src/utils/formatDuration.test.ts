import { describe, expect, it } from 'vitest'
import { formatDurationSeconds } from './formatDuration'

describe('formatDurationSeconds', () => {
  it('formats hours and minutes', () => {
    expect(formatDurationSeconds(3661)).toBe('1h 1m')
  })
  it('formats minutes only', () => {
    expect(formatDurationSeconds(125)).toBe('2m')
  })
  it('formats seconds when under one minute', () => {
    expect(formatDurationSeconds(45)).toBe('45s')
  })
  it('returns empty for null', () => {
    expect(formatDurationSeconds(null)).toBe('')
  })
})
