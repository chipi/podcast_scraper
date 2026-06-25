import { describe, expect, it } from 'vitest'
import type { Segment } from '../services/types'
import { activeSegmentIndex, formatTime, PLAYBACK_RATES } from './transcriptSync'

function seg(id: string, start: number, end: number): Segment {
  return { id, start, end, text: id, speaker: null }
}

const segments: Segment[] = [seg('a', 0, 2.5), seg('b', 2.5, 5), seg('c', 5, 9)]

describe('activeSegmentIndex', () => {
  it('returns -1 before the first segment starts', () => {
    expect(activeSegmentIndex(segments, -1)).toBe(-1)
  })
  it('picks the segment whose start <= t', () => {
    expect(activeSegmentIndex(segments, 0)).toBe(0)
    expect(activeSegmentIndex(segments, 2.4)).toBe(0)
    expect(activeSegmentIndex(segments, 2.5)).toBe(1)
    expect(activeSegmentIndex(segments, 6)).toBe(2)
  })
  it('stays on the last segment past the end', () => {
    expect(activeSegmentIndex(segments, 100)).toBe(2)
  })
  it('handles an empty transcript', () => {
    expect(activeSegmentIndex([], 3)).toBe(-1)
  })
})

describe('formatTime', () => {
  it('formats m:ss', () => {
    expect(formatTime(0)).toBe('0:00')
    expect(formatTime(64)).toBe('1:04')
    expect(formatTime(724)).toBe('12:04')
  })
  it('formats h:mm:ss past an hour', () => {
    expect(formatTime(3725)).toBe('1:02:05')
  })
  it('clamps negatives and NaN', () => {
    expect(formatTime(-5)).toBe('0:00')
    expect(formatTime(Number.NaN)).toBe('0:00')
  })
})

describe('PLAYBACK_RATES', () => {
  it('matches the PRD speed set', () => {
    expect([...PLAYBACK_RATES]).toEqual([0.75, 1, 1.25, 1.5, 2])
  })
})
