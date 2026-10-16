import { describe, expect, it } from 'vitest'

import { toStream, toTranscript, type AudioTimeMap } from './audioTimeMap'

// Measured on a real episode (2026-10-16): +7.9 s, then an old 83 s ad gone (-75 s), then a
// 109 s ad added (+34.2 s).
const MAP: AudioTimeMap = [
  [0, 7.9],
  [1187.5, -75],
  [2564.7, 34.2],
]

describe('audio time map', () => {
  it('without a map, both directions are the identity', () => {
    expect(toStream(null, 42)).toBe(42)
    expect(toTranscript(undefined, 42)).toBe(42)
    expect(toStream([], 42)).toBe(42)
  })

  it('applies the offset in force at each transcript time', () => {
    expect(toStream(MAP, 100)).toBeCloseTo(107.9)
    expect(toStream(MAP, 1187.4)).toBeCloseTo(1195.3)
    expect(toStream(MAP, 1187.5)).toBeCloseTo(1112.5)
    expect(toStream(MAP, 3000)).toBeCloseTo(3034.2)
  })

  it('round-trips everywhere the stream has the transcript', () => {
    for (const t of [0, 5, 600, 1100, 1187.5, 1500, 2564.7, 3000]) {
      expect(toTranscript(MAP, toStream(MAP, t))).toBeCloseTo(t)
    }
  })

  it('inside an inserted ad, holds where the talk resumes', () => {
    // Talk before the ad ends at transcript 2564.7 - 75 = 2489.7 in the stream; resumes at 2598.9.
    expect(toTranscript(MAP, 2550)).toBeCloseTo(2564.7)
  })

  it('where an old ad is gone, the later part wins', () => {
    // Stream 1150 is both "transcript 1142.1" (old offset) and "1225" (new): the old ad is gone,
    // so the stream already plays the later part.
    expect(toTranscript(MAP, 1150)).toBeCloseTo(1225)
  })

  it('before the first anchor, the first offset applies', () => {
    expect(toTranscript([[0, 12]], 5)).toBeCloseTo(-7)
  })
})
