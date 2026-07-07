import { describe, expect, it } from 'vitest'
import type { Insight } from '../services/types'
import { insightScrubberMarkers } from './insightMarkers'

function q(startMs: number | null) {
  return { text: '', speaker: null, char_start: null, char_end: null, start_ms: startMs, end_ms: null }
}
function ins(over: Partial<Insight> = {}): Insight {
  return {
    id: 'i1',
    text: 't',
    grounded: true,
    insight_type: null,
    confidence: null,
    position_hint: null,
    quotes: [],
    ...over,
  }
}

describe('insightScrubberMarkers', () => {
  it('positions a marker by its EARLIEST quote as a % of duration', () => {
    const m = insightScrubberMarkers([ins({ id: 'a', quotes: [q(30000), q(15000)] })], 60)
    expect(m).toHaveLength(1)
    expect(m[0].timeSec).toBe(15) // earliest quote wins
    expect(m[0].pct).toBeCloseTo(25) // 15 / 60
    expect(m[0].grounded).toBe(true)
  })

  it('skips insights with no timestamped quote', () => {
    expect(insightScrubberMarkers([ins({ quotes: [] }), ins({ quotes: [q(null)] })], 60)).toEqual([])
  })

  it('returns [] when duration is unknown (<= 0)', () => {
    expect(insightScrubberMarkers([ins({ quotes: [q(1000)] })], 0)).toEqual([])
  })

  it('weight comes from confidence (clamped 0.2..1), default 0.6', () => {
    expect(insightScrubberMarkers([ins({ confidence: 0.9, quotes: [q(1000)] })], 60)[0].weight).toBeCloseTo(0.9)
    expect(insightScrubberMarkers([ins({ confidence: null, quotes: [q(1000)] })], 60)[0].weight).toBeCloseTo(0.6)
    expect(insightScrubberMarkers([ins({ confidence: 0.05, quotes: [q(1000)] })], 60)[0].weight).toBeCloseTo(0.2)
  })

  it('clamps position to 0..100 and sorts by position', () => {
    const m = insightScrubberMarkers(
      [ins({ id: 'late', quotes: [q(50000)] }), ins({ id: 'early', quotes: [q(5000)] })],
      60,
    )
    expect(m.map((x) => x.id)).toEqual(['early', 'late'])
    expect(m[0].pct).toBeGreaterThanOrEqual(0)
    expect(m[1].pct).toBeLessThanOrEqual(100)
  })
})
