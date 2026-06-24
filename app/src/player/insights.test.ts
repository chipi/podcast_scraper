import { describe, expect, it } from 'vitest'
import type { Insight, SearchHit } from '../services/types'
import { activeInsightIndex, hitStartSeconds, insightStartSeconds } from './insights'

function ins(id: string, startMs: number | null, endMs: number | null = null): Insight {
  return {
    id,
    text: id,
    grounded: true,
    insight_type: null,
    confidence: null,
    position_hint: null,
    quotes: [
      { text: 'q', speaker: null, char_start: null, char_end: null, start_ms: startMs, end_ms: endMs },
    ],
  }
}

describe('insightStartSeconds', () => {
  it('returns the earliest quote start in seconds', () => {
    expect(insightStartSeconds(ins('a', 12000))).toBe(12)
  })
  it('returns null when untimed', () => {
    expect(insightStartSeconds(ins('a', null))).toBeNull()
  })
})

describe('activeInsightIndex', () => {
  const list = [ins('a', 0, 5000), ins('b', 10000, 14000)]
  it('returns the insight whose quote window contains t', () => {
    expect(activeInsightIndex(list, 3)).toBe(0)
    expect(activeInsightIndex(list, 12)).toBe(1)
  })
  it('returns -1 when none active', () => {
    expect(activeInsightIndex(list, 7)).toBe(-1)
  })
  it('assumes ~8s window when no end marker', () => {
    expect(activeInsightIndex([ins('a', 10000, null)], 12)).toBe(0)
    expect(activeInsightIndex([ins('a', 10000, null)], 20)).toBe(-1)
  })
})

describe('hitStartSeconds', () => {
  const base: SearchHit = { doc_id: 'd', score: 1, text: 't', metadata: {}, source_tier: 'segment' }
  it('reads lifted.quote.timestamp_start_ms', () => {
    expect(hitStartSeconds({ ...base, lifted: { quote: { timestamp_start_ms: 20000 } } })).toBe(20)
  })
  it('reads supporting_quotes start_ms', () => {
    expect(hitStartSeconds({ ...base, supporting_quotes: [{ start_ms: 5000 }] })).toBe(5)
  })
  it('falls back to metadata timestamp', () => {
    expect(hitStartSeconds({ ...base, metadata: { timestamp_start_ms: 9000 } })).toBe(9)
  })
  it('returns null when no timestamp is derivable', () => {
    expect(hitStartSeconds(base)).toBeNull()
  })
})
