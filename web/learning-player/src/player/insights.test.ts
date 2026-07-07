import { describe, expect, it } from 'vitest'
import type { Insight, SearchHit, Segment } from '../services/types'
import {
  activeInsightIndex,
  groundedSpansBySegment,
  hitStartSeconds,
  insightStartSeconds,
  quoteHighlight,
} from './insights'

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

describe('groundedSpansBySegment', () => {
  const segs: Segment[] = [
    { id: 's0', start: 0, end: 5, text: 'a', speaker: null },
    { id: 's1', start: 5, end: 10, text: 'b', speaker: null },
    { id: 's2', start: 10, end: 15, text: 'c', speaker: null },
  ]
  it('maps segments overlapping an insight quote (by timeline) to that insight', () => {
    const out = groundedSpansBySegment(segs, [ins('i1', 6000, 9000)])
    expect(Object.keys(out)).toEqual(['1']) // only s1 [5,10) overlaps [6,9]
    expect(out[1].insightId).toBe('i1')
  })
  it('marks every overlapping segment when a quote spans a boundary', () => {
    const out = groundedSpansBySegment(segs, [ins('i1', 4000, 11000)])
    expect(Object.keys(out).sort()).toEqual(['0', '1', '2'])
  })
  it('earliest insight wins a shared segment; untimed quotes are skipped', () => {
    const out = groundedSpansBySegment(segs, [ins('first', 6000, 7000), ins('second', 7000, 8000)])
    expect(out[1].insightId).toBe('first')
    expect(groundedSpansBySegment(segs, [ins('x', null)])).toEqual({})
  })
})

describe('quoteHighlight', () => {
  it('splits the segment around the quote substring (case-insensitive, original casing)', () => {
    expect(quoteHighlight('Hello world.', 'world')).toEqual({
      pre: 'Hello ',
      match: 'world',
      post: '.',
    })
    expect(quoteHighlight('Hello World', 'world')?.match).toBe('World')
  })
  it('matches the whole segment when it sits inside a longer quote', () => {
    expect(quoteHighlight('the middle', 'this is the middle of a long quote')).toEqual({
      pre: '',
      match: 'the middle',
      post: '',
    })
  })
  it('returns null when the quote is not locatable (→ whole-segment fallback) or empty', () => {
    expect(quoteHighlight('abc def', 'xyz')).toBeNull()
    expect(quoteHighlight('abc', '   ')).toBeNull()
    expect(quoteHighlight('', 'abc')).toBeNull()
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
