import { describe, expect, it } from 'vitest'
import type { Insight, SearchHit, Segment } from '../services/types'
import {
  activeInsightIndex,
  groundedMomentCount,
  groundedSpansBySegment,
  hitStartSeconds,
  INSIGHT_LINGER_MS,
  insightStartSeconds,
  nextInsightIndex,
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

  // Regression guard for the rejected prototype: "surfacing now, else next, else the FIRST
  // insight" always had something on screen, including before the episode has said a word.
  it('returns -1 at t=0 when the first quote starts later — never falls back to the first insight', () => {
    expect(activeInsightIndex([ins('a', 5000, 10000)], 0)).toBe(-1)
  })

  describe('with a linger', () => {
    const single = [ins('a', 0, 5000)]

    it('stays on the insight for lingerMs after its window ends', () => {
      // 1s after the 5000ms window ends, well inside a 4000ms linger.
      expect(activeInsightIndex(single, 6, INSIGHT_LINGER_MS)).toBe(0)
    })

    it('drops back to -1 once the linger has elapsed', () => {
      // 5s after the window ends — past a 4000ms linger.
      expect(activeInsightIndex(single, 10, INSIGHT_LINGER_MS)).toBe(-1)
    })

    it('is a no-op for existing 2-arg callers (default lingerMs=0)', () => {
      expect(activeInsightIndex(single, 6)).toBe(-1)
    })
  })
})

describe('nextInsightIndex', () => {
  it('picks the earliest insight starting at least the look-ahead floor after t', () => {
    const list = [ins('a', 0, 5000), ins('b', 20000, 25000), ins('c', 12000, 17000)]
    // At t=0, 'a' is already current (its own start), 'c' starts at 12s (>= 0 + 5s floor) and is
    // earlier than 'b' at 20s, so 'c' wins.
    expect(nextInsightIndex(list, 0)).toBe(2)
  })

  it('excludes a candidate inside the look-ahead floor (would preview a moment already playing)', () => {
    // Starts 3s after t — inside the 5s floor — so it does not count as "next" yet.
    expect(nextInsightIndex([ins('a', 3000, 8000)], 0)).toBe(-1)
  })

  it('never picks a degenerate 0/0-timestamped ("authored") insight as next', () => {
    const list = [ins('authored', 0, 0), ins('real', 20000, 25000)]
    expect(nextInsightIndex(list, 0)).toBe(1)
  })

  it('returns -1 when nothing untimed or upcoming exists', () => {
    expect(nextInsightIndex([ins('a', null)], 0)).toBe(-1)
    expect(nextInsightIndex([], 0)).toBe(-1)
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

describe('degenerate quote windows never surface as "now" (#1978 follow-up)', () => {
  const authored = (): Insight[] =>
    [{ id: 'i1', text: 'x', quotes: [{ start_ms: 0, end_ms: 0 }] }] as unknown as Insight[]

  it('an authored 0/0 quote is not active at t=0, even with the linger', () => {
    // The regression an adversarial review found: `end` resolved to 0, the window became
    // [0, linger], and the insight was "surfacing now" before playback began — on the What's-new
    // hero episode, which is the first thing most testers will open.
    expect(activeInsightIndex(authored(), 0, INSIGHT_LINGER_MS)).toBe(-1)
    expect(activeInsightIndex(authored(), 2, INSIGHT_LINGER_MS)).toBe(-1)
    expect(activeInsightIndex(authored(), 3.9, INSIGHT_LINGER_MS)).toBe(-1)
  })

  it('a real quote is still active at its start and through the linger', () => {
    // The guard must not swallow genuine windows: zero-width is the disqualifier, not zero-start.
    const real = [
      { id: 'i1', text: 'x', quotes: [{ start_ms: 0, end_ms: 6000 }] },
    ] as unknown as Insight[]
    expect(activeInsightIndex(real, 0, INSIGHT_LINGER_MS)).toBe(0)
    expect(activeInsightIndex(real, 6, INSIGHT_LINGER_MS)).toBe(0)
    expect(activeInsightIndex(real, 9.9, INSIGHT_LINGER_MS)).toBe(0) // inside the linger
    expect(activeInsightIndex(real, 10.1, INSIGHT_LINGER_MS)).toBe(-1) // past it
  })

  it('a quote with no end_ms keeps the 8s assumption', () => {
    // Missing end is not degenerate — the quote has a real start, we just do not know where it stops.
    const open = [{ id: 'i1', text: 'x', quotes: [{ start_ms: 1000 }] }] as unknown as Insight[]
    expect(activeInsightIndex(open, 5, INSIGHT_LINGER_MS)).toBe(0)
  })
})

describe('groundedMomentCount reports moments, not transcript chunks (#1978 follow-up)', () => {
  // Six seconds of audio, sliced into three transcript segments. How finely a transcript happens
  // to be chunked is an artefact of the transcriber, and must not change what the reader is told.
  const segs: Segment[] = [
    { start: 0, end: 2, text: 'one', speaker: null },
    { start: 2, end: 4, text: 'two', speaker: null },
    { start: 4, end: 6, text: 'three', speaker: null },
  ]

  function withQuotes(...windows: Array<[number | null, number | null]>): Insight {
    return {
      id: 'i',
      text: 'i',
      grounded: true,
      insight_type: null,
      confidence: null,
      position_hint: null,
      quotes: windows.map(([start_ms, end_ms]) => ({
        text: 'q',
        speaker: null,
        char_start: null,
        char_end: null,
        start_ms,
        end_ms,
      })),
    }
  }

  it('counts ONE moment for a single quote spanning three segments', () => {
    // The exact regression. `groundedSpansBySegment` is keyed by segment, so the old receipt read
    // "Sourced to 3 moments in the transcript" for one continuous quotation.
    const insight = withQuotes([0, 6000])
    expect(Object.keys(groundedSpansBySegment(segs, [insight]))).toHaveLength(3)
    expect(groundedMomentCount(segs, insight)).toBe(1)
  })

  it('counts two separate quotes as two moments', () => {
    expect(groundedMomentCount(segs, withQuotes([0, 1000], [4500, 5500]))).toBe(2)
  })

  it('does not count a quote with no timestamp — it is anchored to nothing', () => {
    expect(groundedMomentCount(segs, withQuotes([null, null]))).toBe(0)
  })

  it('does not count a degenerate window, matching what the panel will show', () => {
    // Same rule as `quoteContains`. If the receipt counted a window the panel refuses to display,
    // the two would disagree about what "sourced" means.
    expect(groundedMomentCount(segs, withQuotes([0, 0]))).toBe(0)
    expect(groundedMomentCount(segs, withQuotes([3000, 2000]))).toBe(0)
  })

  it('does not count a quote timestamped past the end of the transcript', () => {
    expect(groundedMomentCount(segs, withQuotes([60000, 61000]))).toBe(0)
  })
})

describe('the linger boundary is exact, not approximate (#1978 follow-up)', () => {
  // Every existing linger test samples the middle of the window. Comfortably-inside assertions are
  // satisfied by an off-by-one, so the one number the rhythm actually depends on — when the panel
  // lets go — was the one number nothing pinned. The comparison is `t <= end + linger`, i.e. the
  // final millisecond is INCLUSIVE; that is a decision, and this is where it is recorded.
  //
  // `t` is SECONDS and `lingerMs` is milliseconds, and the linger must be passed explicitly — the
  // default is 0. PlayerView.vue:343 passes INSIGHT_LINGER_MS, so these call it the same way; a
  // test that omitted it would pin a boundary the app never uses.
  const insight = ins('a', 10_000, 12_000)
  const endS = (12_000 + INSIGHT_LINGER_MS) / 1000

  it('still shows the insight at the last millisecond of the linger', () => {
    expect(activeInsightIndex([insight], endS, INSIGHT_LINGER_MS)).toBe(0)
  })

  it('drops it one millisecond later', () => {
    expect(activeInsightIndex([insight], endS + 0.001, INSIGHT_LINGER_MS)).toBe(-1)
  })

  it('shows it at the exact moment the quote starts, with no lead-in', () => {
    // The absence of a symmetric lead-in was an explicit rejection, so it gets an explicit test.
    expect(activeInsightIndex([insight], 10, INSIGHT_LINGER_MS)).toBe(0)
    expect(activeInsightIndex([insight], 9.999, INSIGHT_LINGER_MS)).toBe(-1)
  })

  it('applies the same boundary to a quote with no end marker, via the 8s assumption', () => {
    const open = ins('b', 10_000)
    const openEndS = (10_000 + 8_000 + INSIGHT_LINGER_MS) / 1000
    expect(activeInsightIndex([open], openEndS, INSIGHT_LINGER_MS)).toBe(0)
    expect(activeInsightIndex([open], openEndS + 0.001, INSIGHT_LINGER_MS)).toBe(-1)
  })

  it('holds nothing at all when the linger is not requested', () => {
    // Guards the default: if `lingerMs` ever stopped defaulting to 0, every caller that does not
    // ask for a linger would silently start holding insights four seconds too long.
    expect(activeInsightIndex([insight], 12.001)).toBe(-1)
  })
})
