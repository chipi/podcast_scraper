import { describe, expect, it } from 'vitest'
import type { SearchHit, Segment } from '../services/types'
import { groupBriefResults, markTerms, searchTerms, snippetAround } from './briefSearch'

const SEGS: Segment[] = [
  { id: 's0', start: 0, end: 4, text: 'Welcome back to the show everyone, good to have you.', speaker: null },
  { id: 's1', start: 4, end: 9, text: 'Today we talk about sizing positions so the', speaker: null },
  { id: 's2', start: 9, end: 14, text: 'worst week is survivable for a long time.', speaker: null },
  { id: 's3', start: 14, end: 20, text: 'Diversification is the only free lunch, people say, and they repeat it.', speaker: null },
]

function hit(doc_id: string, doc_type: string, text: string, md: Record<string, unknown> = {}): SearchHit {
  return { doc_id, score: 1, text, metadata: { doc_type, ...md } } as unknown as SearchHit
}

describe('searchTerms', () => {
  it('keeps the remembered words, drops filler when there are others', () => {
    expect(searchTerms('the worst week')).toEqual(['worst', 'week'])
    expect(searchTerms('to be')).toEqual(['to', 'be'])
  })
})

describe('groupBriefResults (operator 2026-10-10: C and D)', () => {
  it('two groups: transcript pieces and insights; the episode summary/description/title are left out', () => {
    const g = groupBriefResults(
      [
        hit('i1', 'insight', 'Insight one'),
        hit('sum', 'summary', 'The summary'),
        hit('desc', 'episode_description', 'Desc'),
        hit('q1', 'quote', 'worst week is survivable', { timestamp_start_ms: 9_000, timestamp_end_ms: 14_000 }),
      ],
      SEGS,
      'worst week',
    )
    expect(g.insights.map((h) => h.doc_id)).toEqual(['i1'])
    expect(g.transcript).toHaveLength(1)
  })

  it('a transcript result becomes the segment with the words in it, timed from that segment', () => {
    const chunk = hit('c1', 'transcript', 'a long passage …', { timestamp_start_ms: 0, timestamp_end_ms: 20_000 })
    const [p] = groupBriefResults([chunk], SEGS, 'free lunch').transcript
    expect(p.start).toBe(14)
    expect(p.text).toContain('Diversification is the only free lunch')
    expect(p.segments.map((s) => s.id)).toEqual(['s3'])
  })

  it('a very short segment brings the next one, so the piece reads as a sentence', () => {
    const [p] = groupBriefResults(
      [hit('c1', 'transcript', 'x', { timestamp_start_ms: 4_000, timestamp_end_ms: 9_000 })],
      SEGS,
      'sizing',
    ).transcript
    expect(p.segments.map((s) => s.id)).toEqual(['s1', 's2'])
  })

  it('verbatim passages come first, and the same moment is listed once', () => {
    const g = groupBriefResults(
      [
        hit('c1', 'transcript', 'x', { timestamp_start_ms: 14_000, timestamp_end_ms: 20_000 }),
        hit('e1', 'transcript', 'worst week', { match: 'phrase', timestamp_start_ms: 9_000, timestamp_end_ms: 14_000 }),
        hit('q1', 'quote', 'worst week', { timestamp_start_ms: 9_000, timestamp_end_ms: 14_000 }),
      ],
      SEGS,
      'worst week',
    )
    expect(g.transcript.map((p) => p.start)).toEqual([9, 14])
    expect(g.transcript[0].match).toBe('phrase')
  })

  it('without a timed transcript it cuts the words out of the result text, untimed', () => {
    const long = Array.from({ length: 80 }, (_, i) => (i === 40 ? 'drawdown' : `w${i}`)).join(' ')
    const [p] = groupBriefResults([hit('c1', 'transcript', long)], [], 'drawdown').transcript
    expect(p.start).toBeNull()
    expect(p.text.split(' ').length).toBeLessThan(40)
    expect(p.text).toContain('drawdown')
  })
})

describe('markTerms / snippetAround', () => {
  it('marks whole words in any case, and nothing inside other words', () => {
    expect(markTerms('Worst week; worsted wool', ['worst', 'week'])).toEqual([
      { text: 'Worst', mark: true },
      { text: ' ', mark: false },
      { text: 'week', mark: true },
      { text: '; worsted wool', mark: false },
    ])
  })
  it('a short text is kept whole', () => {
    expect(snippetAround('only a few words here', ['few'])).toBe('only a few words here')
  })
})
