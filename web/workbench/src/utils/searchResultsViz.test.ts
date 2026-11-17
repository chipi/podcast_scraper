import { describe, expect, it } from 'vitest'
import type { SearchHit } from '../api/searchApi'
import {
  computeScoreStats,
  docTypeDistribution,
  episodeDistribution,
  episodeRowTooltip,
  feedDistribution,
  feedRowTooltip,
  insightDominantDocType,
  insightTimeline,
  insightTopTerm,
  publishMonthTimeline,
  scoreBarsForHits,
  topTermsFromHits,
} from './searchResultsViz'

function hit(p: Partial<SearchHit> & { doc_id: string; score: number; text: string }): SearchHit {
  return {
    doc_id: p.doc_id,
    score: p.score,
    text: p.text,
    metadata: p.metadata ?? {},
    supporting_quotes: p.supporting_quotes,
  }
}

describe('searchResultsViz', () => {
  it('docTypeDistribution groups and labels', () => {
    const hits = [
      hit({
        doc_id: 'a',
        score: 1,
        text: 'x',
        metadata: { doc_type: 'insight' },
      }),
      hit({
        doc_id: 'b',
        score: 0.9,
        text: 'y',
        metadata: { doc_type: 'transcript' },
      }),
      hit({
        doc_id: 'c',
        score: 0.8,
        text: 'z',
        metadata: { doc_type: 'insight' },
      }),
    ]
    const rows = docTypeDistribution(hits)
    expect(rows).toHaveLength(2)
    expect(rows[0].key).toBe('insight')
    expect(rows[0].count).toBe(2)
    expect(rows[0].label).toBe('Insights')
    expect(rows[0].pct).toBeCloseTo((2 / 3) * 100, 5)
  })

  it('episodeDistribution respects limit and reports tail', () => {
    const hits: SearchHit[] = []
    for (let i = 0; i < 5; i += 1) {
      hits.push(
        hit({
          doc_id: `e${i}`,
          score: 1,
          text: 't',
          metadata: { episode_id: `ep-${i}` },
        }),
      )
    }
    const out = episodeDistribution(hits, 3)
    expect(out.rows).toHaveLength(3)
    expect(out.tailDistinct).toBe(2)
    expect(out.tailHitCount).toBe(2)
  })

  it('feedDistribution truncates long feed id when no feed_title', () => {
    const longId = `x${'a'.repeat(50)}`
    const out = feedDistribution([
      hit({
        doc_id: '1',
        score: 1,
        text: 't',
        metadata: { feed_id: longId },
      }),
    ])
    expect(out.rows[0].label.length).toBeLessThan(longId.length)
    expect(out.rows[0].key).toBe(longId)
    expect(out.rows[0].fullLabel).toBe(longId)
    expect(out.tailDistinct).toBe(0)
  })

  it('episodeDistribution prefers episode_title and feedDistribution prefers feed_title', () => {
    const hits = [
      hit({
        doc_id: '1',
        score: 1,
        text: '',
        metadata: {
          episode_id: 'e1',
          episode_title: 'My Episode Title Here',
          feed_id: 'f1',
          feed_title: 'My Feed Show Name',
        },
      }),
      hit({ doc_id: '2', score: 1, text: '', metadata: { episode_id: 'e1', feed_id: 'f1' } }),
    ]
    const ep = episodeDistribution(hits, 5)
    expect(ep.rows[0].fullLabel).toBe('My Episode Title Here')
    expect(ep.rows[0].label.length).toBeLessThanOrEqual(53)
    expect(episodeRowTooltip(ep.rows[0])).toContain('e1')

    const fd = feedDistribution(hits, 5)
    expect(fd.rows[0].fullLabel).toBe('My Feed Show Name')
    expect(feedRowTooltip(fd.rows[0])).toContain('f1')
  })

  it('computeScoreStats and scoreBarsForHits use score/max encoding', () => {
    const hits = [
      hit({ doc_id: 'a', score: 0.5, text: '', metadata: { doc_type: 'quote' } }),
      hit({ doc_id: 'b', score: 1, text: '', metadata: { doc_type: 'insight' } }),
    ]
    const st = computeScoreStats(hits)
    expect(st?.min).toBe(0.5)
    expect(st?.max).toBe(1)
    expect(st?.mean).toBe(0.75)
    const bars = scoreBarsForHits(hits)
    expect(bars[0].widthPct).toBe(50)
    expect(bars[1].widthPct).toBe(100)
  })

  it('scoreBarsForHits uses full width when all scores equal', () => {
    const hits = [
      hit({ doc_id: 'a', score: 0.7, text: '', metadata: {} }),
      hit({ doc_id: 'b', score: 0.7, text: '', metadata: {} }),
    ]
    const bars = scoreBarsForHits(hits)
    expect(bars.every((b) => b.widthPct === 100)).toBe(true)
  })

  it('publishMonthTimeline buckets ISO publish_date', () => {
    const hits = [
      hit({
        doc_id: '1',
        score: 1,
        text: '',
        metadata: { publish_date: '2024-03-15T12:00:00Z' },
      }),
      hit({
        doc_id: '2',
        score: 1,
        text: '',
        metadata: { publish_date: '2024-03-20' },
      }),
      hit({ doc_id: '3', score: 1, text: '', metadata: { publish_date: '' } }),
    ]
    const tl = publishMonthTimeline(hits)
    expect(tl.buckets).toEqual([{ label: '2024-03', count: 2 }])
    expect(tl.unparsed).toBe(1)
  })

  it('topTermsFromHits filters stopwords and sorts', () => {
    const hits = [
      hit({
        doc_id: '1',
        score: 1,
        text: 'Acme robotics excels at robotics and precision engineering.',
        metadata: {},
      }),
    ]
    const terms = topTermsFromHits(hits, 10)
    expect(terms.some((t) => t.term === 'robotics')).toBe(true)
    expect(terms.some((t) => t.term === 'that')).toBe(false)
  })

  it('insight helpers', () => {
    const rows = docTypeDistribution([
      hit({ doc_id: 'a', score: 1, text: '', metadata: { doc_type: 'insight' } }),
      hit({ doc_id: 'b', score: 1, text: '', metadata: { doc_type: 'insight' } }),
    ])
    expect(insightDominantDocType(rows, 2)).toMatch(/Insights dominates/)

    const tl = {
      buckets: [{ label: '2024-01', count: 3 }],
      unparsed: 1,
    }
    expect(insightTimeline(tl, 4)).toMatch(/Peak month 2024-01/)
    expect(insightTimeline(tl, 4)).toMatch(/not dated/)

    expect(insightTopTerm([{ term: 'alpha', count: 5 }])).toMatch(/alpha/)
  })
})
