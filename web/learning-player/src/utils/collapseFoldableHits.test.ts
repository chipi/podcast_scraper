import { describe, expect, it } from 'vitest'
import {
  collapseFoldableHits,
  isFoldedCluster,
  type CollapsedRow,
} from './collapseFoldableHits'
import type { SearchHit } from '../services/types'

function h(
  overrides: Partial<SearchHit> & { doc_id: string; doc_type: string; score?: number },
): SearchHit {
  return {
    doc_id: overrides.doc_id,
    score: overrides.score ?? 0.5,
    text: overrides.text ?? '',
    source_tier: overrides.source_tier ?? 'segment',
    metadata: {
      ...(overrides.metadata as Record<string, unknown>),
      doc_type: overrides.doc_type,
    },
    lifted: overrides.lifted ?? null,
  }
}

describe('collapseFoldableHits', () => {
  it('is a no-op on non-foldable rows (insight / kg_topic / kg_entity / quote)', () => {
    const rows = collapseFoldableHits([
      h({ doc_id: 'i1', doc_type: 'insight' }),
      h({ doc_id: 't1', doc_type: 'kg_topic' }),
      h({ doc_id: 'e1', doc_type: 'kg_entity' }),
    ])
    expect(rows).toHaveLength(3)
    expect(rows.every((r) => !isFoldedCluster(r))).toBe(true)
  })

  it('folds N transcript hits into one cluster at the slot of the top-scoring member', () => {
    const rows = collapseFoldableHits([
      h({ doc_id: 't1', doc_type: 'transcript', score: 0.9, text: 'a' }),
      h({ doc_id: 'i1', doc_type: 'insight', score: 0.85 }),
      h({ doc_id: 't2', doc_type: 'transcript', score: 0.7, text: 'b' }),
      h({ doc_id: 't3', doc_type: 'transcript', score: 0.6, text: 'c' }),
    ])
    expect(rows).toHaveLength(2)
    const cluster = rows[0] as Extract<CollapsedRow, { __kind: 'folded_cluster' }>
    expect(isFoldedCluster(cluster)).toBe(true)
    expect(cluster.foldedKind).toBe('transcript')
    expect(cluster.members.map((m) => m.doc_id)).toEqual(['t1', 't2', 't3'])
    expect(cluster.topScore).toBe(0.9)
    expect(rows[1].doc_id).toBe('i1')
  })

  it('keeps title / description / summary as separate clusters (not one merged bucket)', () => {
    const rows = collapseFoldableHits([
      h({ doc_id: 'a', doc_type: 'episode_title', score: 0.9 }),
      h({ doc_id: 'b', doc_type: 'episode_description', score: 0.8 }),
      h({ doc_id: 'c', doc_type: 'summary_short', score: 0.7 }),
      h({ doc_id: 'd', doc_type: 'transcript', score: 0.6 }),
    ])
    expect(rows).toHaveLength(4)
    const kinds = rows
      .filter(isFoldedCluster)
      .map((c) => (c as Extract<CollapsedRow, { __kind: 'folded_cluster' }>).foldedKind)
    expect(kinds).toEqual(['episode_title', 'episode_description', 'summary_short', 'transcript'])
  })

  it('does NOT fold a transcript hit that carries a lifted insight — it stays a card', () => {
    const rows = collapseFoldableHits([
      h({
        doc_id: 't1',
        doc_type: 'transcript',
        score: 0.9,
        lifted: { insight: { id: 'i1', text: 'X' } },
      }),
      h({ doc_id: 't2', doc_type: 'transcript', score: 0.7 }),
    ])
    expect(rows).toHaveLength(2)
    expect(isFoldedCluster(rows[0])).toBe(false)
    expect(isFoldedCluster(rows[1])).toBe(true)
  })

  it('a single foldable hit still becomes a 1-member cluster (uniform render path)', () => {
    const rows = collapseFoldableHits([h({ doc_id: 't1', doc_type: 'transcript', score: 0.5 })])
    expect(rows).toHaveLength(1)
    expect(isFoldedCluster(rows[0])).toBe(true)
  })
})
