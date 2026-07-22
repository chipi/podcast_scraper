import { describe, expect, it } from 'vitest'
import {
  collapseTranscriptHitsByEpisode,
  isTranscriptClusterHit,
  type TranscriptClusterHit,
} from './collapseTranscriptHitsByEpisode'
import type { SearchHit } from '../api/searchApi'

function h(
  overrides: Partial<SearchHit> & { metadata?: Record<string, unknown> } = {},
): SearchHit {
  return {
    doc_id: overrides.doc_id ?? 'd:x',
    score: overrides.score ?? 0.5,
    text: overrides.text ?? '',
    metadata: overrides.metadata ?? {},
    source_tier: overrides.source_tier ?? 'aux',
  } as SearchHit
}

describe('collapseTranscriptHitsByEpisode', () => {
  it('passes non-foldable hits through unchanged', () => {
    const hits = [
      h({ doc_id: 'i:1', source_tier: 'insight', metadata: { doc_type: 'insight' } }),
      h({ doc_id: 'kt:1', source_tier: 'aux', metadata: { doc_type: 'kg_topic' } }),
      h({ doc_id: 's:1', source_tier: 'aux', metadata: { doc_type: 'summary', episode_id: 'ep-a' } }),
    ]
    const rows = collapseTranscriptHitsByEpisode(hits)
    expect(rows).toEqual(hits)
    expect(rows.some(isTranscriptClusterHit)).toBe(false)
  })

  it('folds episode_title / episode_description / summary_short into the same episode cluster as transcript chunks', () => {
    const t = h({
      doc_id: 't:1',
      score: 0.9,
      source_tier: 'segment',
      metadata: {
        doc_type: 'transcript',
        episode_id: 'ep-a',
        episode_title: 'A',
        matched_field: 'transcript',
      },
    })
    const titleHit = h({
      doc_id: 'et:1',
      score: 0.85,
      source_tier: 'aux',
      metadata: {
        doc_type: 'episode_title',
        matched_field: 'title',
        episode_id: 'ep-a',
      },
    })
    const descHit = h({
      doc_id: 'ed:1',
      score: 0.7,
      source_tier: 'aux',
      metadata: {
        doc_type: 'episode_description',
        matched_field: 'description',
        episode_id: 'ep-a',
      },
    })
    const summaryHit = h({
      doc_id: 'ss:1',
      score: 0.6,
      source_tier: 'aux',
      metadata: {
        doc_type: 'summary_short',
        matched_field: 'summary',
        episode_id: 'ep-a',
      },
    })
    const rows = collapseTranscriptHitsByEpisode([t, titleHit, descHit, summaryHit])
    expect(rows).toHaveLength(1)
    const cluster = rows[0] as TranscriptClusterHit
    expect(cluster.members).toHaveLength(4)
    // Members preserved in input order (score-sorted).
    expect(cluster.members.map((m) => m.doc_id)).toEqual(['t:1', 'et:1', 'ed:1', 'ss:1'])
    expect(cluster.topScore).toBe(0.9)
  })

  it('collapses multiple transcript hits with the same episode_id into one cluster', () => {
    const t1 = h({
      doc_id: 't:1',
      score: 0.92,
      text: 'first chunk',
      source_tier: 'segment',
      metadata: {
        doc_type: 'transcript',
        episode_id: 'ep-a',
        episode_title: 'Ep A',
        feed_title: 'Show X',
        publish_date: '2026-04-15',
        source_metadata_relative_path: 'meta/a.metadata.json',
      },
    })
    const t2 = h({
      doc_id: 't:2',
      score: 0.85,
      text: 'second chunk',
      source_tier: 'segment',
      metadata: {
        doc_type: 'transcript',
        episode_id: 'ep-a',
      },
    })
    const t3 = h({
      doc_id: 't:3',
      score: 0.7,
      text: 'third chunk',
      source_tier: 'segment',
      metadata: {
        doc_type: 'transcript',
        episode_id: 'ep-a',
      },
    })
    const rows = collapseTranscriptHitsByEpisode([t1, t2, t3])
    expect(rows).toHaveLength(1)
    const cluster = rows[0] as TranscriptClusterHit
    expect(isTranscriptClusterHit(cluster)).toBe(true)
    expect(cluster.episodeId).toBe('ep-a')
    expect(cluster.episodeTitle).toBe('Ep A')
    expect(cluster.feedTitle).toBe('Show X')
    expect(cluster.publishDate).toBe('2026-04-15')
    expect(cluster.metadataRelativePath).toBe('meta/a.metadata.json')
    expect(cluster.members).toHaveLength(3)
    expect(cluster.topScore).toBe(0.92)
  })

  it('keeps transcript hits from different episodes as separate clusters', () => {
    const a = h({
      doc_id: 'a:1',
      score: 0.9,
      source_tier: 'segment',
      metadata: { doc_type: 'transcript', episode_id: 'ep-a', episode_title: 'A' },
    })
    const b = h({
      doc_id: 'b:1',
      score: 0.8,
      source_tier: 'segment',
      metadata: { doc_type: 'transcript', episode_id: 'ep-b', episode_title: 'B' },
    })
    const rows = collapseTranscriptHitsByEpisode([a, b])
    expect(rows).toHaveLength(2)
    const clA = rows[0] as TranscriptClusterHit
    const clB = rows[1] as TranscriptClusterHit
    expect(clA.episodeId).toBe('ep-a')
    expect(clA.members).toHaveLength(1)
    expect(clB.episodeId).toBe('ep-b')
    expect(clB.members).toHaveLength(1)
  })

  it('preserves position of the FIRST hit for each episode (score-sorted list stays score-sorted)', () => {
    const insight = h({ doc_id: 'i:1', score: 0.95, source_tier: 'insight' })
    const t1 = h({
      doc_id: 't:1',
      score: 0.9,
      source_tier: 'segment',
      metadata: { doc_type: 'transcript', episode_id: 'ep-a' },
    })
    const kg = h({
      doc_id: 'kt:1',
      score: 0.85,
      source_tier: 'aux',
      metadata: { doc_type: 'kg_topic' },
    })
    // Second chunk from the same episode as t1 — should FOLD IN, not push kg down.
    const t2 = h({
      doc_id: 't:2',
      score: 0.6,
      source_tier: 'segment',
      metadata: { doc_type: 'transcript', episode_id: 'ep-a' },
    })
    const rows = collapseTranscriptHitsByEpisode([insight, t1, kg, t2])
    // insight → transcript-cluster(ep-a, 2 members) → kg. 3 rows, kg preserved.
    expect(rows).toHaveLength(3)
    expect(rows[0]).toBe(insight)
    const cluster = rows[1] as TranscriptClusterHit
    expect(isTranscriptClusterHit(cluster)).toBe(true)
    expect(cluster.members).toHaveLength(2)
    expect(cluster.members[0]).toBe(t1)
    expect(cluster.members[1]).toBe(t2)
    expect(rows[2]).toBe(kg)
  })

  it('passes transcript hits WITHOUT episode_id through as plain rows', () => {
    const orphan = h({
      doc_id: 't:orphan',
      source_tier: 'segment',
      metadata: { doc_type: 'transcript' },
    })
    const rows = collapseTranscriptHitsByEpisode([orphan])
    expect(rows).toEqual([orphan])
  })

  it('a single transcript hit still becomes a cluster (so the same card renders everywhere)', () => {
    const t = h({
      doc_id: 't:1',
      score: 0.5,
      source_tier: 'segment',
      metadata: { doc_type: 'transcript', episode_id: 'ep-a', episode_title: 'A' },
    })
    const rows = collapseTranscriptHitsByEpisode([t])
    expect(rows).toHaveLength(1)
    expect(isTranscriptClusterHit(rows[0])).toBe(true)
    expect((rows[0] as TranscriptClusterHit).members).toEqual([t])
  })
})
