import { describe, expect, it } from 'vitest'
import type { TopicClustersDocument } from '../api/corpusTopicClustersApi'
import {
  GRAPH_DEFAULT_EPISODE_CAP,
  GRAPH_SCORE_TOPIC_CLUSTER_BONUS,
  episodeIdsInTopicClustersForGraphScoring,
  selectRelPathsForGraphLoad,
  stemMatchesTopicClusterEpisodeId,
} from './graphEpisodeSelection'

describe('selectRelPathsForGraphLoad', () => {
  const rows = [
    { relative_path: 'm/a.gi.json', kind: 'gi', publish_date: '2024-01-10' },
    { relative_path: 'm/a.kg.json', kind: 'kg', publish_date: '2024-01-10' },
    { relative_path: 'm/a.bridge.json', kind: 'bridge', publish_date: '2024-01-10' },
    { relative_path: 'm/b.gi.json', kind: 'gi', publish_date: '2024-06-20' },
    { relative_path: 'm/b.kg.json', kind: 'kg', publish_date: '2024-06-20' },
  ]

  it('returns all paths for two episodes when cap is high', () => {
    const r = selectRelPathsForGraphLoad(rows, '', 10)
    expect(r.episodeCount).toBe(2)
    expect(r.wasCapped).toBe(false)
    expect(r.selectedRelPaths.length).toBe(5)
  })

  it('caps at N episodes for all-time lens (recency-ordered when no cluster signal)', () => {
    const many = [
      ...rows,
      { relative_path: 'm/c.gi.json', kind: 'gi', publish_date: '2024-03-01' },
      { relative_path: 'm/c.kg.json', kind: 'kg', publish_date: '2024-03-01' },
    ]
    const r = selectRelPathsForGraphLoad(many, '', 2)
    expect(r.episodeCount).toBe(2)
    expect(r.wasCapped).toBe(true)
    expect(r.selectedRelPaths.some((p) => p.includes('/b.'))).toBe(true)
    expect(r.selectedRelPaths.some((p) => p.includes('/c.'))).toBe(true)
    expect(r.selectedRelPaths.some((p) => p.includes('/a.'))).toBe(false)
  })

  it('filters by sinceYmd', () => {
    const r = selectRelPathsForGraphLoad(rows, '2024-06-01', 10)
    expect(r.episodeCount).toBe(1)
    expect(r.selectedRelPaths.every((p) => p.includes('/b.'))).toBe(true)
  })

  it('prefers topic-cluster episode over slightly newer non-cluster when scores tie up', () => {
    const pool = [
      { relative_path: 'm/new.gi.json', kind: 'gi', publish_date: '2024-06-25' },
      { relative_path: 'm/new.kg.json', kind: 'kg', publish_date: '2024-06-25' },
      { relative_path: 'm/mid.gi.json', kind: 'gi', publish_date: '2024-06-10' },
      { relative_path: 'm/mid.kg.json', kind: 'kg', publish_date: '2024-06-10' },
      { relative_path: 'm/old.gi.json', kind: 'gi', publish_date: '2024-01-05' },
      { relative_path: 'm/old.kg.json', kind: 'kg', publish_date: '2024-01-05' },
    ]
    const doc: TopicClustersDocument = {
      clusters: [
        {
          graph_compound_parent_id: 'tc:t',
          members: [{ topic_id: 'topic:x', episode_ids: ['mid'] }],
        },
      ],
    }
    const r = selectRelPathsForGraphLoad(pool, '', 2, doc)
    expect(r.episodeCount).toBe(2)
    expect(r.selectedRelPaths.some((p) => p.includes('/mid.'))).toBe(true)
    expect(r.selectedRelPaths.some((p) => p.includes('/new.'))).toBe(true)
    expect(r.selectedRelPaths.some((p) => p.includes('/old.'))).toBe(false)
  })

  it('uses default cap constant', () => {
    expect(GRAPH_DEFAULT_EPISODE_CAP).toBe(15)
  })
})

describe('topic cluster scoring helpers', () => {
  it('collects episode ids from cluster members', () => {
    const doc: TopicClustersDocument = {
      clusters: [
        {
          members: [
            { topic_id: 'a', episode_ids: ['e1', 'e2'] },
            { topic_id: 'b', episode_ids: ['e3'] },
          ],
        },
      ],
    }
    expect(episodeIdsInTopicClustersForGraphScoring(doc)).toEqual(new Set(['e1', 'e2', 'e3']))
  })

  it('matches stem basename to cluster episode id', () => {
    const ids = new Set(['ep42'])
    expect(stemMatchesTopicClusterEpisodeId('feeds/x/ep42', ids)).toBe(true)
    expect(stemMatchesTopicClusterEpisodeId('ep42', ids)).toBe(true)
    expect(stemMatchesTopicClusterEpisodeId('feeds/x/other', ids)).toBe(false)
  })

  it('adds cluster bonus so clustered episode can rank above slightly newer non-cluster', () => {
    const pool = [
      { relative_path: 'pods/z.gi.json', kind: 'gi', publish_date: '2024-06-30' },
      { relative_path: 'pods/z.kg.json', kind: 'kg', publish_date: '2024-06-30' },
      { relative_path: 'pods/y.gi.json', kind: 'gi', publish_date: '2024-06-15' },
      { relative_path: 'pods/y.kg.json', kind: 'kg', publish_date: '2024-06-15' },
      { relative_path: 'pods/x.gi.json', kind: 'gi', publish_date: '2024-06-10' },
      { relative_path: 'pods/x.kg.json', kind: 'kg', publish_date: '2024-06-10' },
    ]
    const doc: TopicClustersDocument = {
      clusters: [
        {
          members: [{ topic_id: 'topic:t', episode_ids: ['x'] }],
        },
      ],
    }
    const r = selectRelPathsForGraphLoad(pool, '', 2, doc)
    expect(r.selectedRelPaths.some((p) => p.includes('/z.'))).toBe(true)
    expect(r.selectedRelPaths.some((p) => p.includes('/x.'))).toBe(true)
    expect(r.selectedRelPaths.some((p) => p.includes('/y.'))).toBe(false)
  })
})

describe('GRAPH_SCORE_TOPIC_CLUSTER_BONUS', () => {
  it('is the documented default cluster bonus', () => {
    expect(GRAPH_SCORE_TOPIC_CLUSTER_BONUS).toBe(0.4)
  })
})
