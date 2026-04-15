import { describe, expect, it } from 'vitest'
import type { ParsedArtifact } from '../types/artifact'
import type { TopicClustersDocument } from '../api/corpusTopicClustersApi'
import {
  artifactRelPathsForResolvedRow,
  clusterSiblingEpisodeIdCandidates,
  episodeIdsForClusterMember,
  episodeIdsFromParsedArtifacts,
  sortResolvedArtifactsNewestFirst,
} from './clusterSiblingMerge'

describe('episodeIdsForClusterMember', () => {
  it('returns episode_ids for the matching member topic_id', () => {
    const cluster = {
      graph_compound_parent_id: 'tc:x',
      members: [
        { topic_id: 'topic:a', episode_ids: ['e1', 'e2'] },
        { topic_id: 'topic:b', episode_ids: ['e3'] },
      ],
    }
    expect(episodeIdsForClusterMember(cluster, 'topic:b')).toEqual(['e3'])
    expect(episodeIdsForClusterMember(cluster, 'topic:missing')).toEqual([])
    expect(episodeIdsForClusterMember(null, 'topic:a')).toEqual([])
  })
})

describe('clusterSiblingEpisodeIdCandidates', () => {
  it('returns empty when doc has no clusters', () => {
    expect(clusterSiblingEpisodeIdCandidates(null, new Set())).toEqual({
      candidateIds: [],
      mTotal: 0,
    })
  })

  it('unions episode ids for clusters that share an episode with loaded set', () => {
    const doc: TopicClustersDocument = {
      clusters: [
        {
          graph_compound_parent_id: 'tc:a',
          canonical_label: 'A',
          members: [
            { topic_id: 'topic:x', episode_ids: ['e1', 'e2'] },
            { topic_id: 'topic:y', episode_ids: ['e2', 'e3'] },
          ],
        },
      ],
    }
    const loaded = new Set(['e2'])
    const { candidateIds, mTotal } = clusterSiblingEpisodeIdCandidates(doc, loaded)
    expect(mTotal).toBe(3)
    expect(new Set(candidateIds)).toEqual(new Set(['e1', 'e3']))
  })
})

describe('episodeIdsFromParsedArtifacts', () => {
  it('collects episode_id only from gi artifacts', () => {
    const gi: ParsedArtifact = {
      name: 'a',
      kind: 'gi',
      episodeId: 'e1',
      nodes: 0,
      edges: 0,
      nodeTypes: {},
      data: { episode_id: 'ep-a', nodes: [], edges: [] },
    }
    const kg: ParsedArtifact = {
      name: 'b',
      kind: 'kg',
      episodeId: null,
      nodes: 0,
      edges: 0,
      nodeTypes: {},
      data: { episode_id: 'should-ignore', nodes: [], edges: [] },
    }
    expect(episodeIdsFromParsedArtifacts([gi, kg])).toEqual(new Set(['ep-a']))
  })
})

describe('sortResolvedArtifactsNewestFirst', () => {
  it('sorts by publish_date descending', () => {
    const rows = sortResolvedArtifactsNewestFirst([
      { episode_id: 'a', publish_date: '2024-01-01', gi_relative_path: 'a.gi.json' },
      { episode_id: 'b', publish_date: '2024-06-01', gi_relative_path: 'b.gi.json' },
    ])
    expect(rows.map((r) => r.episode_id)).toEqual(['b', 'a'])
  })
})

describe('artifactRelPathsForResolvedRow', () => {
  it('returns gi, kg, bridge paths when present', () => {
    expect(
      artifactRelPathsForResolvedRow({
        episode_id: 'e',
        gi_relative_path: 'm/a.gi.json',
        kg_relative_path: 'm/a.kg.json',
        bridge_relative_path: 'm/a.bridge.json',
      }),
    ).toEqual(['m/a.gi.json', 'm/a.kg.json', 'm/a.bridge.json'])
  })
})
