import { describe, expect, it } from 'vitest'
import type { ParsedArtifact } from '../types/artifact'
import {
  artifactRelPathsForResolvedRow,
  episodeIdsFromParsedArtifacts,
  sortResolvedArtifactsNewestFirst,
} from './resolvedArtifacts'

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
      {
        episode_id: 'a',
        publish_date: '2024-01-01',
        gi_relative_path: 'a.gi.json',
        kg_relative_path: null,
        bridge_relative_path: null,
      },
      {
        episode_id: 'b',
        publish_date: '2024-06-01',
        gi_relative_path: 'b.gi.json',
        kg_relative_path: null,
        bridge_relative_path: null,
      },
    ])
    expect(rows.map((r) => r.episode_id)).toEqual(['b', 'a'])
  })
})

describe('artifactRelPathsForResolvedRow', () => {
  it('returns gi, kg, bridge paths when present', () => {
    expect(
      artifactRelPathsForResolvedRow({
        episode_id: 'e',
        publish_date: null,
        gi_relative_path: 'm/a.gi.json',
        kg_relative_path: 'm/a.kg.json',
        bridge_relative_path: 'm/a.bridge.json',
      }),
    ).toEqual(['m/a.gi.json', 'm/a.kg.json', 'm/a.bridge.json'])
  })
})
