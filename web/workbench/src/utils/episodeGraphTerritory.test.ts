import { describe, expect, it } from 'vitest'
import type { ParsedArtifact } from '../types/artifact'
import { collectGraphNodeIdsForEpisodeTerritory } from './episodeGraphTerritory'

function art(
  nodes: unknown[],
  edges: unknown[] = [],
  episodeId: string | null = 'ep1',
): ParsedArtifact {
  return {
    name: 't.gi.json',
    kind: 'gi',
    episodeId,
    nodes: nodes.length,
    edges: edges.length,
    nodeTypes: {},
    data: { nodes: nodes as never[], edges: edges as never[] },
    sourceCorpusRelPath: 'metadata/x.gi.json',
    sourceCorpusRelPathByEpisodeId: { ep1: 'metadata/x.gi.json' },
  }
}

describe('collectGraphNodeIdsForEpisodeTerritory', () => {
  it('collects Insight by episode_id and skips Episode container', () => {
    const want = 'metadata/x.metadata.json'
    const a = art([
      {
        id: 'episode:ep1',
        type: 'Episode',
        properties: {
          episode_id: 'ep1',
          metadata_relative_path: want,
        },
      },
      { id: 'insight:a', type: 'Insight', properties: { episode_id: 'ep1' } },
      { id: 'topic:t', type: 'Topic', properties: { episode_id: 'other' } },
    ])
    const ids = collectGraphNodeIdsForEpisodeTerritory(a, want)
    expect(ids).toContain('insight:a')
    expect(ids).not.toContain('episode:ep1')
    expect(ids).not.toContain('topic:t')
  })

  it('matches metadata path on non-Episode nodes', () => {
    const want = 'metadata/z.metadata.json'
    const a = art(
      [
        {
          id: 'episode:ep9',
          type: 'Episode',
          properties: { episode_id: 'ep9', metadata_relative_path: want },
        },
        {
          id: 'quote:q',
          type: 'Quote',
          properties: { source_metadata_relative_path: want },
        },
      ],
      [],
      'ep9',
    )
    const ids = collectGraphNodeIdsForEpisodeTerritory(a, want)
    expect(ids).toEqual(['quote:q'])
  })
})
