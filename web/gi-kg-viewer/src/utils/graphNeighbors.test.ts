import { describe, expect, it } from 'vitest'
import type { ParsedArtifact } from '../types/artifact'
import {
  graphNeighborsForMemberGraphIds,
  graphNeighborsForNode,
  mergeNeighborRowsByNeighborId,
} from './graphNeighbors'

describe('mergeNeighborRowsByNeighborId', () => {
  it('dedupes rows with the same neighbor id', () => {
    const merged = mergeNeighborRowsByNeighborId([
      {
        id: 'ep:1',
        label: 'E',
        type: 'Episode',
        visualType: 'Episode',
        edgeType: 'mentions',
        direction: 'out',
        viaMemberTopicIds: ['k:topic:a'],
      },
      {
        id: 'ep:1',
        label: 'E',
        type: 'Episode',
        visualType: 'Episode',
        edgeType: 'about',
        direction: 'out',
        viaMemberTopicIds: ['k:topic:b'],
      },
    ])
    expect(merged).toHaveLength(1)
    expect(merged[0]!.edgeType).toContain('mentions')
    expect(merged[0]!.edgeType).toContain('about')
    expect(merged[0]!.viaMemberTopicIds?.sort()).toEqual(['k:topic:a', 'k:topic:b'])
  })
})

describe('graphNeighborsForMemberGraphIds', () => {
  it('merges neighbors from multiple topic nodes', () => {
    const art: ParsedArtifact = {
      name: 'x',
      kind: 'kg',
      episodeId: null,
      nodes: 4,
      edges: 2,
      nodeTypes: {},
      data: {
        nodes: [
          { id: 'k:topic:a', type: 'Topic', properties: { label: 'A' } },
          { id: 'k:topic:b', type: 'Topic', properties: { label: 'B' } },
          { id: 'ep:1', type: 'Episode', properties: { title: 'One' } },
          { id: 'ep:2', type: 'Episode', properties: { title: 'Two' } },
        ],
        edges: [
          { from: 'k:topic:a', to: 'ep:1', type: 'mentions' },
          { from: 'k:topic:b', to: 'ep:2', type: 'mentions' },
        ],
      },
    }
    const rows = graphNeighborsForMemberGraphIds(art, ['k:topic:a', 'k:topic:b'])
    expect(rows.map((r) => r.id).sort()).toEqual(['ep:1', 'ep:2'])
  })

  it('returns empty when artifact null', () => {
    expect(graphNeighborsForMemberGraphIds(null, ['k:topic:a'])).toEqual([])
    expect(graphNeighborsForNode(null, 'x')).toEqual([])
  })
})
