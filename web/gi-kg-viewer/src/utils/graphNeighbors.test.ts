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

describe('graphNeighborsForNode — RFC-097 v3.0 typed MENTIONS family', () => {
  /**
   * The neighbor walk preserves the raw artifact edge type on the
   * resulting row. The details panel (``GraphConnectionsSection.vue``)
   * shows ``row.edgeType`` directly to the user, so typed variants
   * surface as their precise type ("MENTIONS_PERSON" not "MENTIONS").
   */

  function artWith(edges: { from: string; to: string; type: string }[]): ParsedArtifact {
    return {
      name: 'x',
      kind: 'gi',
      episodeId: null,
      nodes: 4,
      edges: edges.length,
      nodeTypes: {},
      data: {
        nodes: [
          { id: 'insight:1', type: 'Insight', properties: { text: 'i1' } },
          { id: 'person:alice', type: 'Person', properties: { name: 'Alice' } },
          { id: 'org:acme', type: 'Organization', properties: { name: 'Acme' } },
          { id: 'topic:ai', type: 'Topic', properties: { label: 'AI' } },
        ],
        edges,
      },
    }
  }

  it('emits MENTIONS_PERSON edgeType verbatim on the neighbor row', () => {
    const art = artWith([
      { from: 'insight:1', to: 'person:alice', type: 'MENTIONS_PERSON' },
    ])
    const rows = graphNeighborsForNode(art, 'insight:1')
    expect(rows).toHaveLength(1)
    expect(rows[0]!.id).toBe('person:alice')
    expect(rows[0]!.edgeType).toBe('MENTIONS_PERSON')
  })

  it('emits MENTIONS_ORG edgeType verbatim on the neighbor row', () => {
    const art = artWith([{ from: 'insight:1', to: 'org:acme', type: 'MENTIONS_ORG' }])
    const rows = graphNeighborsForNode(art, 'insight:1')
    expect(rows).toHaveLength(1)
    expect(rows[0]!.id).toBe('org:acme')
    expect(rows[0]!.edgeType).toBe('MENTIONS_ORG')
  })

  it('returns BOTH typed mentions when an insight references person + org', () => {
    /** The most common real-world shape: one insight mentions a person AND an org. */
    const art = artWith([
      { from: 'insight:1', to: 'person:alice', type: 'MENTIONS_PERSON' },
      { from: 'insight:1', to: 'org:acme', type: 'MENTIONS_ORG' },
    ])
    const rows = graphNeighborsForNode(art, 'insight:1')
    const byId = new Map(rows.map((r) => [r.id, r.edgeType]))
    expect(byId.get('person:alice')).toBe('MENTIONS_PERSON')
    expect(byId.get('org:acme')).toBe('MENTIONS_ORG')
  })

  it('preserves the typed edgeType when the subject is the target (incoming direction)', () => {
    /**
     * Asking for a Person's neighbors: the insight→person edge surfaces
     * with direction "in" and the typed variant intact. The details panel
     * uses both fields to label the relationship correctly.
     */
    const art = artWith([
      { from: 'insight:1', to: 'person:alice', type: 'MENTIONS_PERSON' },
    ])
    const rows = graphNeighborsForNode(art, 'person:alice')
    expect(rows).toHaveLength(1)
    expect(rows[0]!.id).toBe('insight:1')
    expect(rows[0]!.edgeType).toBe('MENTIONS_PERSON')
    expect(rows[0]!.direction).toBe('in')
  })

  it('mixed legacy + typed MENTIONS on one center node both surface', () => {
    /**
     * Mid-migration corpus: a Person might be linked both by a legacy
     * generic ``MENTIONS`` edge AND a typed ``MENTIONS_PERSON`` edge from
     * different insights. The neighbor walk surfaces both rows with
     * their precise edge types so the user / debugger can see the
     * migration state.
     */
    const art = artWith([
      { from: 'insight:1', to: 'person:alice', type: 'MENTIONS' },
      { from: 'topic:ai', to: 'person:alice', type: 'MENTIONS_PERSON' },
    ])
    const rows = graphNeighborsForNode(art, 'person:alice')
    const edgeTypes = rows.map((r) => r.edgeType).sort()
    expect(edgeTypes).toEqual(['MENTIONS', 'MENTIONS_PERSON'])
  })
})
