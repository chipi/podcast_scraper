import { describe, expect, it } from 'vitest'
import type { ParsedArtifact } from '../types/artifact'
import { filterArtifactEgoOneHop } from './parsing'
import {
  applyTopicClustersOverlay,
  clusterTimelineCilTopicIdsForCluster,
  clusterTimelineCilTopicIdsFromMemberRows,
  expandFilteredArtifactEgoWithTopicClusterNeighbors,
  findClusterByCompoundId,
  findTopicClusterContextForGraphNode,
  topicClusterMemberRowsForDetail,
  topicIdsFromGraphClusterCompound,
  withTopicClustersOnDisplay,
} from './topicClustersOverlay'

describe('applyTopicClustersOverlay', () => {
  it('adds TopicCluster parent and sets parent on prefixed topic ids', () => {
    const data = {
      nodes: [{ id: 'k:topic:alpha', type: 'Topic', properties: { label: 'Alpha' } }],
      edges: [],
    }
    const doc = {
      clusters: [
        {
          graph_compound_parent_id: 'tc:grp',
          canonical_label: 'Group',
          members: [{ topic_id: 'topic:alpha' }],
        },
      ],
    }
    const out = applyTopicClustersOverlay(data, doc)
    expect(out.nodes?.length).toBe(2)
    const topic = out.nodes?.find((n) => String(n.id) === 'k:topic:alpha')
    expect(topic?.parent).toBe('tc:grp')
    const parent = out.nodes?.find((n) => String(n.id) === 'tc:grp')
    expect(parent?.type).toBe('TopicCluster')
  })

  it('accepts v1 cluster_id as graph compound parent', () => {
    const data = {
      nodes: [{ id: 'k:topic:gamma', type: 'Topic', properties: { label: 'G' } }],
      edges: [],
    }
    const out = applyTopicClustersOverlay(data, {
      clusters: [
        {
          cluster_id: 'tc:legacy',
          canonical_label: 'Legacy',
          members: [{ topic_id: 'topic:gamma' }],
        },
      ],
    })
    expect(out.nodes?.find((n) => String(n.id) === 'k:topic:gamma')?.parent).toBe('tc:legacy')
  })

  it('returns data unchanged when doc empty', () => {
    const data = { nodes: [], edges: [] }
    expect(applyTopicClustersOverlay(data, null)).toBe(data)
    expect(applyTopicClustersOverlay(data, {})).toEqual(data)
  })
})

describe('findTopicClusterContextForGraphNode', () => {
  it('returns canonical label for prefixed topic id', () => {
    const doc = {
      clusters: [
        {
          graph_compound_parent_id: 'tc:g',
          canonical_label: 'Group label',
          members: [{ topic_id: 'topic:alpha' }],
        },
      ],
    }
    const ctx = findTopicClusterContextForGraphNode('k:topic:alpha', doc)
    expect(ctx).toEqual({ compoundParentId: 'tc:g', canonicalLabel: 'Group label' })
  })

  it('returns null when not a member', () => {
    const doc = {
      clusters: [
        {
          graph_compound_parent_id: 'tc:g',
          canonical_label: 'G',
          members: [{ topic_id: 'topic:other' }],
        },
      ],
    }
    expect(findTopicClusterContextForGraphNode('k:topic:alpha', doc)).toBeNull()
  })

  it('returns null when doc missing', () => {
    expect(findTopicClusterContextForGraphNode('k:topic:alpha', null)).toBeNull()
  })
})

describe('findClusterByCompoundId', () => {
  it('returns cluster matching graph_compound_parent_id', () => {
    const doc = {
      clusters: [
        {
          graph_compound_parent_id: 'tc:a',
          canonical_label: 'A',
          members: [{ topic_id: 'topic:x' }],
        },
      ],
    }
    const cl = findClusterByCompoundId(doc, 'tc:a')
    expect(cl?.canonical_label).toBe('A')
    expect(findClusterByCompoundId(doc, 'tc:missing')).toBeNull()
  })
})

describe('topicIdsFromGraphClusterCompound', () => {
  it('collects stripped topic ids for Topic nodes parented to compound', () => {
    const art: ParsedArtifact = {
      name: 'm',
      kind: 'gi',
      episodeId: null,
      nodes: 3,
      edges: 0,
      nodeTypes: {},
      data: {
        nodes: [
          { id: 'g:topic:alpha', type: 'Topic', parent: 'tc:x', properties: {} },
          { id: 'g:topic:beta', type: 'Topic', parent: 'tc:x', properties: {} },
          { id: 'other', type: 'Topic', parent: 'tc:y', properties: {} },
        ],
        edges: [],
      },
    }
    expect(topicIdsFromGraphClusterCompound(art, 'tc:x').sort()).toEqual(['topic:alpha', 'topic:beta'])
  })

  it('returns empty when compound id missing or no parents', () => {
    const art: ParsedArtifact = {
      name: 'm',
      kind: 'gi',
      episodeId: null,
      nodes: 1,
      edges: 0,
      nodeTypes: {},
      data: {
        nodes: [{ id: 'g:topic:alpha', type: 'Topic', parent: 'tc:z', properties: {} }],
        edges: [],
      },
    }
    expect(topicIdsFromGraphClusterCompound(art, 'tc:x')).toEqual([])
    expect(topicIdsFromGraphClusterCompound(null, 'tc:x')).toEqual([])
  })
})

describe('topicClusterMemberRowsForDetail', () => {
  it('resolves graph node ids and labels for members', () => {
    const art: ParsedArtifact = {
      name: 'm',
      kind: 'kg',
      episodeId: null,
      nodes: 2,
      edges: 1,
      nodeTypes: {},
      data: {
        nodes: [
          { id: 'k:topic:alpha', type: 'Topic', properties: { label: 'Alpha' } },
          { id: 'ep:1', type: 'Episode', properties: { title: 'E1' } },
        ],
        edges: [{ from: 'k:topic:alpha', to: 'ep:1', type: 'mentions' }],
      },
    }
    const cl = {
      graph_compound_parent_id: 'tc:g',
      members: [
        { topic_id: 'topic:alpha' },
        { topic_id: 'topic:orphan' },
      ],
    }
    const rows = topicClusterMemberRowsForDetail(art, cl)
    expect(rows).toHaveLength(2)
    expect(rows[0]).toMatchObject({
      topicId: 'topic:alpha',
      graphNodeId: 'k:topic:alpha',
      label: 'Alpha',
    })
    expect(rows[1]).toMatchObject({
      topicId: 'topic:orphan',
      graphNodeId: null,
    })
  })
})

describe('clusterTimelineCilTopicIdsFromMemberRows', () => {
  it('prefers stripped graph node id over JSON topic_id when graphNodeId is set', () => {
    const ids = clusterTimelineCilTopicIdsFromMemberRows([
      {
        topicId: 'topic:wrong-slug-from-json',
        graphNodeId: 'g:topic:actual-from-gi',
        label: 'X',
      },
      { topicId: 'topic:only-json', graphNodeId: null, label: 'Y' },
    ])
    expect(ids).toEqual(['topic:actual-from-gi', 'topic:only-json'])
  })

  it('dedupes identical CIL ids', () => {
    const ids = clusterTimelineCilTopicIdsFromMemberRows([
      { topicId: 'topic:a', graphNodeId: 'g:topic:a', label: '1' },
      { topicId: 'topic:a', graphNodeId: null, label: '2' },
    ])
    expect(ids).toEqual(['topic:a'])
  })
})

describe('clusterTimelineCilTopicIdsForCluster', () => {
  it('prefers graph Topic children under compound when JSON slugs drift from GI ids', () => {
    const art: ParsedArtifact = {
      name: 'm',
      kind: 'gi',
      episodeId: null,
      nodes: 2,
      edges: 0,
      nodeTypes: {},
      data: {
        nodes: [
          { id: 'tc:iran', type: 'TopicCluster', properties: {} },
          {
            id: 'k:topic:the-podcast-discusses-iran-economic',
            type: 'Topic',
            parent: 'tc:iran',
            properties: { label: 'Iran economic' },
          },
        ],
        edges: [],
      },
    }
    const memberRows = [
      {
        topicId: 'topic:economic-crisis-in-iran',
        graphNodeId: null,
        label: 'Economic Crisis in Iran',
      },
    ]
    const ids = clusterTimelineCilTopicIdsForCluster(art, 'tc:iran', memberRows)
    expect(ids).toEqual(['topic:the-podcast-discusses-iran-economic'])
  })
})

describe('expandFilteredArtifactEgoWithTopicClusterNeighbors', () => {
  it('merges cluster neighborhood when ego intersects a member topic', () => {
    const art: ParsedArtifact = {
      name: 't',
      kind: 'gi',
      episodeId: null,
      nodes: 4,
      edges: 2,
      nodeTypes: {},
      data: {
        nodes: [
          { id: 'g:ep1', type: 'Episode', properties: {} },
          { id: 'k:topic:a', type: 'Topic', properties: { label: 'A' } },
          { id: 'k:topic:b', type: 'Topic', properties: { label: 'B' } },
          { id: 'tc:x', type: 'TopicCluster', properties: { label: 'X' } },
        ],
        edges: [
          { from: 'g:ep1', to: 'k:topic:a', type: 'HAS_TOPIC' },
          { from: 'k:topic:a', to: 'k:topic:b', type: 'REL' },
        ],
      },
    }
    const doc = {
      clusters: [
        {
          graph_compound_parent_id: 'tc:x',
          canonical_label: 'X',
          members: [{ topic_id: 'topic:a' }, { topic_id: 'topic:b' }],
        },
      ],
    }
    const plainEgo = filterArtifactEgoOneHop(art, 'g:ep1')
    expect(plainEgo.data.nodes?.length).toBe(2)

    const expanded = expandFilteredArtifactEgoWithTopicClusterNeighbors(art, 'g:ep1', doc)
    expect((expanded.data.nodes || []).length).toBeGreaterThan(plainEgo.data.nodes!.length)
    const ids = new Set((expanded.data.nodes || []).map((n) => String(n.id)))
    expect(ids.has('k:topic:b')).toBe(true)
    expect(ids.has('tc:x')).toBe(true)
  })

  it('matches plain ego when focus is null', () => {
    const art: ParsedArtifact = {
      name: 't',
      kind: 'gi',
      episodeId: null,
      nodes: 1,
      edges: 0,
      nodeTypes: {},
      data: { nodes: [{ id: 'n1', type: 'Episode', properties: {} }], edges: [] },
    }
    expect(
      expandFilteredArtifactEgoWithTopicClusterNeighbors(art, null, { clusters: [] }),
    ).toEqual(filterArtifactEgoOneHop(art, null))
  })
})

describe('withTopicClustersOnDisplay', () => {
  it('returns null when artifact null', () => {
    expect(withTopicClustersOnDisplay(null, { clusters: [] })).toBeNull()
  })

  it('updates node counts when cluster parents added', () => {
    const art: ParsedArtifact = {
      name: 'x',
      kind: 'kg',
      episodeId: null,
      nodes: 1,
      edges: 0,
      nodeTypes: { Topic: 1 },
      data: {
        nodes: [{ id: 'k:topic:beta', type: 'Topic', properties: { label: 'B' } }],
        edges: [],
      },
    }
    const out = withTopicClustersOnDisplay(art, {
      clusters: [
        {
          graph_compound_parent_id: 'tc:z',
          canonical_label: 'Z',
          members: [{ topic_id: 'topic:beta' }],
        },
      ],
    })
    expect(out?.nodes).toBe(2)
    expect(out?.nodeTypes.TopicCluster).toBe(1)
  })
})
