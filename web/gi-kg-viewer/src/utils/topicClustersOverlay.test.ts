import { describe, expect, it } from 'vitest'
import type { ParsedArtifact } from '../types/artifact'
import { filterArtifactEgoOneHop } from './parsing'
import {
  applyThemeClustersOverlay,
  applyTopicClustersOverlay,
  themeClusterMemberTopicIdsForTopic,
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

describe('applyThemeClustersOverlay', () => {
  it('tags member Topic nodes with themeClusterId (a ring, NOT a compound parent)', () => {
    const data = {
      nodes: [
        { id: 'k:topic:alpha', type: 'Topic', properties: { label: 'Alpha' } },
        { id: 'k:topic:beta', type: 'Topic', properties: { label: 'Beta' } },
      ],
      edges: [],
    }
    const doc = {
      clusters: [
        {
          cluster_type: 'theme',
          graph_compound_parent_id: 'thc:energy',
          canonical_label: 'energy',
          members: [{ topic_id: 'topic:alpha' }],
        },
      ],
    }
    const out = applyThemeClustersOverlay(data, doc)
    // No new nodes added (unlike the semantic compound parents) — just a decoration.
    expect(out.nodes?.length).toBe(2)
    const alpha = out.nodes?.find((n) => String(n.id) === 'k:topic:alpha') as {
      themeClusterId?: string
    }
    expect(alpha?.themeClusterId).toBe('thc:energy')
    const beta = out.nodes?.find((n) => String(n.id) === 'k:topic:beta') as {
      themeClusterId?: string
    }
    expect(beta?.themeClusterId).toBeUndefined()
    // The member node is NOT re-parented (coexists with any semantic box).
    expect((out.nodes?.[0] as { parent?: string })?.parent).toBeUndefined()
  })

  it('returns data unchanged when doc empty', () => {
    const data = { nodes: [], edges: [] }
    expect(applyThemeClustersOverlay(data, null)).toBe(data)
    expect(applyThemeClustersOverlay(data, {})).toEqual(data)
  })

  // graph-v3 Tier 5A-2 — propagation walk paths (harden follow-up test coverage).
  it('tags Episode nodes via cluster.members[].episode_ids (stripping __unified_ep__: prefix)', () => {
    const data = {
      nodes: [
        // Episode ids in the merged graph carry a `__unified_ep__:` prefix;
        // the artifact stores raw ids in member.episode_ids.
        { id: '__unified_ep__:ep-abc', type: 'Episode' },
        { id: 'ep-def', type: 'Episode' }, // bare id — should still match
      ],
      edges: [],
    }
    const doc = {
      clusters: [
        {
          graph_compound_parent_id: 'thc:x',
          members: [{ topic_id: 'topic:x', episode_ids: ['ep-abc', 'ep-def'] }],
        },
      ],
    }
    const out = applyThemeClustersOverlay(data, doc)
    const ep1 = out.nodes?.find((n) => String(n.id) === '__unified_ep__:ep-abc') as {
      themeClusterId?: string
    }
    const ep2 = out.nodes?.find((n) => String(n.id) === 'ep-def') as { themeClusterId?: string }
    expect(ep1?.themeClusterId).toBe('thc:x')
    expect(ep2?.themeClusterId).toBe('thc:x')
  })

  it('propagates from Topic seeds to connected Insights + Persons via the edge list', () => {
    // Topic → Insight (ABOUT edge) → Person (MENTIONS_PERSON edge)
    const data = {
      nodes: [
        { id: 'g:topic:alpha', type: 'Topic' },
        { id: 'g:insight:i1', type: 'Insight' },
        { id: 'g:person:p1', type: 'Person' },
      ],
      edges: [
        { from: 'g:insight:i1', to: 'g:topic:alpha', type: 'ABOUT' },
        { from: 'g:insight:i1', to: 'g:person:p1', type: 'MENTIONS_PERSON' },
      ],
    }
    const doc = {
      clusters: [
        {
          graph_compound_parent_id: 'thc:x',
          members: [{ topic_id: 'topic:alpha' }],
        },
      ],
    }
    const out = applyThemeClustersOverlay(data, doc)
    const insight = out.nodes?.find((n) => n.id === 'g:insight:i1') as { themeClusterId?: string }
    const person = out.nodes?.find((n) => n.id === 'g:person:p1') as { themeClusterId?: string }
    // Round-1 propagation catches the Insight; round-2 catches the Person
    // (two rounds of BFS in applyThemeClustersOverlay).
    expect(insight?.themeClusterId).toBe('thc:x')
    expect(person?.themeClusterId).toBe('thc:x')
  })

  it('propagates from Episode seed to connected Podcast + Insight via HAS_EPISODE / HAS_INSIGHT edges', () => {
    const data = {
      nodes: [
        { id: '__unified_ep__:ep-x', type: 'Episode' },
        { id: 'k:podcast:show', type: 'Podcast' },
        { id: 'g:insight:i1', type: 'Insight' },
      ],
      edges: [
        { from: 'k:podcast:show', to: '__unified_ep__:ep-x', type: 'HAS_EPISODE' },
        { from: '__unified_ep__:ep-x', to: 'g:insight:i1', type: 'HAS_INSIGHT' },
      ],
    }
    const doc = {
      clusters: [
        {
          graph_compound_parent_id: 'thc:x',
          members: [{ topic_id: 'topic:none', episode_ids: ['ep-x'] }],
        },
      ],
    }
    const out = applyThemeClustersOverlay(data, doc)
    const podcast = out.nodes?.find((n) => n.id === 'k:podcast:show') as {
      themeClusterId?: string
    }
    const insight = out.nodes?.find((n) => n.id === 'g:insight:i1') as { themeClusterId?: string }
    expect(podcast?.themeClusterId).toBe('thc:x')
    expect(insight?.themeClusterId).toBe('thc:x')
  })

  it('first-cluster-wins: a node touched by two clusters joins the doc-order-first one', () => {
    // Both clusters include topic:alpha. Cluster A appears first in the doc
    // (largest by convention); Cluster B second. Alpha and everything it
    // propagates to should be tagged with cluster A.
    const data = {
      nodes: [
        { id: 'g:topic:alpha', type: 'Topic' },
        { id: 'g:insight:i1', type: 'Insight' },
      ],
      edges: [{ from: 'g:insight:i1', to: 'g:topic:alpha', type: 'ABOUT' }],
    }
    const doc = {
      clusters: [
        {
          graph_compound_parent_id: 'thc:A',
          member_count: 5,
          members: [{ topic_id: 'topic:alpha' }],
        },
        {
          graph_compound_parent_id: 'thc:B',
          member_count: 3,
          members: [{ topic_id: 'topic:alpha' }],
        },
      ],
    }
    const out = applyThemeClustersOverlay(data, doc)
    const alpha = out.nodes?.find((n) => n.id === 'g:topic:alpha') as { themeClusterId?: string }
    const insight = out.nodes?.find((n) => n.id === 'g:insight:i1') as { themeClusterId?: string }
    expect(alpha?.themeClusterId).toBe('thc:A')
    expect(insight?.themeClusterId).toBe('thc:A')
  })

  it('does not propagate to node types outside the allowlist (e.g. does not tag a random Segment node)', () => {
    const data = {
      nodes: [
        { id: 'g:topic:alpha', type: 'Topic' },
        { id: 'g:segment:s1', type: 'Segment' }, // not in propagateToTypes
      ],
      edges: [{ from: 'g:segment:s1', to: 'g:topic:alpha', type: 'SEGMENT_ABOUT' }],
    }
    const doc = {
      clusters: [
        { graph_compound_parent_id: 'thc:x', members: [{ topic_id: 'topic:alpha' }] },
      ],
    }
    const out = applyThemeClustersOverlay(data, doc)
    const seg = out.nodes?.find((n) => n.id === 'g:segment:s1') as { themeClusterId?: string }
    expect(seg?.themeClusterId).toBeUndefined()
  })

  it('tags both Person (raw) and Entity variants — raw KG uses either', () => {
    const data = {
      nodes: [
        { id: 'g:topic:alpha', type: 'Topic' },
        { id: 'g:insight:i1', type: 'Insight' },
        { id: 'g:person:p1', type: 'Person' },
        { id: 'g:entity:e1', type: 'Entity' },
      ],
      edges: [
        { from: 'g:insight:i1', to: 'g:topic:alpha', type: 'ABOUT' },
        { from: 'g:insight:i1', to: 'g:person:p1', type: 'MENTIONS' },
        { from: 'g:insight:i1', to: 'g:entity:e1', type: 'MENTIONS' },
      ],
    }
    const doc = {
      clusters: [
        { graph_compound_parent_id: 'thc:x', members: [{ topic_id: 'topic:alpha' }] },
      ],
    }
    const out = applyThemeClustersOverlay(data, doc)
    const p = out.nodes?.find((n) => n.id === 'g:person:p1') as { themeClusterId?: string }
    const e = out.nodes?.find((n) => n.id === 'g:entity:e1') as { themeClusterId?: string }
    expect(p?.themeClusterId).toBe('thc:x')
    expect(e?.themeClusterId).toBe('thc:x')
  })
})

describe('themeClusterMemberTopicIdsForTopic', () => {
  const doc = {
    clusters: [
      {
        graph_compound_parent_id: 'thc:energy',
        members: [{ topic_id: 'topic:oil' }, { topic_id: 'topic:lng' }],
      },
      { graph_compound_parent_id: 'thc:ai', members: [{ topic_id: 'topic:ml' }] },
    ],
  }

  it('returns the theme members for a member topic (bare-matched from a prefixed id)', () => {
    expect(themeClusterMemberTopicIdsForTopic(doc, 'k:topic:oil')).toEqual(['topic:oil', 'topic:lng'])
  })

  it('returns [] for a non-member topic, empty doc, or empty id', () => {
    expect(themeClusterMemberTopicIdsForTopic(doc, 'k:topic:none')).toEqual([])
    expect(themeClusterMemberTopicIdsForTopic(null, 'k:topic:oil')).toEqual([])
    expect(themeClusterMemberTopicIdsForTopic(doc, '')).toEqual([])
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

  it('falls back to the cluster doc member topic_ids when the ego slice omits members (#5)', () => {
    // Empty slice + no member rows: the compound and member-row paths both yield nothing, which
    // left topic-cluster / cluster-member Timeline tabs with no chart (only the theme path, resolved
    // from the full doc, worked). The doc is always complete, so its member topic_ids are the floor.
    const emptyArt: ParsedArtifact = {
      name: 'm',
      kind: 'gi',
      episodeId: null,
      nodes: 0,
      edges: 0,
      nodeTypes: {},
      data: { nodes: [], edges: [] },
    }
    const docMembers = [
      { topic_id: 'topic:economic-crisis-in-iran' },
      { topic_id: 'topic:iran-sanctions' },
      { topic_id: '   ' }, // blank → ignored
      { topic_id: 'topic:economic-crisis-in-iran' }, // duplicate → deduped
    ]
    const ids = clusterTimelineCilTopicIdsForCluster(emptyArt, 'tc:iran', [], docMembers)
    expect(ids).toEqual(['topic:economic-crisis-in-iran', 'topic:iran-sanctions'])
  })

  it('ignores the doc fallback once graph/member resolution finds ids', () => {
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
          { id: 'k:topic:iran-a', type: 'Topic', parent: 'tc:iran', properties: {} },
        ],
        edges: [],
      },
    }
    const ids = clusterTimelineCilTopicIdsForCluster(art, 'tc:iran', [], [
      { topic_id: 'topic:should-not-appear' },
    ])
    expect(ids).toEqual(['topic:iran-a'])
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
