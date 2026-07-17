import { describe, expect, it } from 'vitest'
import { buildTopDownSlice } from './topDownSlice'

describe('buildTopDownSlice (graph-v3 tier 8-1)', () => {
  it('returns an empty artifact when no theme doc is supplied', () => {
    const out = buildTopDownSlice({ themeDoc: null, fullArtifact: null })
    expect(out.nodes).toEqual([])
    expect(out.edges).toEqual([])
  })

  it('returns an empty artifact when no clusters carry a super_theme_id', () => {
    const out = buildTopDownSlice({
      themeDoc: { clusters: [{ graph_compound_parent_id: 'thc:x', canonical_label: 'X' }] },
      fullArtifact: null,
    })
    expect(out.nodes).toEqual([])
  })

  it('emits one SuperTheme node per unique super_theme_id', () => {
    const out = buildTopDownSlice({
      themeDoc: {
        clusters: [
          {
            graph_compound_parent_id: 'thc:a',
            canonical_label: 'A',
            super_theme_id: 'sth:one',
            super_theme_label: 'One',
          },
          {
            graph_compound_parent_id: 'thc:b',
            canonical_label: 'B',
            super_theme_id: 'sth:one',
            super_theme_label: 'One',
          },
          {
            graph_compound_parent_id: 'thc:c',
            canonical_label: 'C',
            super_theme_id: 'sth:two',
            super_theme_label: 'Two',
          },
        ],
      },
      fullArtifact: null,
    })
    expect(out.nodes.length).toBe(2)
    const byId = Object.fromEntries(out.nodes.map((n) => [n.id, n]))
    expect(byId['sth:one']?.type).toBe('SuperTheme')
    expect(byId['sth:one']?.properties?.label).toBe('One')
    expect(byId['sth:one']?.properties?.child_cluster_count).toBe(2)
    expect(byId['sth:two']?.properties?.child_cluster_count).toBe(1)
  })

  it('emits _topdown_link edges between every pair of super-themes', () => {
    const out = buildTopDownSlice({
      themeDoc: {
        clusters: [
          {
            graph_compound_parent_id: 'thc:a',
            super_theme_id: 'sth:a',
            super_theme_label: 'A',
          },
          {
            graph_compound_parent_id: 'thc:b',
            super_theme_id: 'sth:b',
            super_theme_label: 'B',
          },
          {
            graph_compound_parent_id: 'thc:c',
            super_theme_id: 'sth:c',
            super_theme_label: 'C',
          },
        ],
      },
      fullArtifact: null,
    })
    // 3 super-themes → 3 pairs (a-b, a-c, b-c).
    expect(out.edges.length).toBe(3)
    for (const e of out.edges) {
      expect(e.type).toBe('_topdown_link')
    }
  })

  it('deduplicates edges when the same super_theme_id spans multiple clusters', () => {
    const out = buildTopDownSlice({
      themeDoc: {
        clusters: [
          { graph_compound_parent_id: 'thc:a1', super_theme_id: 'sth:a' },
          { graph_compound_parent_id: 'thc:a2', super_theme_id: 'sth:a' },
          { graph_compound_parent_id: 'thc:b1', super_theme_id: 'sth:b' },
          { graph_compound_parent_id: 'thc:b2', super_theme_id: 'sth:b' },
        ],
      },
      fullArtifact: null,
    })
    // 2 super-themes → 1 edge; the extra child clusters don't duplicate.
    expect(out.nodes.length).toBe(2)
    expect(out.edges.length).toBe(1)
  })

  it('marks top_down_expanded true on nodes whose super_theme_id is in the expanded set', () => {
    const out = buildTopDownSlice({
      themeDoc: {
        clusters: [
          { graph_compound_parent_id: 'thc:a', super_theme_id: 'sth:a' },
          { graph_compound_parent_id: 'thc:b', super_theme_id: 'sth:b' },
        ],
      },
      fullArtifact: null,
      expandedSuperThemeIds: new Set(['sth:a']),
    })
    const byId = Object.fromEntries(out.nodes.map((n) => [n.id, n]))
    expect(byId['sth:a']?.properties?.top_down_expanded).toBe(true)
    expect(byId['sth:b']?.properties?.top_down_expanded).toBe(false)
  })

  it('projects tagged Topics + one hop of Insights under expanded super-themes', () => {
    const out = buildTopDownSlice({
      themeDoc: {
        clusters: [
          { graph_compound_parent_id: 'thc:a', super_theme_id: 'sth:a' },
          { graph_compound_parent_id: 'thc:b', super_theme_id: 'sth:b' },
        ],
      },
      fullArtifact: {
        nodes: [
          /* Topic tagged with cluster thc:a (rolls up to sth:a). */
          { id: 'topic:x', type: 'Topic', themeClusterId: 'thc:a' } as never,
          /* Insight not tagged directly — inherits sth:a via one-hop from topic:x. */
          { id: 'insight:i1', type: 'Insight' } as never,
          /* Topic tagged with cluster thc:b — should NOT come in because sth:b isn't expanded. */
          { id: 'topic:y', type: 'Topic', themeClusterId: 'thc:b' } as never,
        ],
        edges: [
          { type: 'SUPPORTED_BY', from: 'insight:i1', to: 'topic:x' },
        ],
      },
      expandedSuperThemeIds: new Set(['sth:a']),
    })
    const ids = new Set(out.nodes.map((n) => n.id))
    expect(ids.has('sth:a')).toBe(true)
    expect(ids.has('sth:b')).toBe(true)
    expect(ids.has('topic:x')).toBe(true)
    expect(ids.has('insight:i1')).toBe(true) // pulled in via one-hop
    expect(ids.has('topic:y')).toBe(false) // sth:b not expanded
    // Projected children attach to their super-theme via `parent`.
    const topicX = out.nodes.find((n) => n.id === 'topic:x')
    expect((topicX as { parent?: string })?.parent).toBe('sth:a')
    const insightI1 = out.nodes.find((n) => n.id === 'insight:i1')
    expect((insightI1 as { parent?: string })?.parent).toBe('sth:a')
  })

  it('drops edges whose endpoints are not both in the projected set', () => {
    const out = buildTopDownSlice({
      themeDoc: {
        clusters: [
          { graph_compound_parent_id: 'thc:a', super_theme_id: 'sth:a' },
          { graph_compound_parent_id: 'thc:b', super_theme_id: 'sth:b' },
        ],
      },
      fullArtifact: {
        nodes: [
          { id: 'topic:x', type: 'Topic', themeClusterId: 'thc:a' } as never,
          { id: 'insight:i1', type: 'Insight' } as never,
          /* Under sth:b (not expanded); nothing about it should survive. */
          { id: 'topic:y', type: 'Topic', themeClusterId: 'thc:b' } as never,
          { id: 'insight:i2', type: 'Insight' } as never,
        ],
        edges: [
          { type: 'SUPPORTED_BY', from: 'insight:i1', to: 'topic:x' },
          { type: 'SUPPORTED_BY', from: 'insight:i2', to: 'topic:y' },
          /* Cross-super-theme edge — dropped because topic:y isn't projected. */
          { type: 'RELATED', from: 'topic:x', to: 'topic:y' },
        ],
      },
      expandedSuperThemeIds: new Set(['sth:a']),
    })
    const ids = new Set(out.nodes.map((n) => n.id))
    expect(ids.has('topic:y')).toBe(false)
    expect(ids.has('insight:i2')).toBe(false)
    const realEdges = out.edges.filter((e) => e.type !== '_topdown_link')
    // Only the SUPPORTED_BY on topic:x survives — the sth:b edges and the
    // cross-super RELATED edge all drop because their other endpoint
    // isn't in the projection.
    expect(realEdges.length).toBe(1)
    expect(realEdges[0]?.type).toBe('SUPPORTED_BY')
  })

  it('falls back to super_theme_id when super_theme_label is absent', () => {
    const out = buildTopDownSlice({
      themeDoc: {
        clusters: [
          { graph_compound_parent_id: 'thc:a', super_theme_id: 'sth:only' },
        ],
      },
      fullArtifact: null,
    })
    expect(out.nodes[0]?.properties?.label).toBe('sth:only')
  })
})
