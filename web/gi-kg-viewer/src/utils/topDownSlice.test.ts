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
