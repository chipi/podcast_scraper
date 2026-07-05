import { describe, expect, it } from 'vitest'
import { augmentArtifactWithTrail } from './graphTrail'
import type { ParsedArtifact } from '../types/artifact'

function art(
  nodes: Array<{ id: string; type?: string }>,
  edges: Array<{ from: string; to: string; type?: string }> = [],
): ParsedArtifact {
  return {
    name: 'm',
    kind: 'gi',
    episodeId: null,
    nodes: nodes.length,
    edges: edges.length,
    nodeTypes: {},
    data: { nodes, edges },
  }
}

const merged = art(
  [{ id: 'A' }, { id: 'B' }, { id: 'C' }, { id: 'D' }],
  [
    { from: 'A', to: 'B' },
    { from: 'B', to: 'C' },
    { from: 'C', to: 'D' },
  ],
)

describe('augmentArtifactWithTrail', () => {
  it('returns the view unchanged (same ref) when the trail is empty', () => {
    const view = art([{ id: 'A' }])
    expect(augmentArtifactWithTrail(view, [], merged)).toBe(view)
  })

  it('adds a trail node + only edges connecting it to the visible set (no neighbours pulled)', () => {
    const view = art([{ id: 'A' }, { id: 'B' }], [{ from: 'A', to: 'B' }])
    const out = augmentArtifactWithTrail(view, ['C'], merged)!
    // C is added; D (C's neighbour in merged) is NOT pulled in — minimal expansion.
    expect(out.data.nodes!.map((n) => n.id)).toEqual(['A', 'B', 'C'])
    const edgeKeys = out.data.edges!.map((e) => `${e.from}->${e.to}`)
    expect(edgeKeys).toContain('B->C') // induced: both endpoints now visible
    expect(edgeKeys).not.toContain('C->D') // D not visible → C–D edge not added
    expect(out.nodes).toBe(3)
    expect(out.edges).toBe(2)
  })

  it('adds a genuinely disconnected trail node as a floating node (no edges)', () => {
    const view = art([{ id: 'A' }])
    const isolated = art([{ id: 'A' }, { id: 'Z' }], []) // Z has no edges
    const out = augmentArtifactWithTrail(view, ['Z'], isolated)!
    expect(out.data.nodes!.map((n) => n.id)).toEqual(['A', 'Z'])
    expect(out.data.edges!.length).toBe(0)
  })

  it('skips ids absent from merged and dedupes already-visible ones (no-op → same ref)', () => {
    const view = art([{ id: 'A' }, { id: 'B' }], [{ from: 'A', to: 'B' }])
    // B already present, "missing" not in merged → nothing to add.
    expect(augmentArtifactWithTrail(view, ['B', 'missing'], merged)).toBe(view)
  })
})
