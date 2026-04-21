import { describe, expect, it } from 'vitest'
import type { ParsedArtifact } from '../types/artifact'
import { weaklyConnectedComponentCount } from './graphWeakComponents'

function miniArt(nodes: { id: string }[], edges: { from: string; to: string }[]): ParsedArtifact {
  return {
    name: 't.gi.json',
    kind: 'gi',
    episodeId: null,
    nodes: nodes.length,
    edges: edges.length,
    nodeTypes: {},
    data: { nodes: nodes as never[], edges: edges as never[] },
    sourceCorpusRelPath: 'm/a.gi.json',
  } as ParsedArtifact
}

describe('weaklyConnectedComponentCount', () => {
  it('returns 0 for null or empty nodes', () => {
    expect(weaklyConnectedComponentCount(null)).toBe(0)
    expect(
      weaklyConnectedComponentCount(
        miniArt([], [{ from: 'a', to: 'b' }]),
      ),
    ).toBe(0)
  })

  it('counts one component when fully connected', () => {
    const a = miniArt([{ id: 'a' }, { id: 'b' }, { id: 'c' }], [
      { from: 'a', to: 'b' },
      { from: 'b', to: 'c' },
    ])
    expect(weaklyConnectedComponentCount(a)).toBe(1)
  })

  it('counts two islands', () => {
    const a = miniArt([{ id: 'a' }, { id: 'b' }, { id: 'x' }, { id: 'y' }], [
      { from: 'a', to: 'b' },
      { from: 'x', to: 'y' },
    ])
    expect(weaklyConnectedComponentCount(a)).toBe(2)
  })

  it('isolates nodes without edges as separate components', () => {
    const a = miniArt([{ id: 'solo' }, { id: 'a' }, { id: 'b' }], [{ from: 'a', to: 'b' }])
    expect(weaklyConnectedComponentCount(a)).toBe(2)
  })

  it('ignores edges whose endpoints are missing from nodes', () => {
    const a = miniArt([{ id: 'a' }], [{ from: 'a', to: 'ghost' }])
    expect(weaklyConnectedComponentCount(a)).toBe(1)
  })
})
