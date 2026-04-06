import { describe, expect, it } from 'vitest'
import type { Core } from 'cytoscape'
import { graphNodeIdFromSearchHit, resolveCyNodeId } from './searchFocus'
import type { SearchHit } from '../api/searchApi'

function hit(docType: string, sourceId?: string): SearchHit {
  return {
    doc_id: 'd1',
    score: 0.9,
    text: 'text',
    metadata: { doc_type: docType, source_id: sourceId },
  }
}

describe('graphNodeIdFromSearchHit', () => {
  it('returns source_id for focusable doc_types', () => {
    for (const dt of ['insight', 'quote', 'kg_topic', 'kg_entity']) {
      expect(graphNodeIdFromSearchHit(hit(dt, 'abc'))).toBe('abc')
    }
  })

  it('trims whitespace from source_id', () => {
    expect(graphNodeIdFromSearchHit(hit('insight', '  x  '))).toBe('x')
  })

  it('returns null for non-focusable doc_type', () => {
    expect(graphNodeIdFromSearchHit(hit('episode', 'abc'))).toBeNull()
  })

  it('returns null when source_id is missing', () => {
    expect(graphNodeIdFromSearchHit(hit('insight'))).toBeNull()
  })

  it('returns null when source_id is blank', () => {
    expect(graphNodeIdFromSearchHit(hit('insight', '  '))).toBeNull()
  })
})

// ── resolveCyNodeId ──

function mockCore(existingIds: string[]): Core {
  const idSet = new Set(existingIds)
  return {
    $id: (id: string) => ({ empty: () => !idSet.has(id) }),
  } as unknown as Core
}

describe('resolveCyNodeId', () => {
  it('returns bare id when it exists in the graph', () => {
    const core = mockCore(['topic:foo'])
    expect(resolveCyNodeId(core, 'topic:foo')).toBe('topic:foo')
  })

  it('resolves g: prefix for GI nodes', () => {
    const core = mockCore(['g:insight:abc123'])
    expect(resolveCyNodeId(core, 'insight:abc123')).toBe('g:insight:abc123')
  })

  it('resolves k: prefix for KG nodes', () => {
    const core = mockCore(['k:topic:cuba-s-economic-crisis'])
    expect(resolveCyNodeId(core, 'topic:cuba-s-economic-crisis')).toBe(
      'k:topic:cuba-s-economic-crisis',
    )
  })

  it('resolves k:kg: prefix for double-prefixed KG nodes', () => {
    const core = mockCore(['k:kg:entity:org:cuba'])
    expect(resolveCyNodeId(core, 'entity:org:cuba')).toBe('k:kg:entity:org:cuba')
  })

  it('resolves g:gi: prefix', () => {
    const core = mockCore(['g:gi:quote:abc'])
    expect(resolveCyNodeId(core, 'quote:abc')).toBe('g:gi:quote:abc')
  })

  it('prefers bare id over prefixed when both exist', () => {
    const core = mockCore(['insight:x', 'g:insight:x'])
    expect(resolveCyNodeId(core, 'insight:x')).toBe('insight:x')
  })

  it('prefers g: over k: when both exist', () => {
    const core = mockCore(['g:topic:x', 'k:topic:x'])
    expect(resolveCyNodeId(core, 'topic:x')).toBe('g:topic:x')
  })

  it('returns null when no variant matches', () => {
    const core = mockCore(['unrelated:node'])
    expect(resolveCyNodeId(core, 'topic:missing')).toBeNull()
  })

  it('returns null for empty rawId', () => {
    const core = mockCore(['g:anything'])
    expect(resolveCyNodeId(core, '')).toBeNull()
  })
})
