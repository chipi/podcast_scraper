import { describe, expect, it } from 'vitest'
import { computeArtifactMetrics } from './metrics'
import type { ParsedArtifact } from '../types/artifact'

function giArtifact(overrides?: Partial<ParsedArtifact>): ParsedArtifact {
  return {
    name: 'ep1.gi.json',
    kind: 'gi',
    episodeId: 'ep1',
    nodes: 4,
    edges: 2,
    nodeTypes: { Episode: 1, Insight: 2, Quote: 1 },
    data: {
      episode_id: 'ep1',
      model_version: 'gpt-4o',
      prompt_version: 'v2',
      nodes: [
        { id: 'episode:ep1', type: 'Episode' },
        { id: 'i1', type: 'Insight', properties: { grounded: true } },
        { id: 'i2', type: 'Insight', properties: { grounded: false } },
        { id: 'q1', type: 'Quote' },
      ],
      edges: [
        { type: 'supported_by', from: 'i1', to: 'q1' },
        { type: 'HAS_INSIGHT', from: 'episode:ep1', to: 'i1' },
      ],
    },
    ...overrides,
  }
}

function kgArtifact(): ParsedArtifact {
  return {
    name: 'ep1.kg.json',
    kind: 'kg',
    episodeId: 'ep1',
    nodes: 3,
    edges: 1,
    nodeTypes: { Episode: 1, Topic: 1, Entity: 1 },
    data: {
      episode_id: 'ep1',
      extraction: { model_version: 'gpt-4o', extracted_at: '2026-01-01' },
      nodes: [
        { id: 'episode:ep1', type: 'Episode' },
        { id: 't1', type: 'Topic', properties: { label: 'climate' } },
        { id: 'e1', type: 'Entity', properties: { entity_kind: 'person' } },
      ],
      edges: [{ type: 'MENTIONS', from: 'episode:ep1', to: 't1' }],
    },
  }
}

describe('computeArtifactMetrics', () => {
  it('produces GI-specific rows', () => {
    const result = computeArtifactMetrics(giArtifact())
    const keys = result.rows.map((r) => r.k)
    expect(keys).toContain('Model')
    expect(keys).toContain('Insights')
    expect(keys).toContain('Grounded (true)')
    expect(keys).toContain('% grounded')
    expect(keys).toContain('Quotes')
  })

  it('computes grounded percentage', () => {
    const result = computeArtifactMetrics(giArtifact())
    const pctRow = result.rows.find((r) => r.k === '% grounded')
    expect(pctRow?.v).toBe('50.0%')
  })

  it('produces KG-specific rows', () => {
    const result = computeArtifactMetrics(kgArtifact())
    const keys = result.rows.map((r) => r.k)
    expect(keys).toContain('Extraction')
    expect(keys).toContain('Extracted at')
    expect(keys).toContain('Topics')
  })

  it('counts edge types', () => {
    const result = computeArtifactMetrics(giArtifact())
    expect(result.edgeTypeCounts).toHaveProperty('supported_by', 1)
  })

  it('returns visual node type counts', () => {
    const result = computeArtifactMetrics(kgArtifact())
    expect(result.visualNodeTypeCounts).toHaveProperty('Entity_person', 1)
    expect(result.visualNodeTypeCounts).toHaveProperty('Topic', 1)
  })

  it('handles "both" kind', () => {
    const art: ParsedArtifact = {
      name: 'merged',
      kind: 'both',
      episodeId: 'ep1',
      nodes: 2,
      edges: 0,
      nodeTypes: { Episode: 1, Insight: 1 },
      data: {
        model_version: 'gpt-4o',
        prompt_version: 'v2',
        extraction: { model_version: 'gpt-4o', extracted_at: '2026-01-01' },
        nodes: [
          { id: 'ep', type: 'Episode' },
          { id: 'i1', type: 'Insight' },
        ],
        edges: [],
      },
    }
    const result = computeArtifactMetrics(art)
    const keys = result.rows.map((r) => r.k)
    expect(keys).toContain('GI model')
    expect(keys).toContain('KG extraction')
  })

  it('common rows always present', () => {
    const result = computeArtifactMetrics(giArtifact())
    const keys = result.rows.map((r) => r.k)
    expect(keys.slice(0, 5)).toEqual(['File', 'Layer', 'Episode', 'Nodes', 'Edges'])
  })
})
