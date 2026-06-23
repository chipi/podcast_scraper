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

  // ---------------------------------------------------------------------------
  // RFC-097 v3.0 typed-entity counts — Person + Organization show up as
  // distinct rows in the visual node counts (and the rendered metric rows).
  // ---------------------------------------------------------------------------

  function v3KgArtifact(): ParsedArtifact {
    return {
      name: 'ep1.kg.json',
      kind: 'kg',
      episodeId: 'ep1',
      nodes: 5,
      edges: 3,
      nodeTypes: { Episode: 1, Topic: 1, Person: 2, Organization: 1 },
      data: {
        episode_id: 'ep1',
        extraction: { model_version: 'gpt-4o', extracted_at: '2026-06-22' },
        nodes: [
          { id: 'episode:ep1', type: 'Episode' },
          { id: 't1', type: 'Topic', properties: { label: 'AI' } },
          { id: 'p1', type: 'Person', properties: { name: 'Ada' } },
          { id: 'p2', type: 'Person', properties: { name: 'Bob' } },
          { id: 'o1', type: 'Organization', properties: { name: 'Acme' } },
        ],
        edges: [
          { type: 'MENTIONS', from: 'p1', to: 'episode:ep1' },
          { type: 'MENTIONS', from: 'o1', to: 'episode:ep1' },
          { type: 'HAS_EPISODE', from: 'podcast:show', to: 'episode:ep1' },
        ],
      },
    }
  }

  it('typed Person nodes count under Entity_person (RFC-097 v3.0)', () => {
    const result = computeArtifactMetrics(v3KgArtifact())
    expect(result.visualNodeTypeCounts).toHaveProperty('Entity_person', 2)
  })

  it('typed Organization nodes count under Entity_organization (RFC-097 v3.0)', () => {
    const result = computeArtifactMetrics(v3KgArtifact())
    expect(result.visualNodeTypeCounts).toHaveProperty('Entity_organization', 1)
  })

  it('v3.0 KG metrics rows surface both Person and Organization counts when present', () => {
    /** Strict: the Person row must have key containing "person" / "people"
     * AND value = 2 (the fixture has 2 Person nodes). The Organization row
     * must have key containing "organi[sz]ation" AND value = 1. Without
     * the explicit key constraints, the assertion could pass on any row
     * with matching value.
     */
    const result = computeArtifactMetrics(v3KgArtifact())
    const personRow = result.rows.find(
      (r) => /person|people/i.test(r.k) && r.v === '2',
    )
    const orgRow = result.rows.find(
      (r) => /organi[sz]ation/i.test(r.k) && r.v === '1',
    )
    const debugRows = result.rows.map((r) => `${r.k}=${r.v}`).join(', ')
    expect(personRow, `Person count row missing or wrong value. Rows: ${debugRows}`).toBeDefined()
    expect(orgRow, `Organization count row missing or wrong value. Rows: ${debugRows}`).toBeDefined()
  })

  it('v3.0 KG edge counts include HAS_EPISODE and MENTIONS', () => {
    const result = computeArtifactMetrics(v3KgArtifact())
    expect(result.edgeTypeCounts).toHaveProperty('MENTIONS', 2)
    expect(result.edgeTypeCounts).toHaveProperty('HAS_EPISODE', 1)
  })

  it('v3.0 GI artifact: typed MENTIONS edges count distinctly from legacy MENTIONS', () => {
    /** Mid-migration corpus: one typed MENTIONS_PERSON + one typed MENTIONS_ORG
     * + one legacy MENTIONS. The edge counter must surface all three distinctly.
     */
    const v3Gi: ParsedArtifact = {
      name: 'ep1.gi.json',
      kind: 'gi',
      episodeId: 'ep1',
      nodes: 1,
      edges: 3,
      nodeTypes: { Insight: 1 },
      data: {
        episode_id: 'ep1',
        model_version: 'stub',
        prompt_version: 'v1',
        nodes: [
          {
            id: 'i1',
            type: 'Insight',
            properties: {
              grounded: true,
              insight_type: 'claim',
              position_hint: 0.3,
            },
          },
        ],
        edges: [
          { type: 'MENTIONS_PERSON', from: 'i1', to: 'p:1' },
          { type: 'MENTIONS_ORG', from: 'i1', to: 'o:1' },
          { type: 'MENTIONS', from: 'i1', to: 'legacy:1' },
        ],
      },
    }
    const result = computeArtifactMetrics(v3Gi)
    expect(result.edgeTypeCounts).toHaveProperty('MENTIONS_PERSON', 1)
    expect(result.edgeTypeCounts).toHaveProperty('MENTIONS_ORG', 1)
    expect(result.edgeTypeCounts).toHaveProperty('MENTIONS', 1)
  })
})
