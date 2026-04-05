import { describe, expect, it } from 'vitest'
import type { ArtifactData, GraphFilterState, ParsedArtifact } from '../types/artifact'
import {
  applyGraphFilters,
  buildNodeTitle,
  defaultFilterState,
  ensureEpisodeToInsightEdges,
  entityDisplayNameFromId,
  filterArtifactEgoOneHop,
  filtersActive,
  findRawNodeInArtifact,
  nodeLabel,
  parseArtifact,
  toCytoElements,
  toGraphElements,
} from './parsing'

// ── fixtures ──

function giData(): ArtifactData {
  return {
    episode_id: 'ep1',
    model_version: 'gpt-4o',
    prompt_version: 'v2',
    nodes: [
      { id: 'episode:ep1', type: 'Episode', properties: { title: 'Ep 1' } },
      { id: 'i1', type: 'Insight', properties: { text: 'Climate change', grounded: true, episode_id: 'ep1' } },
      { id: 'i2', type: 'Insight', properties: { text: 'Warming', grounded: false, episode_id: 'ep1' } },
      { id: 'q1', type: 'Quote', properties: { text: 'The planet warms' } },
    ],
    edges: [{ type: 'supported_by', from: 'i1', to: 'q1' }],
  }
}

function parsedGi(): ParsedArtifact {
  return parseArtifact('ep1.gi.json', giData())
}

// ── parseArtifact ──

describe('parseArtifact', () => {
  it('detects GI from filename', () => {
    const art = parseArtifact('ep1.gi.json', { nodes: [], edges: [] })
    expect(art.kind).toBe('gi')
  })

  it('detects KG from filename', () => {
    const art = parseArtifact('ep1.kg.json', { nodes: [], edges: [] })
    expect(art.kind).toBe('kg')
  })

  it('infers GI from model_version + prompt_version', () => {
    const art = parseArtifact('data.json', { model_version: 'x', prompt_version: 'y', nodes: [], edges: [] })
    expect(art.kind).toBe('gi')
  })

  it('infers KG from extraction field', () => {
    const art = parseArtifact('data.json', { extraction: { model_version: 'x' }, nodes: [], edges: [] })
    expect(art.kind).toBe('kg')
  })

  it('counts nodes and edges', () => {
    const art = parsedGi()
    expect(art.nodes).toBe(4)
    expect(art.edges).toBeGreaterThanOrEqual(2)
  })

  it('builds nodeTypes map', () => {
    const art = parsedGi()
    expect(art.nodeTypes.Episode).toBe(1)
    expect(art.nodeTypes.Insight).toBe(2)
  })

  it('adds synthetic HAS_INSIGHT edges for GI', () => {
    const art = parsedGi()
    const hasInsight = (art.data.edges ?? []).filter((e) => e.type === 'HAS_INSIGHT')
    expect(hasInsight.length).toBeGreaterThanOrEqual(1)
  })
})

// ── ensureEpisodeToInsightEdges ──

describe('ensureEpisodeToInsightEdges', () => {
  it('adds HAS_INSIGHT for matching episode_id', () => {
    const nodes = [
      { id: 'episode:ep1', type: 'Episode' },
      { id: 'i1', type: 'Insight', properties: { episode_id: 'ep1' } },
    ]
    const result = ensureEpisodeToInsightEdges(nodes, [])
    expect(result.edges).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: 'HAS_INSIGHT', from: 'episode:ep1', to: 'i1' }),
      ]),
    )
  })

  it('does not duplicate existing edges', () => {
    const nodes = [
      { id: 'episode:ep1', type: 'Episode' },
      { id: 'i1', type: 'Insight', properties: { episode_id: 'ep1' } },
    ]
    const edges = [{ type: 'HAS_INSIGHT', from: 'episode:ep1', to: 'i1' }]
    const result = ensureEpisodeToInsightEdges(nodes, edges)
    const hi = result.edges.filter((e) => e.type === 'HAS_INSIGHT')
    expect(hi).toHaveLength(1)
  })

  it('returns unchanged when no episodes', () => {
    const nodes = [{ id: 'i1', type: 'Insight' }]
    const result = ensureEpisodeToInsightEdges(nodes, [])
    expect(result.edges).toHaveLength(0)
  })
})

// ── nodeLabel ──

describe('nodeLabel', () => {
  it('returns truncated text for Insight', () => {
    expect(nodeLabel({ type: 'Insight', properties: { text: 'A'.repeat(50) } })).toHaveLength(36)
  })

  it('returns label for Topic', () => {
    expect(nodeLabel({ type: 'Topic', properties: { label: 'climate' } })).toBe('climate')
  })

  it('returns title for Episode', () => {
    expect(nodeLabel({ type: 'Episode', properties: { title: 'Ep 1' } })).toBe('Ep 1')
  })

  it('falls back to type: id', () => {
    expect(nodeLabel({ id: 'x', type: 'Custom' })).toBe('Custom: x')
  })
})

// ── entityDisplayNameFromId ──

describe('entityDisplayNameFromId', () => {
  it('extracts and humanises person name', () => {
    expect(entityDisplayNameFromId('entity:person:john-doe')).toBe('John Doe')
  })

  it('strips g: prefix', () => {
    expect(entityDisplayNameFromId('g:entity:organization:acme-corp')).toBe('Acme Corp')
  })

  it('returns empty for non-entity id', () => {
    expect(entityDisplayNameFromId('topic:climate')).toBe('')
  })
})

// ── buildNodeTitle ──

describe('buildNodeTitle', () => {
  it('includes type and properties', () => {
    const title = buildNodeTitle({ id: 'i1', type: 'Insight', properties: { text: 'hello' } })
    expect(title).toContain('Insight')
    expect(title).toContain('hello')
  })
})

// ── findRawNodeInArtifact ──

describe('findRawNodeInArtifact', () => {
  it('finds node by id', () => {
    const art = parsedGi()
    expect(findRawNodeInArtifact(art, 'i1')).toBeTruthy()
  })

  it('returns null for missing id', () => {
    expect(findRawNodeInArtifact(parsedGi(), 'nope')).toBeNull()
  })

  it('returns null for null artifact', () => {
    expect(findRawNodeInArtifact(null, 'x')).toBeNull()
  })
})

// ── defaultFilterState ──

describe('defaultFilterState', () => {
  it('creates allowedTypes from node types', () => {
    const state = defaultFilterState(parsedGi())!
    expect(state.allowedTypes.Episode).toBe(true)
    expect(state.allowedTypes.Insight).toBe(true)
    expect(state.hideUngroundedInsights).toBe(false)
    expect(state.legendSoloVisual).toBeNull()
  })

  it('returns null for null artifact', () => {
    expect(defaultFilterState(null)).toBeNull()
  })
})

// ── filtersActive ──

describe('filtersActive', () => {
  it('returns false when all defaults', () => {
    const art = parsedGi()
    const state = defaultFilterState(art)!
    expect(filtersActive(art, state)).toBe(false)
  })

  it('returns true when a type is disabled', () => {
    const art = parsedGi()
    const state = defaultFilterState(art)!
    state.allowedTypes.Quote = false
    expect(filtersActive(art, state)).toBe(true)
  })

  it('returns true when hideUngroundedInsights is on for GI', () => {
    const art = parsedGi()
    const state = defaultFilterState(art)!
    state.hideUngroundedInsights = true
    expect(filtersActive(art, state)).toBe(true)
  })

  it('returns true when legendSoloVisual is set', () => {
    const art = parsedGi()
    const state = defaultFilterState(art)!
    state.legendSoloVisual = 'Insight'
    expect(filtersActive(art, state)).toBe(true)
  })
})

// ── applyGraphFilters ──

describe('applyGraphFilters', () => {
  it('hides disabled types', () => {
    const art = parsedGi()
    const state = defaultFilterState(art)!
    state.allowedTypes.Quote = false
    const filtered = applyGraphFilters(art, state)
    expect(filtered.nodeTypes.Quote).toBeUndefined()
    expect(filtered.nodes).toBeLessThan(art.nodes)
  })

  it('hides ungrounded insights', () => {
    const art = parsedGi()
    const state = defaultFilterState(art)!
    state.hideUngroundedInsights = true
    const filtered = applyGraphFilters(art, state)
    expect(filtered.nodeTypes.Insight).toBe(1)
  })

  it('solo visual keeps only matching group', () => {
    const art = parsedGi()
    const state = defaultFilterState(art)!
    state.legendSoloVisual = 'Episode'
    const filtered = applyGraphFilters(art, state)
    expect(filtered.nodes).toBe(1)
  })

  it('filters edges to remaining nodes', () => {
    const art = parsedGi()
    const state = defaultFilterState(art)!
    state.allowedTypes.Quote = false
    const filtered = applyGraphFilters(art, state)
    for (const e of filtered.data.edges ?? []) {
      const ids = new Set((filtered.data.nodes ?? []).map((n) => String(n.id)))
      expect(ids.has(String(e.from))).toBe(true)
      expect(ids.has(String(e.to))).toBe(true)
    }
  })

  it('filters GI/KG layers for "both" kind', () => {
    const art: ParsedArtifact = {
      name: 'merged',
      kind: 'both',
      episodeId: 'ep1',
      nodes: 3,
      edges: 0,
      nodeTypes: {},
      data: {
        nodes: [
          { id: 'g:i1', type: 'Insight' },
          { id: 'k:t1', type: 'Topic' },
          { id: '__unified_ep__:ep1', type: 'Episode' },
        ],
        edges: [],
      },
    }
    const state: GraphFilterState = {
      allowedTypes: { Insight: true, Topic: true, Episode: true },
      hideUngroundedInsights: false,
      legendSoloVisual: null,
      showGiLayer: true,
      showKgLayer: false,
    }
    const filtered = applyGraphFilters(art, state)
    const ids = (filtered.data.nodes ?? []).map((n) => String(n.id))
    expect(ids).toContain('g:i1')
    expect(ids).toContain('__unified_ep__:ep1')
    expect(ids).not.toContain('k:t1')
  })
})

// ── filterArtifactEgoOneHop ──

describe('filterArtifactEgoOneHop', () => {
  it('returns full artifact when focusId is null', () => {
    const art = parsedGi()
    expect(filterArtifactEgoOneHop(art, null)).toBe(art)
  })

  it('returns full artifact when focusId not in graph', () => {
    const art = parsedGi()
    expect(filterArtifactEgoOneHop(art, 'missing')).toBe(art)
  })

  it('keeps focus node and 1-hop neighbours', () => {
    const art = parsedGi()
    const ego = filterArtifactEgoOneHop(art, 'i1')
    const ids = (ego.data.nodes ?? []).map((n) => String(n.id))
    expect(ids).toContain('i1')
    expect(ids).toContain('q1')
    expect(ego.nodes).toBeLessThan(art.nodes)
  })
})

// ── toGraphElements / toCytoElements ──

describe('toGraphElements', () => {
  it('produces visNodes and visEdges', () => {
    const g = toGraphElements(parsedGi())
    expect(g.visNodes.length).toBeGreaterThan(0)
    expect(g.visEdges.length).toBeGreaterThan(0)
    expect(g.idSet.size).toBe(g.visNodes.length)
  })

  it('assigns groups from visualGroupForNode', () => {
    const g = toGraphElements(parsedGi())
    const groups = new Set(g.visNodes.map((n) => n.group))
    expect(groups.has('Episode')).toBe(true)
    expect(groups.has('Insight')).toBe(true)
  })
})

describe('toCytoElements', () => {
  it('produces Cytoscape element definitions', () => {
    const elems = toCytoElements(parsedGi())
    expect(elems.length).toBeGreaterThan(0)
    const nodeElem = elems.find((e) => e.data.id === 'i1')
    expect(nodeElem).toBeTruthy()
    expect(nodeElem!.data.type).toBe('Insight')
  })
})
