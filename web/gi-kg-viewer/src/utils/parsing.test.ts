import { describe, expect, it } from 'vitest'
import type { ArtifactData, GraphFilterState, ParsedArtifact } from '../types/artifact'
import {
  applyGraphDefaultNodeTypeVisibility,
  applyGraphFilters,
  buildNodeTitle,
  countPersonEntityIncidentEdges,
  defaultFilterState,
  ensureEpisodeToInsightEdges,
  entityDisplayNameFromId,
  findEpisodeGraphNodeIdForEpisodeKey,
  quoteAttributionDisplayFromId,
  filterArtifactEgoAroundTopicCluster,
  filterArtifactEgoOneHop,
  filtersActive,
  filtersActiveExcludingNodeTypes,
  graphTypesDeviateFromGraphSpec,
  findRawNodeInArtifact,
  fullPrimaryNodeLabel,
  insightProvenanceLine,
  insightRelatedTopicRows,
  insightSupportingQuoteRows,
  insightSupportingTranscriptAggregate,
  nodeLabel,
  primaryTextFromLooseGiNode,
  parseArtifact,
  confidenceTierFromInsightProperties,
  isInsightUngrounded,
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
  it('truncates long text to max graph label length', () => {
    const label = nodeLabel({ type: 'Insight', properties: { text: 'A'.repeat(50) } })
    expect(label.length).toBeLessThanOrEqual(40)
  })

  it('returns short label for Topic', () => {
    expect(nodeLabel({ type: 'Topic', properties: { label: 'climate' } })).toBe('climate')
  })

  it('shortens long Topic label at natural break', () => {
    const label = nodeLabel({
      type: 'Topic',
      properties: { label: 'Cuba faces a severe crisis, leading to blackouts and food shortages' },
    })
    expect(label).toBe('Cuba faces a severe crisis')
  })

  it('returns title for Episode', () => {
    expect(nodeLabel({ type: 'Episode', properties: { title: 'Ep 1' } })).toBe('Ep 1')
  })

  it('prefers name over label', () => {
    expect(nodeLabel({ type: 'Entity', properties: { name: 'Cuba', label: 'cuba-label' } })).toBe('Cuba')
  })

  it('falls back to type: id', () => {
    expect(nodeLabel({ id: 'x', type: 'Custom' })).toBe('Custom: x')
  })

  it('works for any unknown node type with properties', () => {
    expect(nodeLabel({ type: 'Foo', properties: { name: 'Bar' } })).toBe('Bar')
  })
})

describe('fullPrimaryNodeLabel', () => {
  it('returns uncapped text when nodeLabel would shorten', () => {
    const long = 'A'.repeat(50)
    expect(nodeLabel({ type: 'Insight', properties: { text: long } }).length).toBeLessThanOrEqual(40)
    expect(fullPrimaryNodeLabel({ type: 'Insight', properties: { text: long } })).toBe(long)
  })

  it('matches nodeLabel for short strings', () => {
    expect(fullPrimaryNodeLabel({ type: 'Topic', properties: { label: 'climate' } })).toBe('climate')
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

  it('humanises CIL person: and org: ids', () => {
    expect(entityDisplayNameFromId('person:john-doe')).toBe('John Doe')
    expect(entityDisplayNameFromId('k:kg:org:acme-corp')).toBe('Acme Corp')
  })

  it('returns empty for non-entity id', () => {
    expect(entityDisplayNameFromId('topic:climate')).toBe('')
  })
})

// ── quoteAttributionDisplayFromId ──

describe('quoteAttributionDisplayFromId', () => {
  it('delegates to entityDisplayNameFromId for person and org ids', () => {
    expect(quoteAttributionDisplayFromId('person:jane-doe')).toBe('Jane Doe')
    expect(quoteAttributionDisplayFromId('org:acme-inc')).toBe('Acme Inc')
  })

  it('humanises legacy speaker: slugs', () => {
    expect(quoteAttributionDisplayFromId('speaker:host-one')).toBe('Host One')
  })

  it('returns raw id when not a known pattern', () => {
    expect(quoteAttributionDisplayFromId('raw-id-123')).toBe('raw-id-123')
  })

  it('returns empty for null or blank', () => {
    expect(quoteAttributionDisplayFromId(null)).toBe('')
    expect(quoteAttributionDisplayFromId('')).toBe('')
    expect(quoteAttributionDisplayFromId('   ')).toBe('')
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
    expect(state.allowedEdgeTypes.supported_by).toBe(true)
    expect(state.allowedEdgeTypes.HAS_INSIGHT).toBe(true)
  })

  it('returns null for null artifact', () => {
    expect(defaultFilterState(null)).toBeNull()
  })
})

describe('applyGraphDefaultNodeTypeVisibility', () => {
  it('turns off Quote Speaker by default; Episode stays on', () => {
    const state = defaultFilterState(parsedGi())!
    applyGraphDefaultNodeTypeVisibility(state)
    expect(state.allowedTypes.Episode).toBe(true)
    expect(state.allowedTypes.Quote).toBe(false)
    expect(state.allowedTypes.Insight).toBe(true)
    expect(graphTypesDeviateFromGraphSpec(state)).toBe(false)
  })

  it('graphTypesDeviateFromGraphSpec detects user overrides', () => {
    const state = defaultFilterState(parsedGi())!
    applyGraphDefaultNodeTypeVisibility(state)
    state.allowedTypes.Episode = false
    expect(graphTypesDeviateFromGraphSpec(state)).toBe(true)
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

  it('returns true when an edge type is disabled', () => {
    const art = parsedGi()
    const state = defaultFilterState(art)!
    const firstKey = Object.keys(state.allowedEdgeTypes)[0]
    expect(firstKey).toBeDefined()
    state.allowedEdgeTypes[firstKey] = false
    expect(filtersActive(art, state)).toBe(true)
  })
})

describe('filtersActiveExcludingNodeTypes', () => {
  it('returns false when only node types differ from defaults', () => {
    const art = parsedGi()
    const state = defaultFilterState(art)!
    state.allowedTypes.Quote = false
    expect(filtersActive(art, state)).toBe(true)
    expect(filtersActiveExcludingNodeTypes(art, state)).toBe(false)
  })

  it('returns true when hideUngroundedInsights is on', () => {
    const art = parsedGi()
    const state = defaultFilterState(art)!
    state.hideUngroundedInsights = true
    expect(filtersActiveExcludingNodeTypes(art, state)).toBe(true)
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

  it('preserves transcript source corpus paths on filtered artifact', () => {
    const art: ParsedArtifact = {
      ...parsedGi(),
      sourceCorpusRelPath: 'feeds/x/run/metadata/a.gi.json',
      sourceCorpusRelPathByEpisodeId: { ep1: 'feeds/x/run/metadata/a.gi.json' },
    }
    const state = defaultFilterState(art)!
    const filtered = applyGraphFilters(art, state)
    expect(filtered.sourceCorpusRelPath).toBe(art.sourceCorpusRelPath)
    expect(filtered.sourceCorpusRelPathByEpisodeId).toEqual(art.sourceCorpusRelPathByEpisodeId)
  })

  it('hides ungrounded insights', () => {
    const art = parsedGi()
    const state = defaultFilterState(art)!
    state.hideUngroundedInsights = true
    const filtered = applyGraphFilters(art, state)
    expect(filtered.nodeTypes.Insight).toBe(1)
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

  it('hides edges when edge type is disabled', () => {
    const art = parsedGi()
    const state = defaultFilterState(art)!
    state.allowedEdgeTypes.supported_by = false
    const filtered = applyGraphFilters(art, state)
    const types = (filtered.data.edges ?? []).map((e) => String(e.type || ''))
    expect(types).not.toContain('supported_by')
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
      allowedEdgeTypes: {},
      hideUngroundedInsights: false,
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

  it('preserves transcript source corpus paths on ego subgraph', () => {
    const art: ParsedArtifact = {
      ...parsedGi(),
      sourceCorpusRelPath: null,
      sourceCorpusRelPathByEpisodeId: { ep1: 'feeds/y/metadata/z.gi.json' },
    }
    const ego = filterArtifactEgoOneHop(art, 'i1')
    expect(ego.sourceCorpusRelPath).toBeNull()
    expect(ego.sourceCorpusRelPathByEpisodeId).toEqual(art.sourceCorpusRelPathByEpisodeId)
  })
})

describe('filterArtifactEgoAroundTopicCluster', () => {
  it('includes compound, members, and 1-hop neighbors of members', () => {
    const art: ParsedArtifact = {
      name: 'x',
      kind: 'kg',
      episodeId: null,
      nodes: 5,
      edges: 3,
      nodeTypes: {},
      data: {
        nodes: [
          { id: 'tc:g', type: 'TopicCluster', properties: { label: 'G' } },
          { id: 'k:topic:a', type: 'Topic', properties: { label: 'A' }, parent: 'tc:g' },
          { id: 'k:topic:b', type: 'Topic', properties: { label: 'B' }, parent: 'tc:g' },
          { id: 'ep:1', type: 'Episode', properties: { title: 'E1' } },
          { id: 'ep:2', type: 'Episode', properties: { title: 'E2' } },
        ],
        edges: [
          { from: 'k:topic:a', to: 'ep:1', type: 'mentions' },
          { from: 'k:topic:b', to: 'ep:2', type: 'mentions' },
        ],
      },
    }
    const sub = filterArtifactEgoAroundTopicCluster(art, 'tc:g', ['k:topic:a', 'k:topic:b'])
    const ids = (sub.data.nodes ?? []).map((n) => String(n.id)).sort()
    expect(ids).toEqual(['ep:1', 'ep:2', 'k:topic:a', 'k:topic:b', 'tc:g'])
    expect(sub.data.edges?.length).toBe(2)
  })
})

describe('primaryTextFromLooseGiNode', () => {
  it('matches fullPrimaryNodeLabel for CIL-style dict', () => {
    const loose = {
      id: 'ins1',
      type: 'Insight',
      properties: { text: 'Hello from API' },
    }
    expect(primaryTextFromLooseGiNode(loose)).toBe(
      fullPrimaryNodeLabel({
        id: 'ins1',
        type: 'Insight',
        properties: { text: 'Hello from API' },
      }),
    )
  })
})

describe('findEpisodeGraphNodeIdForEpisodeKey', () => {
  it('resolves episode: prefix id', () => {
    const art = parsedGi()
    expect(findEpisodeGraphNodeIdForEpisodeKey(art, 'ep1')).toBe('episode:ep1')
  })

  it('resolves g:episode merged-style id', () => {
    const art = parseArtifact('x.gi.json', {
      episode_id: 'u1',
      model_version: 'm',
      prompt_version: 'p',
      nodes: [
        { id: 'g:episode:u1', type: 'Episode', properties: { title: 'T' } },
        { id: 'ins1', type: 'Insight', properties: { text: 'x', episode_id: 'u1', grounded: true } },
      ],
      edges: [],
    })
    expect(findEpisodeGraphNodeIdForEpisodeKey(art, 'u1')).toBe('g:episode:u1')
  })

  it('returns null when missing', () => {
    expect(findEpisodeGraphNodeIdForEpisodeKey(parsedGi(), 'nope')).toBeNull()
  })
})

describe('insightSupportingQuoteRows', () => {
  it('lists quotes from SUPPORTED_BY out-edges (underscore type)', () => {
    const art = parsedGi()
    const rows = insightSupportingQuoteRows(art, 'i1')
    expect(rows.map((r) => r.id)).toEqual(['q1'])
    expect(rows[0]?.preview).toContain('planet')
  })

  it('matches SUPPORTED_BY uppercase from GI JSON', () => {
    const art = parseArtifact('x.gi.json', {
      episode_id: 'e',
      model_version: 'm',
      prompt_version: 'p',
      nodes: [
        { id: 'episode:e', type: 'Episode', properties: { title: 'E' } },
        { id: 'ins:x', type: 'Insight', properties: { text: 'I', episode_id: 'e', grounded: true } },
        { id: 'quo:y', type: 'Quote', properties: { text: 'Quote body here.' } },
      ],
      edges: [{ type: 'SUPPORTED_BY', from: 'ins:x', to: 'quo:y' }],
    })
    const rows = insightSupportingQuoteRows(art, 'ins:x')
    expect(rows).toHaveLength(1)
    expect(rows[0]?.id).toBe('quo:y')
  })

  it('sorts by char_start then timestamp_start_ms', () => {
    const art = parseArtifact('x.gi.json', {
      episode_id: 'e',
      model_version: 'm',
      prompt_version: 'p',
      nodes: [
        {
          id: 'episode:e',
          type: 'Episode',
          properties: {
            podcast_id: 'p',
            title: 'E',
            publish_date: '2020-01-01T00:00:00Z',
          },
        },
        { id: 'ins', type: 'Insight', properties: { text: 'I', episode_id: 'e', grounded: true } },
        {
          id: 'qLate',
          type: 'Quote',
          properties: {
            text: 'Late',
            episode_id: 'e',
            char_start: 100,
            char_end: 101,
            timestamp_start_ms: 0,
            timestamp_end_ms: 1,
            transcript_ref: 't.txt',
          },
        },
        {
          id: 'qEarly',
          type: 'Quote',
          properties: {
            text: 'Early',
            episode_id: 'e',
            char_start: 10,
            char_end: 11,
            timestamp_start_ms: 0,
            timestamp_end_ms: 1,
            transcript_ref: 't.txt',
          },
        },
      ],
      edges: [
        { type: 'SUPPORTED_BY', from: 'ins', to: 'qLate' },
        { type: 'SUPPORTED_BY', from: 'ins', to: 'qEarly' },
      ],
    })
    const rows = insightSupportingQuoteRows(art, 'ins')
    expect(rows.map((r) => r.id)).toEqual(['qEarly', 'qLate'])
  })
})

describe('insightSupportingTranscriptAggregate', () => {
  it('returns one transcript ref and all char ranges when quotes agree', () => {
    const art = parseArtifact('x.gi.json', {
      episode_id: 'e',
      model_version: 'm',
      prompt_version: 'p',
      nodes: [
        { id: 'episode:e', type: 'Episode', properties: { title: 'E' } },
        { id: 'ins', type: 'Insight', properties: { text: 'I', episode_id: 'e', grounded: true } },
        {
          id: 'q1',
          type: 'Quote',
          properties: {
            text: 'A',
            episode_id: 'e',
            char_start: 0,
            char_end: 2,
            transcript_ref: 't.txt',
          },
        },
        {
          id: 'q2',
          type: 'Quote',
          properties: {
            text: 'B',
            episode_id: 'e',
            char_start: 10,
            char_end: 12,
            transcript_ref: 't.txt',
          },
        },
      ],
      edges: [
        { type: 'SUPPORTED_BY', from: 'ins', to: 'q1' },
        { type: 'SUPPORTED_BY', from: 'ins', to: 'q2' },
      ],
    })
    const agg = insightSupportingTranscriptAggregate(art, 'ins')
    expect(agg).not.toBeNull()
    expect(agg!.transcriptRef).toBe('t.txt')
    expect(agg!.episodeId).toBe('e')
    expect(agg!.charRanges).toEqual([
      { charStart: 0, charEnd: 2 },
      { charStart: 10, charEnd: 12 },
    ])
  })

  it('uses first quote episode_id when insight omits episode_id', () => {
    const art = parseArtifact('x.gi.json', {
      episode_id: 'e',
      model_version: 'm',
      prompt_version: 'p',
      nodes: [
        { id: 'episode:e', type: 'Episode', properties: { title: 'E' } },
        { id: 'ins', type: 'Insight', properties: { text: 'I', grounded: true } },
        {
          id: 'q1',
          type: 'Quote',
          properties: {
            text: 'A',
            episode_id: 'e',
            char_start: 0,
            char_end: 2,
            transcript_ref: 't.txt',
          },
        },
        {
          id: 'q2',
          type: 'Quote',
          properties: {
            text: 'B',
            episode_id: 'e',
            char_start: 10,
            char_end: 12,
            transcript_ref: 't.txt',
          },
        },
      ],
      edges: [
        { type: 'SUPPORTED_BY', from: 'ins', to: 'q1' },
        { type: 'SUPPORTED_BY', from: 'ins', to: 'q2' },
      ],
    })
    const agg = insightSupportingTranscriptAggregate(art, 'ins')
    expect(agg).not.toBeNull()
    expect(agg!.episodeId).toBe('e')
  })

  it('returns null when supporting quotes use different transcript_ref values', () => {
    const art = parseArtifact('x.gi.json', {
      episode_id: 'e',
      model_version: 'm',
      prompt_version: 'p',
      nodes: [
        { id: 'episode:e', type: 'Episode', properties: { title: 'E' } },
        { id: 'ins', type: 'Insight', properties: { text: 'I', episode_id: 'e', grounded: true } },
        {
          id: 'q1',
          type: 'Quote',
          properties: {
            text: 'A',
            char_start: 0,
            char_end: 1,
            transcript_ref: 'a.txt',
          },
        },
        {
          id: 'q2',
          type: 'Quote',
          properties: {
            text: 'B',
            char_start: 0,
            char_end: 1,
            transcript_ref: 'b.txt',
          },
        },
      ],
      edges: [
        { type: 'SUPPORTED_BY', from: 'ins', to: 'q1' },
        { type: 'SUPPORTED_BY', from: 'ins', to: 'q2' },
      ],
    })
    expect(insightSupportingTranscriptAggregate(art, 'ins')).toBeNull()
  })

  it('returns null when a quote lacks finite char offsets', () => {
    const art = parseArtifact('x.gi.json', {
      episode_id: 'e',
      model_version: 'm',
      prompt_version: 'p',
      nodes: [
        { id: 'episode:e', type: 'Episode', properties: { title: 'E' } },
        { id: 'ins', type: 'Insight', properties: { text: 'I', episode_id: 'e', grounded: true } },
        {
          id: 'q1',
          type: 'Quote',
          properties: { text: 'A', char_start: 0, char_end: 1, transcript_ref: 't.txt' },
        },
        {
          id: 'q2',
          type: 'Quote',
          properties: { text: 'B', transcript_ref: 't.txt' },
        },
      ],
      edges: [
        { type: 'SUPPORTED_BY', from: 'ins', to: 'q1' },
        { type: 'SUPPORTED_BY', from: 'ins', to: 'q2' },
      ],
    })
    expect(insightSupportingTranscriptAggregate(art, 'ins')).toBeNull()
  })
})

describe('insightRelatedTopicRows', () => {
  it('collects Topic neighbors via ABOUT / RELATED_TO (either direction)', () => {
    const art = parseArtifact('x.gi.json', {
      episode_id: 'e',
      model_version: 'm',
      prompt_version: 'p',
      nodes: [
        {
          id: 'episode:e',
          type: 'Episode',
          properties: {
            podcast_id: 'p',
            title: 'E',
            publish_date: '2020-01-01T00:00:00Z',
          },
        },
        { id: 'ins', type: 'Insight', properties: { text: 'I', episode_id: 'e', grounded: true } },
        { id: 't1', type: 'Topic', properties: { label: 'Alpha' } },
        { id: 't2', type: 'Topic', properties: { label: 'Beta' } },
      ],
      edges: [
        { type: 'ABOUT', from: 'ins', to: 't1' },
        { type: 'related-to', from: 't2', to: 'ins' },
      ],
    })
    const rows = insightRelatedTopicRows(art, 'ins')
    expect(rows.map((r) => r.label)).toEqual(['Alpha', 'Beta'])
  })
})

describe('insightProvenanceLine', () => {
  it('joins model, prompt, optional extraction timestamp, and filename', () => {
    const art = parseArtifact('file.gi.json', {
      episode_id: 'e',
      model_version: 'mv',
      prompt_version: 'pv',
      extraction: { extracted_at: '2024-02-01' },
      nodes: [],
      edges: [],
    })
    const line = insightProvenanceLine(art)
    expect(line).toContain('model mv')
    expect(line).toContain('prompt pv')
    expect(line).toContain('extracted 2024-02-01')
    expect(line).toContain('from file.gi.json')
  })
})

describe('countPersonEntityIncidentEdges', () => {
  it('returns zeros when artifact or id missing', () => {
    expect(countPersonEntityIncidentEdges(null, 'p1')).toEqual({
      spokenByQuotes: 0,
      spokeInEpisodes: 0,
    })
    const art = parseArtifact('x.gi.json', { nodes: [], edges: [] })
    expect(countPersonEntityIncidentEdges(art, null)).toEqual({
      spokenByQuotes: 0,
      spokeInEpisodes: 0,
    })
  })

  it('counts SPOKEN_BY into node and SPOKE_IN out from node', () => {
    const art = parseArtifact('x.gi.json', {
      nodes: [],
      edges: [
        { type: 'SPOKEN_BY', from: 'q1', to: 'person:ada' },
        { type: 'SPOKEN_BY', from: 'q2', to: 'person:ada' },
        { type: 'SPOKE_IN', from: 'person:ada', to: 'episode:e1' },
      ],
    })
    expect(countPersonEntityIncidentEdges(art, 'person:ada')).toEqual({
      spokenByQuotes: 2,
      spokeInEpisodes: 1,
    })
  })

  it('normalizes edge type casing', () => {
    const art = parseArtifact('x.gi.json', {
      nodes: [],
      edges: [{ type: 'spoken_by', from: 'q1', to: 'person:bob' }],
    })
    expect(countPersonEntityIncidentEdges(art, 'person:bob')).toEqual({
      spokenByQuotes: 1,
      spokeInEpisodes: 0,
    })
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

  it('clears Topic canvas label when it duplicates parent TopicCluster label', () => {
    const art: ParsedArtifact = {
      name: 'x',
      kind: 'kg',
      episodeId: null,
      nodes: 2,
      edges: 0,
      nodeTypes: {},
      data: {
        nodes: [
          { id: 'tc:g', type: 'TopicCluster', properties: { label: 'Shared' } },
          {
            id: 'k:topic:a',
            type: 'Topic',
            parent: 'tc:g',
            properties: { label: 'Shared' },
          },
        ],
        edges: [],
      },
    }
    const g = toGraphElements(art)
    expect(g.visNodes.find((v) => v.id === 'tc:g')?.label).toBeTruthy()
    expect(g.visNodes.find((v) => v.id === 'k:topic:a')?.label).toBe('')
  })

  it('keeps Topic canvas label when it differs from TopicCluster parent', () => {
    const art: ParsedArtifact = {
      name: 'x',
      kind: 'kg',
      episodeId: null,
      nodes: 2,
      edges: 0,
      nodeTypes: {},
      data: {
        nodes: [
          { id: 'tc:g', type: 'TopicCluster', properties: { label: 'Group' } },
          {
            id: 'k:topic:a',
            type: 'Topic',
            parent: 'tc:g',
            properties: { label: 'Other' },
          },
        ],
        edges: [],
      },
    }
    const g = toGraphElements(art)
    expect(g.visNodes.find((v) => v.id === 'k:topic:a')?.label).toContain('Other')
  })
})

describe('toCytoElements', () => {
  it('produces Cytoscape element definitions', () => {
    const elems = toCytoElements(parsedGi())
    expect(elems.length).toBeGreaterThan(0)
    const nodeElem = elems.find((e) => e.data.id === 'i1')
    expect(nodeElem).toBeTruthy()
    expect(nodeElem!.data.type).toBe('Insight')
    expect(typeof (nodeElem!.data as { shortLabel?: string }).shortLabel).toBe('string')
    expect(Number((nodeElem!.data as { recencyWeight?: number }).recencyWeight)).toBe(1)
  })

  it('canonicalises edge types for Cytoscape edgeType', () => {
    const elems = toCytoElements(parsedGi())
    const edge = elems.find((e) => 'source' in e.data && e.data.source === 'i1')
    expect(edge).toBeTruthy()
    expect((edge!.data as { edgeType?: string }).edgeType).toBe('SUPPORTED_BY')
  })

  it('includes data.parent when RawGraphNode has parent', () => {
    const art = parsedGi()
    const first = art.data.nodes?.find((n) => n?.id)
    expect(first).toBeTruthy()
    first!.parent = 'tc:test'
    const elems = toCytoElements(art)
    const hit = elems.find((e) => e.data.id === first!.id)
    expect(hit?.data.parent).toBe('tc:test')
  })

  // RFC-080 V2 — Insight grounding + confidence tier classes are
  // attached at element-build time so the stylesheet can hook them
  // (insight-confidence-high|medium|low + insight-ungrounded). Legacy
  // insights without confidence get no tier class — the default
  // Insight styling still applies.
  it('attaches insight-ungrounded class when grounded is false', () => {
    const art: ParsedArtifact = {
      ...parsedGi(),
    }
    const elems = toCytoElements(art)
    const ungroundedInsight = elems.find((e) => e.data.id === 'i2')
    expect(ungroundedInsight?.classes).toMatch(/\binsight-ungrounded\b/)
  })

  it('omits insight-ungrounded class when grounded is true or missing', () => {
    const elems = toCytoElements(parsedGi())
    const groundedInsight = elems.find((e) => e.data.id === 'i1')
    expect(groundedInsight?.classes ?? '').not.toMatch(/insight-ungrounded/)
  })

  it('attaches confidence tier class based on properties.confidence buckets', () => {
    const buildArt = (insightConf: number | undefined): ParsedArtifact =>
      parseArtifact('ep.gi.json', {
        episode_id: 'ep1',
        model_version: 'm',
        prompt_version: 'v',
        nodes: [
          {
            id: 'i:tier',
            type: 'Insight',
            properties: insightConf == null
              ? { text: 'no conf', episode_id: 'ep1' }
              : { text: 't', episode_id: 'ep1', confidence: insightConf },
          },
        ],
        edges: [],
      })
    const high = toCytoElements(buildArt(0.85)).find((e) => e.data.id === 'i:tier')
    const med = toCytoElements(buildArt(0.5)).find((e) => e.data.id === 'i:tier')
    const low = toCytoElements(buildArt(0.2)).find((e) => e.data.id === 'i:tier')
    const none = toCytoElements(buildArt(undefined)).find((e) => e.data.id === 'i:tier')
    expect(high?.classes).toMatch(/insight-confidence-high/)
    expect(med?.classes).toMatch(/insight-confidence-medium/)
    expect(low?.classes).toMatch(/insight-confidence-low/)
    // Legacy artifacts (no confidence): no tier class assigned.
    expect(none?.classes ?? '').not.toMatch(/insight-confidence-/)
  })
})

describe('confidenceTierFromInsightProperties (RFC-080 V2)', () => {
  it('buckets confidence at the documented thresholds', () => {
    expect(confidenceTierFromInsightProperties({ confidence: 1.0 })).toBe('high')
    expect(confidenceTierFromInsightProperties({ confidence: 0.7 })).toBe('high')
    expect(confidenceTierFromInsightProperties({ confidence: 0.69 })).toBe('medium')
    expect(confidenceTierFromInsightProperties({ confidence: 0.4 })).toBe('medium')
    expect(confidenceTierFromInsightProperties({ confidence: 0.39 })).toBe('low')
    expect(confidenceTierFromInsightProperties({ confidence: 0 })).toBe('low')
  })

  it('parses stringified numeric confidence', () => {
    expect(confidenceTierFromInsightProperties({ confidence: '0.8' })).toBe('high')
  })

  it('returns null for missing / non-finite confidence', () => {
    expect(confidenceTierFromInsightProperties(undefined)).toBeNull()
    expect(confidenceTierFromInsightProperties({})).toBeNull()
    expect(confidenceTierFromInsightProperties({ confidence: 'abc' })).toBeNull()
    expect(confidenceTierFromInsightProperties({ confidence: NaN })).toBeNull()
  })
})

describe('isInsightUngrounded (RFC-080 V2)', () => {
  it('returns true only when grounded === false', () => {
    expect(isInsightUngrounded({ grounded: false })).toBe(true)
    expect(isInsightUngrounded({ grounded: true })).toBe(false)
    // Missing / null / non-boolean → grounded by default (no warning border
    // for legacy artifacts predating the field).
    expect(isInsightUngrounded({})).toBe(false)
    expect(isInsightUngrounded(undefined)).toBe(false)
    expect(isInsightUngrounded({ grounded: null as unknown as boolean })).toBe(false)
  })
})

describe('applyGraphFilters topic cluster pruning', () => {
  it('removes TopicCluster nodes that have no remaining children after type filter', () => {
    const fullArt: ParsedArtifact = {
      name: 't',
      kind: 'kg',
      episodeId: null,
      nodes: 2,
      edges: 0,
      nodeTypes: {},
      data: {
        nodes: [
          { id: 'tc:1', type: 'TopicCluster', properties: { label: 'C' } },
          {
            id: 'k:topic:a',
            type: 'Topic',
            properties: { label: 'A' },
            parent: 'tc:1',
          },
        ],
        edges: [],
      },
    }
    const state: GraphFilterState = {
      allowedTypes: { TopicCluster: true, Topic: false },
      allowedEdgeTypes: {},
      hideUngroundedInsights: false,
      showGiLayer: true,
      showKgLayer: true,
    }
    const out = applyGraphFilters(fullArt, state)
    expect(out.data.nodes?.some((n) => String(n.id) === 'tc:1')).toBe(false)
    expect(out.data.nodes?.length).toBe(0)
  })
})
