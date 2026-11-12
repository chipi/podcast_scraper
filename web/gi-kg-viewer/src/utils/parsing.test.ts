import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import type { ArtifactData, GraphFilterState, ParsedArtifact } from '../types/artifact'
import {
  applyGraphDefaultNodeTypeVisibility,
  applyGraphFilters,
  buildNodeTitle,
  countPersonEntityIncidentEdges,
  defaultFilterState,
  entityDisplayNameFromId,
  findEpisodeGraphNodeIdForEpisodeKey,
  quoteAttributionDisplayFromId,
  filterArtifactEgoAroundCompound,
  filterArtifactEgoOneHop,
  filtersActive,
  filtersActiveExcludingNodeTypes,
  graphTypesDeviateFromGraphSpec,
  findRawNodeInArtifact,
  fullPrimaryNodeLabel,
  canonicalTopicKey,
  nodeLabel,
  personEpisodeAppearances,
  personRoleFromNode,
  primaryTextFromLooseGiNode,
  parseArtifact,
  rankedPersonOrganizations,
  rankedPersonTopicMentions,
  toCytoElements,
  toGraphElements,
} from './parsing'
import { graphTypeTraits } from '../extensions'

// The traits a GI build registers (E04 step 8b), so the viewer's generic type mechanics are
// exercised with the types they were written for.
beforeEach(() => graphTypeTraits.register({ id: 'gi', defaultHidden: ['Quote', 'Speaker'] }))
afterEach(() => graphTypeTraits.reset())

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
  return parseArtifact('ep1.kg.json', giData())
}

// ── parseArtifact ──

describe('parseArtifact', () => {
  it('reads a .gi.json as unknown (GI is a downstream build\'s reader)', () => {
    const art = parseArtifact('ep1.gi.json', { nodes: [], edges: [] })
    expect(art.kind).toBe('unknown')
  })

  it('detects KG from filename', () => {
    const art = parseArtifact('ep1.kg.json', { nodes: [], edges: [] })
    expect(art.kind).toBe('kg')
  })

  it('does not infer GI from model_version + prompt_version', () => {
    const art = parseArtifact('data.json', { model_version: 'x', prompt_version: 'y', nodes: [], edges: [] })
    expect(art.kind).toBe('unknown')
  })

  it('infers KG from extraction field', () => {
    const art = parseArtifact('data.json', { extraction: { model_version: 'x' }, nodes: [], edges: [] })
    expect(art.kind).toBe('kg')
  })

  it('counts nodes and edges', () => {
    const art = parsedGi()
    expect(art.nodes).toBe(4)
    expect(art.edges).toBe(1)
  })

  it('builds nodeTypes map', () => {
    const art = parsedGi()
    expect(art.nodeTypes.Episode).toBe(1)
    expect(art.nodeTypes.Insight).toBe(2)
  })

  it('adds no edges of its own (the data is drawn as it is)', () => {
    const art = parsedGi()
    expect((art.data.edges ?? []).map((e) => e.type)).toEqual(['supported_by'])
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
    expect(state.allowedEdgeTypes.supported_by).toBe(true)
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
    // Insight defaults to visible; disabling it is a real deviation.
    // (Quote is default-hidden per tier 6-3, so setting Quote=false is
    // not a deviation and does not trigger the "filters active" indicator.)
    state.allowedTypes.Insight = false
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
    // Disable Insight (default-visible) so allowedTypes deviates.
    state.allowedTypes.Insight = false
    expect(filtersActive(art, state)).toBe(true)
    expect(filtersActiveExcludingNodeTypes(art, state)).toBe(false)
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

  // #658 — feed filter prunes Episode nodes by ``properties.feed_id`` and
  // drops non-Episode nodes that lose all their connections to surviving
  // Episodes (no orphan Insights / Quotes left dangling).
  describe('feed filter (graphFeedFilterId)', () => {
    function multiFeedArt(): ParsedArtifact {
      return {
        name: 'multi',
        kind: 'gi',
        episodeId: null,
        nodes: 0,
        edges: 0,
        nodeTypes: {},
        data: {
          nodes: [
            { id: 'epA', type: 'Episode', properties: { feed_id: 'feed-A' } },
            { id: 'epB', type: 'Episode', properties: { feed_id: 'feed-B' } },
            { id: 'iA', type: 'Insight', properties: { grounded: true, episode_id: 'epA' } },
            { id: 'iB', type: 'Insight', properties: { grounded: true, episode_id: 'epB' } },
            { id: 'tShared', type: 'Topic' },
          ],
          edges: [
            { type: 'HAS_INSIGHT', from: 'epA', to: 'iA' },
            { type: 'HAS_INSIGHT', from: 'epB', to: 'iB' },
            { type: 'ABOUT', from: 'iA', to: 'tShared' },
            { type: 'ABOUT', from: 'iB', to: 'tShared' },
          ],
        },
      }
    }

    it('returns the full artifact when graphFeedFilterId is null', () => {
      const art = multiFeedArt()
      const state = defaultFilterState(art)!
      expect(state.graphFeedFilterId).toBeNull()
      const filtered = applyGraphFilters(art, state)
      const ids = (filtered.data.nodes ?? []).map((n) => String(n.id))
      expect(ids).toEqual(expect.arrayContaining(['epA', 'epB', 'iA', 'iB', 'tShared']))
    })

    it('keeps only Episode nodes whose properties.feed_id matches', () => {
      const art = multiFeedArt()
      const state = defaultFilterState(art)!
      state.graphFeedFilterId = 'feed-A'
      const filtered = applyGraphFilters(art, state)
      const ids = (filtered.data.nodes ?? []).map((n) => String(n.id))
      expect(ids).toContain('epA')
      expect(ids).not.toContain('epB')
    })

    it('keeps non-Episode nodes that have any path to a surviving Episode (BFS reachability)', () => {
      const art = multiFeedArt()
      const state = defaultFilterState(art)!
      state.graphFeedFilterId = 'feed-A'
      const filtered = applyGraphFilters(art, state)
      const ids = (filtered.data.nodes ?? []).map((n) => String(n.id))
      // iA and tShared both reach epA via the edge graph
      expect(ids).toContain('iA')
      expect(ids).toContain('tShared')
      // iB still has a transitive path epA → iA → tShared → iB so under
      // BFS connectivity it stays. (A stricter "owner-episode" semantic
      // would drop iB; the issue specifies path-based connectivity.)
      expect(ids).toContain('iB')
    })

    it('prunes orphan non-Episode nodes when ALL their Episode neighbours are filtered out', () => {
      const art = multiFeedArt()
      // make tShared connect ONLY to epB
      art.data.edges = [
        { type: 'HAS_INSIGHT', from: 'epA', to: 'iA' },
        { type: 'HAS_INSIGHT', from: 'epB', to: 'iB' },
        { type: 'ABOUT', from: 'iB', to: 'tShared' },
      ]
      const state = defaultFilterState(art)!
      state.graphFeedFilterId = 'feed-A'
      const filtered = applyGraphFilters(art, state)
      const ids = (filtered.data.nodes ?? []).map((n) => String(n.id))
      expect(ids).toContain('epA')
      expect(ids).toContain('iA')
      expect(ids).not.toContain('tShared') // dropped — only neighbour was epB
    })

    it('treats __unified_ep__: synthetic ids as Episode nodes for feed matching', () => {
      const art: ParsedArtifact = {
        name: 'merged',
        kind: 'both',
        episodeId: null,
        nodes: 0,
        edges: 0,
        nodeTypes: {},
        data: {
          nodes: [
            { id: '__unified_ep__:epA', type: 'Episode', properties: { feed_id: 'feed-A' } },
            { id: '__unified_ep__:epB', type: 'Episode', properties: { feed_id: 'feed-B' } },
            { id: 'g:i1', type: 'Insight', properties: { grounded: true } },
          ],
          edges: [{ type: 'HAS_INSIGHT', from: '__unified_ep__:epA', to: 'g:i1' }],
        },
      }
      const state = defaultFilterState(art)!
      state.graphFeedFilterId = 'feed-A'
      const filtered = applyGraphFilters(art, state)
      const ids = (filtered.data.nodes ?? []).map((n) => String(n.id))
      expect(ids).toContain('__unified_ep__:epA')
      expect(ids).not.toContain('__unified_ep__:epB')
      expect(ids).toContain('g:i1')
    })

    it('filtersActive returns true when graphFeedFilterId is non-null', () => {
      const art = multiFeedArt()
      const state = defaultFilterState(art)!
      expect(filtersActive(art, state)).toBe(false)
      state.graphFeedFilterId = 'feed-A'
      expect(filtersActive(art, state)).toBe(true)
      expect(filtersActiveExcludingNodeTypes(art, state)).toBe(true)
    })
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

// #1048 — Person Landing shared component helpers.
describe('rankedPersonTopicMentions', () => {
  it('returns [] for missing artifact or id', () => {
    expect(rankedPersonTopicMentions(null, 'person:x')).toEqual([])
    const art = parseArtifact('x.gi.json', { nodes: [], edges: [] })
    expect(rankedPersonTopicMentions(art, null)).toEqual([])
  })

  it('ranks topics by collapsed (insight, topic) pairs, alphabetic tiebreak', () => {
    const art = parseArtifact('x.gi.json', {
      nodes: [
        { id: 'topic:ai', type: 'Topic', properties: { name: 'AI ethics' } },
        { id: 'topic:reg', type: 'Topic', properties: { name: 'AI regulation' } },
      ],
      edges: [
        { type: 'MENTIONS_PERSON', from: 'i1', to: 'person:a' },
        { type: 'MENTIONS_PERSON', from: 'i2', to: 'person:a' },
        { type: 'MENTIONS_PERSON', from: 'i3', to: 'person:a' },
        // i1 → topic:ai counted twice in edges but collapses to 1 pair
        { type: 'ABOUT', from: 'i1', to: 'topic:ai' },
        { type: 'ABOUT', from: 'i1', to: 'topic:ai' },
        { type: 'ABOUT', from: 'i2', to: 'topic:ai' }, // distinct pair (i2, ai) = 1 more
        { type: 'ABOUT', from: 'i3', to: 'topic:reg' },
      ],
    })
    expect(rankedPersonTopicMentions(art, 'person:a')).toEqual([
      { id: 'topic:ai', name: 'AI ethics', count: 2 },
      { id: 'topic:reg', name: 'AI regulation', count: 1 },
    ])
  })

  it('falls back to id when Topic node lacks a name/label', () => {
    const art = parseArtifact('x.gi.json', {
      nodes: [{ id: 'topic:unknown', type: 'Topic', properties: {} }],
      edges: [
        { type: 'MENTIONS_PERSON', from: 'i1', to: 'person:a' },
        { type: 'ABOUT', from: 'i1', to: 'topic:unknown' },
      ],
    })
    expect(rankedPersonTopicMentions(art, 'person:a')).toEqual([
      { id: 'topic:unknown', name: 'topic:unknown', count: 1 },
    ])
  })

  it('skips ABOUT edges whose Insight does not MENTIONS_PERSON this person', () => {
    const art = parseArtifact('x.gi.json', {
      nodes: [{ id: 'topic:ai', type: 'Topic', properties: { name: 'AI' } }],
      edges: [
        { type: 'MENTIONS_PERSON', from: 'i1', to: 'person:other' },
        { type: 'ABOUT', from: 'i1', to: 'topic:ai' },
      ],
    })
    expect(rankedPersonTopicMentions(art, 'person:a')).toEqual([])
  })
})

describe('rankedPersonOrganizations', () => {
  it('returns [] when no MENTIONS_PERSON edges target the person', () => {
    const art = parseArtifact('x.gi.json', {
      nodes: [],
      edges: [{ type: 'MENTIONS_ORG', from: 'i1', to: 'org:openai' }],
    })
    expect(rankedPersonOrganizations(art, 'person:a')).toEqual([])
  })

  it('ranks co-mentioned orgs via MENTIONS_PERSON ∩ MENTIONS_ORG on the same Insight', () => {
    const art = parseArtifact('x.gi.json', {
      nodes: [
        { id: 'org:openai', type: 'Organization', properties: { name: 'OpenAI' } },
        { id: 'org:google', type: 'Organization', properties: { name: 'Google' } },
      ],
      edges: [
        { type: 'MENTIONS_PERSON', from: 'i1', to: 'person:a' },
        { type: 'MENTIONS_PERSON', from: 'i2', to: 'person:a' },
        { type: 'MENTIONS_ORG', from: 'i1', to: 'org:openai' },
        { type: 'MENTIONS_ORG', from: 'i2', to: 'org:openai' },
        { type: 'MENTIONS_ORG', from: 'i1', to: 'org:google' },
      ],
    })
    expect(rankedPersonOrganizations(art, 'person:a')).toEqual([
      { id: 'org:openai', name: 'OpenAI', count: 2 },
      { id: 'org:google', name: 'Google', count: 1 },
    ])
  })
})

describe('canonicalTopicKey (#1076 chunk 2)', () => {
  it('lowercases', () => {
    expect(canonicalTopicKey('AI Ethics')).toBe('ai ethics')
  })
  it('strips ASCII punctuation', () => {
    expect(canonicalTopicKey('AI-Ethics')).toBe('ai ethics')
    expect(canonicalTopicKey('AI/Ethics?')).toBe('ai ethics')
  })
  it('collapses whitespace and trims', () => {
    expect(canonicalTopicKey('  AI   ethics  ')).toBe('ai ethics')
  })
  it('returns empty string for non-strings', () => {
    expect(canonicalTopicKey(null)).toBe('')
    expect(canonicalTopicKey(undefined)).toBe('')
  })
  it('folds three near-duplicates to a single canonical key', () => {
    const key1 = canonicalTopicKey('AI ethics')
    const key2 = canonicalTopicKey('AI Ethics')
    const key3 = canonicalTopicKey('AI-ethics')
    expect(key1).toBe(key2)
    expect(key2).toBe(key3)
  })
})

describe('rankedPersonTopicMentions — near-duplicate fold (#1076 chunk 2)', () => {
  it('groups Topic rows with the same canonical key under the alphabetically-smallest id', () => {
    const art = parseArtifact('x.gi.json', {
      nodes: [
        // Three near-duplicate Topics, alphabetically t:ai-1 < t:ai-2 < t:ai-3.
        { id: 't:ai-1', type: 'Topic', properties: { name: 'AI ethics' } },
        { id: 't:ai-2', type: 'Topic', properties: { name: 'AI Ethics' } },
        { id: 't:ai-3', type: 'Topic', properties: { name: 'AI-Ethics' } },
        { id: 't:other', type: 'Topic', properties: { name: 'Climate' } },
      ],
      edges: [
        { type: 'MENTIONS_PERSON', from: 'i1', to: 'p:a' },
        { type: 'MENTIONS_PERSON', from: 'i2', to: 'p:a' },
        { type: 'MENTIONS_PERSON', from: 'i3', to: 'p:a' },
        { type: 'MENTIONS_PERSON', from: 'i4', to: 'p:a' },
        { type: 'ABOUT', from: 'i1', to: 't:ai-1' },
        { type: 'ABOUT', from: 'i2', to: 't:ai-2' },
        { type: 'ABOUT', from: 'i3', to: 't:ai-3' },
        { type: 'ABOUT', from: 'i4', to: 't:other' },
      ],
    })
    const rows = rankedPersonTopicMentions(art, 'p:a')
    expect(rows.length).toBe(2)
    // Folded row uses the smallest id (t:ai-1) + sums counts (3 across the 3 near-dups).
    const ai = rows.find((r) => r.name === 'AI ethics')
    expect(ai).toBeDefined()
    expect(ai!.id).toBe('t:ai-1')
    expect(ai!.count).toBe(3)
    const other = rows.find((r) => r.name === 'Climate')
    expect(other!.count).toBe(1)
  })

  it('does not fold topics with no extractable name (empty canonical key)', () => {
    // Topics fall back to id when name is missing → canonical key on
    // the id includes the colon punctuation stripped → may collide
    // BUT the helper uses the FALLBACK name (= id) for keying which
    // contains the colon. canonicalTopicKey strips ":" so the ids
    // would collide. Confirm distinct Topics with id-fallback names
    // still surface as distinct rows.
    const art = parseArtifact('x.gi.json', {
      nodes: [
        { id: 't:foo', type: 'Topic', properties: {} },
        { id: 't:bar', type: 'Topic', properties: {} },
      ],
      edges: [
        { type: 'MENTIONS_PERSON', from: 'i1', to: 'p:a' },
        { type: 'MENTIONS_PERSON', from: 'i2', to: 'p:a' },
        { type: 'ABOUT', from: 'i1', to: 't:foo' },
        { type: 'ABOUT', from: 'i2', to: 't:bar' },
      ],
    })
    const rows = rankedPersonTopicMentions(art, 'p:a')
    // Both Topics surface (they have different canonical keys via id).
    expect(rows.length).toBe(2)
    const ids = rows.map((r) => r.id).sort()
    expect(ids).toEqual(['t:bar', 't:foo'])
  })
})

describe('personEpisodeAppearances (#1050)', () => {
  it('returns [] for missing artifact or id', () => {
    expect(personEpisodeAppearances(null, 'p1')).toEqual([])
    const art = parseArtifact('x.gi.json', { nodes: [], edges: [] })
    expect(personEpisodeAppearances(art, null)).toEqual([])
  })

  it('lists SPOKE_IN targets with publish_date sorted desc; undated sink to bottom', () => {
    const art = parseArtifact('x.gi.json', {
      nodes: [
        { id: 'ep:new', type: 'Episode', properties: { episode_title: 'Newer', publish_date: '2026-03-15' } },
        { id: 'ep:old', type: 'Episode', properties: { episode_title: 'Older', publish_date: '2026-01-10' } },
        { id: 'ep:undated', type: 'Episode', properties: { episode_title: 'No date' } },
      ],
      edges: [
        { type: 'SPOKE_IN', from: 'p:alice', to: 'ep:old' },
        { type: 'SPOKE_IN', from: 'p:alice', to: 'ep:new' },
        { type: 'SPOKE_IN', from: 'p:alice', to: 'ep:undated' },
        // distractor
        { type: 'SPOKE_IN', from: 'p:bob', to: 'ep:new' },
      ],
    })
    const rows = personEpisodeAppearances(art, 'p:alice')
    expect(rows.map((r) => r.episodeId)).toEqual(['ep:new', 'ep:old', 'ep:undated'])
    expect(rows[0].title).toBe('Newer')
    expect(rows[0].publishDate).toBe('2026-03-15')
    expect(rows[2].publishDate).toBeNull()
  })

  it('collapses duplicate SPOKE_IN edges to the same episode', () => {
    const art = parseArtifact('x.gi.json', {
      nodes: [{ id: 'ep:1', type: 'Episode', properties: { publish_date: '2026-04-01' } }],
      edges: [
        { type: 'SPOKE_IN', from: 'p:alice', to: 'ep:1' },
        { type: 'SPOKE_IN', from: 'p:alice', to: 'ep:1' },
      ],
    })
    expect(personEpisodeAppearances(art, 'p:alice').length).toBe(1)
  })

  it('falls back to title then id when episode_title is missing', () => {
    const art = parseArtifact('x.gi.json', {
      nodes: [
        { id: 'ep:1', type: 'Episode', properties: { title: 'Alt title' } },
        { id: 'ep:2', type: 'Episode', properties: {} },
      ],
      edges: [
        { type: 'SPOKE_IN', from: 'p:a', to: 'ep:1' },
        { type: 'SPOKE_IN', from: 'p:a', to: 'ep:2' },
      ],
    })
    const rows = personEpisodeAppearances(art, 'p:a')
    const byId = new Map(rows.map((r) => [r.episodeId, r]))
    expect(byId.get('ep:1')!.title).toBe('Alt title')
    expect(byId.get('ep:2')!.title).toBeNull()
  })
})

describe('personRoleFromNode', () => {
  it('returns null for null node or missing role', () => {
    expect(personRoleFromNode(null)).toBeNull()
    expect(personRoleFromNode({ id: 'p1', type: 'Person', properties: {} })).toBeNull()
  })

  it('lowercases and trims the role string', () => {
    expect(
      personRoleFromNode({ id: 'p1', type: 'Person', properties: { role: '  Host  ' } }),
    ).toBe('host')
  })

  it('returns null for non-string role values', () => {
    expect(
      personRoleFromNode({ id: 'p1', type: 'Person', properties: { role: 7 } }),
    ).toBeNull()
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

})

describe('toCytoElements — Episode publishDate (RFC-080 V3)', () => {
  // Episode nodes carry `publishDate` (Unix ms) so the timeline layout can order them.
  it('attaches publishDate (Unix ms) on Episode nodes when publish_date is parseable', () => {
    const art = parseArtifact('p.gi.json', {
      episode_id: 'e1',
      model_version: 'm',
      prompt_version: 'v',
      nodes: [
        {
          id: 'episode:e1',
          type: 'Episode',
          properties: { title: 'E1', publish_date: '2026-04-17T00:00:00Z' },
        },
      ],
      edges: [],
    })
    const ep = toCytoElements(art).find((e) => e.data.id === 'episode:e1')
    const pd = (ep?.data as { publishDate?: number }).publishDate
    expect(typeof pd).toBe('number')
    expect(pd).toBe(Date.parse('2026-04-17T00:00:00Z'))
  })

  it('omits publishDate when publish_date is missing or unparseable', () => {
    const art = parseArtifact('np.gi.json', {
      episode_id: 'e1',
      model_version: 'm',
      prompt_version: 'v',
      nodes: [
        { id: 'episode:e1', type: 'Episode', properties: {} },
        { id: 'episode:e2', type: 'Episode', properties: { publish_date: 'not-a-date' } },
      ],
      edges: [],
    })
    const elems = toCytoElements(art)
    expect((elems.find((e) => e.data.id === 'episode:e1')?.data as { publishDate?: number }).publishDate).toBeUndefined()
    expect((elems.find((e) => e.data.id === 'episode:e2')?.data as { publishDate?: number }).publishDate).toBeUndefined()
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
    }
    const out = applyGraphFilters(fullArt, state)
    expect(out.data.nodes?.some((n) => String(n.id) === 'tc:1')).toBe(false)
    expect(out.data.nodes?.length).toBe(0)
  })
})

describe('filterArtifactEgoAroundCompound', () => {
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
          { id: 'tc:g', type: 'Group', properties: { label: 'G' } },
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
    const sub = filterArtifactEgoAroundCompound(art, 'tc:g', ['k:topic:a', 'k:topic:b'])
    const ids = (sub.data.nodes ?? []).map((n) => String(n.id)).sort()
    expect(ids).toEqual(['ep:1', 'ep:2', 'k:topic:a', 'k:topic:b', 'tc:g'])
    expect(sub.data.edges?.length).toBe(2)
  })
})
