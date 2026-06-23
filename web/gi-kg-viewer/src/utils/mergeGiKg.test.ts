import { describe, expect, it } from 'vitest'
import type { ParsedArtifact } from '../types/artifact'
import { buildDisplayArtifact, combineGiKgParsedArtifacts, mergeGiKgFromArtifactArrays, mergeParsedArtifacts } from './mergeGiKg'
import { parseArtifact } from './parsing'

// ── fixtures ──

function giArt(episodeId = 'ep1', extra: string[] = []): ParsedArtifact {
  return parseArtifact(`${episodeId}.gi.json`, {
    episode_id: episodeId,
    model_version: 'gpt-4o',
    prompt_version: 'v2',
    nodes: [
      { id: `episode:${episodeId}`, type: 'Episode', properties: { title: episodeId } },
      { id: `i1-${episodeId}`, type: 'Insight', properties: { text: 'insight', grounded: true, episode_id: episodeId } },
      { id: `q1-${episodeId}`, type: 'Quote', properties: { text: 'quote' } },
      ...extra.map((x) => ({ id: x, type: 'Insight' as const })),
    ],
    edges: [{ type: 'supported_by', from: `i1-${episodeId}`, to: `q1-${episodeId}` }],
  })
}

function kgArt(episodeId = 'ep1'): ParsedArtifact {
  return parseArtifact(`${episodeId}.kg.json`, {
    episode_id: episodeId,
    extraction: { model_version: 'gpt-4o', extracted_at: '2026-01-01' },
    nodes: [
      { id: `episode:${episodeId}`, type: 'Episode', properties: { title: episodeId } },
      { id: `t1-${episodeId}`, type: 'Topic', properties: { label: 'climate' } },
      { id: `e1-${episodeId}`, type: 'Entity', properties: { entity_kind: 'person', label: 'Dr Smith' } },
    ],
    edges: [{ type: 'MENTIONS', from: `episode:${episodeId}`, to: `t1-${episodeId}` }],
  })
}

// ── mergeParsedArtifacts ──

describe('mergeParsedArtifacts', () => {
  it('returns null for fewer than 2 artifacts', () => {
    expect(mergeParsedArtifacts([giArt()])).toBeNull()
    expect(mergeParsedArtifacts([])).toBeNull()
  })

  it('returns null for mixed kinds', () => {
    expect(mergeParsedArtifacts([giArt(), kgArt()])).toBeNull()
  })

  it('merges two GI artifacts deduplicating nodes', () => {
    const a = giArt('ep1')
    const b = giArt('ep2')
    const merged = mergeParsedArtifacts([a, b])!
    expect(merged).not.toBeNull()
    expect(merged.kind).toBe('gi')
    expect(merged.nodes).toBe(a.nodes + b.nodes)
    expect(merged.name).toContain('Merged GI')
  })

  it('deduplicates nodes with same id', () => {
    const a = giArt('ep1', ['shared'])
    const b = giArt('ep2', ['shared'])
    const merged = mergeParsedArtifacts([a, b])!
    const sharedCount = (merged.data.nodes ?? []).filter((n) => String(n.id) === 'shared').length
    expect(sharedCount).toBe(1)
  })

  it('deduplicates edges', () => {
    const a = giArt('ep1')
    const merged = mergeParsedArtifacts([a, a])!
    const edgeTypes = (merged.data.edges ?? []).map((e) => e.type)
    const supportedBy = edgeTypes.filter((t) => t === 'supported_by')
    expect(supportedBy).toHaveLength(1)
  })

  it('aggregates sourceCorpusRelPathByEpisodeId for multi-file GI merge', () => {
    const p1 = 'feeds/show/run_a/metadata/e1.gi.json'
    const p2 = 'feeds/show/run_b/metadata/e2.gi.json'
    const a = parseArtifact('e1.gi.json', giArt('ep1').data, p1)
    const b = parseArtifact('e2.gi.json', giArt('ep2').data, p2)
    const merged = mergeParsedArtifacts([a, b])!
    expect(merged.sourceCorpusRelPath).toBeNull()
    expect(merged.sourceCorpusRelPathByEpisodeId).toEqual({
      ep1: p1,
      ep2: p2,
    })
  })
})

// ── combineGiKgParsedArtifacts ──

describe('combineGiKgParsedArtifacts', () => {
  it('returns null for wrong kinds', () => {
    expect(combineGiKgParsedArtifacts(kgArt(), giArt())).toBeNull()
  })

  it('combines GI + KG into "both"', () => {
    const combined = combineGiKgParsedArtifacts(giArt(), kgArt())!
    expect(combined).not.toBeNull()
    expect(combined.kind).toBe('both')
    expect(combined.name).toBe('Merged GI + KG')
  })

  it('unifies episode nodes with matching ids', () => {
    const combined = combineGiKgParsedArtifacts(giArt('ep1'), kgArt('ep1'))!
    const episodes = (combined.data.nodes ?? []).filter((n) => n.type === 'Episode')
    expect(episodes).toHaveLength(1)
    expect(String(episodes[0].id)).toContain('__unified_ep__')
  })

  it('preserves KG extraction metadata', () => {
    const combined = combineGiKgParsedArtifacts(giArt(), kgArt())!
    expect(combined.data.extraction).toBeTruthy()
  })

  it('passes through sourceCorpusRelPathByEpisodeId from merged GI', () => {
    const p1 = 'feeds/x/run1/metadata/a.gi.json'
    const p2 = 'feeds/x/run2/metadata/b.gi.json'
    const g1 = parseArtifact('a.gi.json', giArt('ep1').data, p1)
    const g2 = parseArtifact('b.gi.json', giArt('ep2').data, p2)
    const mergedGi = mergeParsedArtifacts([g1, g2])!
    const combined = combineGiKgParsedArtifacts(mergedGi, kgArt('ep1'))!
    expect(combined.sourceCorpusRelPathByEpisodeId).toEqual({
      ep1: p1,
      ep2: p2,
    })
  })

  it('prefixes node ids with g:/k:', () => {
    const combined = combineGiKgParsedArtifacts(giArt(), kgArt())!
    const ids = (combined.data.nodes ?? []).map((n) => String(n.id))
    const prefixed = ids.filter((id) => id.startsWith('g:') || id.startsWith('k:') || id.startsWith('__unified_ep__'))
    expect(prefixed.length).toBe(ids.length)
  })

  it('unifies episode nodes when GI uses ep: and KG uses kg:episode: prefix', () => {
    const uuid = '1689a44a-a66a-4008-aabd-7e745227f152'
    const gi = parseArtifact('ep1.gi.json', {
      episode_id: 'ep1',
      nodes: [
        { id: `ep:${uuid}`, type: 'Episode', properties: { title: 'Test' } },
        { id: 'i1', type: 'Insight', properties: { text: 'insight', grounded: true } },
      ],
      edges: [{ type: 'BELONGS_TO', from: 'i1', to: `ep:${uuid}` }],
    })
    const kg = parseArtifact('ep1.kg.json', {
      episode_id: 'ep1',
      extraction: { model_version: 'gpt-4o' },
      nodes: [
        { id: `kg:episode:${uuid}`, type: 'Episode', properties: { title: 'Test' } },
        { id: 't1', type: 'Topic', properties: { label: 'climate' } },
      ],
      edges: [{ type: 'MENTIONS', from: `kg:episode:${uuid}`, to: 't1' }],
    })
    const combined = combineGiKgParsedArtifacts(gi, kg)!
    expect(combined).not.toBeNull()
    const episodes = (combined.data.nodes ?? []).filter((n) => n.type === 'Episode')
    expect(episodes).toHaveLength(1)
    expect(String(episodes[0].id)).toBe(`__unified_ep__:${uuid}`)
  })

  it('unifies across 3 episodes with mixed ep:/kg:episode: prefixes', () => {
    const uuids = ['aaa', 'bbb', 'ccc']
    const giArts = uuids.map((u) =>
      parseArtifact(`${u}.gi.json`, {
        episode_id: u,
        nodes: [
          { id: `ep:${u}`, type: 'Episode', properties: { title: u } },
          { id: `i-${u}`, type: 'Insight', properties: { text: 'x' } },
        ],
        edges: [],
      }),
    )
    const kgArts = uuids.map((u) =>
      parseArtifact(`${u}.kg.json`, {
        episode_id: u,
        extraction: { model_version: 'v1' },
        nodes: [
          { id: `kg:episode:${u}`, type: 'Episode', properties: { title: u } },
          { id: `t-${u}`, type: 'Topic', properties: { label: u } },
        ],
        edges: [],
      }),
    )
    const result = mergeGiKgFromArtifactArrays(giArts, kgArts)!
    expect(result).not.toBeNull()
    const episodes = (result.data.nodes ?? []).filter((n) => n.type === 'Episode')
    expect(episodes).toHaveLength(3)
    for (const ep of episodes) {
      expect(String(ep.id)).toMatch(/^__unified_ep__:/)
    }
  })
})

// ── cross-episode entity/topic deduplication ──

describe('cross-episode entity deduplication', () => {
  it('deduplicates Entity nodes with the same name across episodes (KG merge)', () => {
    const kg1 = parseArtifact('ep1.kg.json', {
      episode_id: 'ep1',
      extraction: { model_version: 'v1' },
      nodes: [
        { id: 'kg:episode:aaa', type: 'Episode', properties: { title: 'Ep1' } },
        { id: 'kg:entity:aaa:0', type: 'Entity', properties: { name: 'Planet Money', entity_kind: 'organization' } },
        { id: 'kg:entity:aaa:1', type: 'Entity', properties: { name: 'NPR', entity_kind: 'organization' } },
      ],
      edges: [
        { type: 'MENTIONS', from: 'kg:entity:aaa:0', to: 'kg:episode:aaa' },
        { type: 'MENTIONS', from: 'kg:entity:aaa:1', to: 'kg:episode:aaa' },
      ],
    })
    const kg2 = parseArtifact('ep2.kg.json', {
      episode_id: 'ep2',
      extraction: { model_version: 'v1' },
      nodes: [
        { id: 'kg:episode:bbb', type: 'Episode', properties: { title: 'Ep2' } },
        { id: 'kg:entity:bbb:0', type: 'Entity', properties: { name: 'Planet Money', entity_kind: 'organization' } },
        { id: 'kg:entity:bbb:1', type: 'Entity', properties: { name: 'Alex Goldmark', entity_kind: 'person' } },
      ],
      edges: [
        { type: 'MENTIONS', from: 'kg:entity:bbb:0', to: 'kg:episode:bbb' },
        { type: 'MENTIONS', from: 'kg:entity:bbb:1', to: 'kg:episode:bbb' },
      ],
    })
    const merged = mergeParsedArtifacts([kg1, kg2])!
    expect(merged).not.toBeNull()

    const planetMoneyNodes = (merged.data.nodes ?? []).filter(
      (n) => n.type === 'Entity' && (n.properties as Record<string, unknown>)?.name === 'Planet Money',
    )
    expect(planetMoneyNodes).toHaveLength(1)

    const allEntities = (merged.data.nodes ?? []).filter((n) => n.type === 'Entity')
    expect(allEntities).toHaveLength(3)

    const edges = merged.data.edges ?? []
    const winnerId = String(planetMoneyNodes[0].id)
    const mentionsFromPM = edges.filter((e) => String(e.from) === winnerId)
    expect(mentionsFromPM).toHaveLength(2)
  })

  it('deduplicates Person nodes with the same name across episodes (GI merge)', () => {
    const gi1 = parseArtifact('ep1.gi.json', {
      episode_id: 'ep1',
      nodes: [
        { id: 'episode:ep1', type: 'Episode', properties: { title: 'Ep1' } },
        { id: 'person:dup-a', type: 'Person', properties: { name: 'Alice Example' } },
        { id: 'q1', type: 'Quote', properties: { text: 'hi' } },
      ],
      edges: [{ type: 'SPOKEN_BY', from: 'q1', to: 'person:dup-a' }],
    })
    const gi2 = parseArtifact('ep2.gi.json', {
      episode_id: 'ep2',
      nodes: [
        { id: 'episode:ep2', type: 'Episode', properties: { title: 'Ep2' } },
        { id: 'person:dup-b', type: 'Person', properties: { name: 'Alice Example' } },
        { id: 'q2', type: 'Quote', properties: { text: 'yo' } },
      ],
      edges: [{ type: 'SPOKEN_BY', from: 'q2', to: 'person:dup-b' }],
    })
    const merged = mergeParsedArtifacts([gi1, gi2])!
    expect(merged).not.toBeNull()

    const people = (merged.data.nodes ?? []).filter((n) => n.type === 'Person')
    expect(people).toHaveLength(1)
    const winnerId = String(people[0].id)

    const spoken = (merged.data.edges ?? []).filter((e) => e.type === 'SPOKEN_BY')
    expect(spoken).toHaveLength(2)
    for (const e of spoken) {
      expect(String(e.to)).toBe(winnerId)
    }
  })

  it('deduplicates Topic nodes with the same label across episodes (KG merge)', () => {
    const kg1 = parseArtifact('ep1.kg.json', {
      episode_id: 'ep1',
      extraction: { model_version: 'v1' },
      nodes: [
        { id: 'kg:episode:aaa', type: 'Episode', properties: { title: 'Ep1' } },
        { id: 'kg:topic:aaa:climate', type: 'Topic', properties: { label: 'Climate Change' } },
      ],
      edges: [{ type: 'MENTIONS', from: 'kg:topic:aaa:climate', to: 'kg:episode:aaa' }],
    })
    const kg2 = parseArtifact('ep2.kg.json', {
      episode_id: 'ep2',
      extraction: { model_version: 'v1' },
      nodes: [
        { id: 'kg:episode:bbb', type: 'Episode', properties: { title: 'Ep2' } },
        { id: 'kg:topic:bbb:climate', type: 'Topic', properties: { label: 'Climate Change' } },
        { id: 'kg:topic:bbb:ai', type: 'Topic', properties: { label: 'Artificial Intelligence' } },
      ],
      edges: [
        { type: 'MENTIONS', from: 'kg:topic:bbb:climate', to: 'kg:episode:bbb' },
        { type: 'MENTIONS', from: 'kg:topic:bbb:ai', to: 'kg:episode:bbb' },
      ],
    })
    const merged = mergeParsedArtifacts([kg1, kg2])!
    expect(merged).not.toBeNull()

    const climateNodes = (merged.data.nodes ?? []).filter(
      (n) => n.type === 'Topic' && ((n.properties as Record<string, unknown>)?.label as string)?.toLowerCase() === 'climate change',
    )
    expect(climateNodes).toHaveLength(1)

    const allTopics = (merged.data.nodes ?? []).filter((n) => n.type === 'Topic')
    expect(allTopics).toHaveLength(2)
  })

  it('deduplicates entities in GI+KG combine path across multiple episodes', () => {
    const uuids = ['aaa', 'bbb', 'ccc']
    const giArts = uuids.map((u) =>
      parseArtifact(`${u}.gi.json`, {
        episode_id: u,
        nodes: [
          { id: `ep:${u}`, type: 'Episode', properties: { title: u } },
          { id: `i-${u}`, type: 'Insight', properties: { text: 'x' } },
        ],
        edges: [],
      }),
    )
    const kgArts = uuids.map((u) =>
      parseArtifact(`${u}.kg.json`, {
        episode_id: u,
        extraction: { model_version: 'v1' },
        nodes: [
          { id: `kg:episode:${u}`, type: 'Episode', properties: { title: u } },
          { id: `kg:entity:${u}:pm`, type: 'Entity', properties: { name: 'Planet Money', entity_kind: 'organization' } },
          { id: `kg:topic:${u}:econ`, type: 'Topic', properties: { label: 'Economics' } },
        ],
        edges: [
          { type: 'MENTIONS', from: `kg:entity:${u}:pm`, to: `kg:episode:${u}` },
          { type: 'MENTIONS', from: `kg:topic:${u}:econ`, to: `kg:episode:${u}` },
        ],
      }),
    )
    const result = mergeGiKgFromArtifactArrays(giArts, kgArts)!
    expect(result).not.toBeNull()

    const episodes = (result.data.nodes ?? []).filter((n) => n.type === 'Episode')
    expect(episodes).toHaveLength(3)

    const entities = (result.data.nodes ?? []).filter((n) => n.type === 'Entity')
    expect(entities).toHaveLength(1)
    expect((entities[0].properties as Record<string, unknown>)?.name).toBe('Planet Money')

    const topics = (result.data.nodes ?? []).filter((n) => n.type === 'Topic')
    expect(topics).toHaveLength(1)

    const edges = result.data.edges ?? []
    const pmId = String(entities[0].id)
    const pmEdges = edges.filter((e) => String(e.from) === pmId && e.type === 'MENTIONS')
    expect(pmEdges).toHaveLength(3)
  })

  it('is case-insensitive for deduplication', () => {
    const kg1 = parseArtifact('ep1.kg.json', {
      episode_id: 'ep1',
      extraction: { model_version: 'v1' },
      nodes: [
        { id: 'kg:episode:aaa', type: 'Episode', properties: { title: 'Ep1' } },
        { id: 'e1', type: 'Entity', properties: { name: 'Planet Money' } },
      ],
      edges: [],
    })
    const kg2 = parseArtifact('ep2.kg.json', {
      episode_id: 'ep2',
      extraction: { model_version: 'v1' },
      nodes: [
        { id: 'kg:episode:bbb', type: 'Episode', properties: { title: 'Ep2' } },
        { id: 'e2', type: 'Entity', properties: { name: 'planet money' } },
      ],
      edges: [],
    })
    const merged = mergeParsedArtifacts([kg1, kg2])!
    const entities = (merged.data.nodes ?? []).filter((n) => n.type === 'Entity')
    expect(entities).toHaveLength(1)
  })

  it('does not deduplicate Insight or Quote nodes', () => {
    const gi1 = parseArtifact('ep1.gi.json', {
      episode_id: 'ep1',
      nodes: [
        { id: 'episode:ep1', type: 'Episode', properties: { title: 'Ep1' } },
        { id: 'i1', type: 'Insight', properties: { text: 'same insight' } },
      ],
      edges: [],
    })
    const gi2 = parseArtifact('ep2.gi.json', {
      episode_id: 'ep2',
      nodes: [
        { id: 'episode:ep2', type: 'Episode', properties: { title: 'Ep2' } },
        { id: 'i2', type: 'Insight', properties: { text: 'same insight' } },
      ],
      edges: [],
    })
    const merged = mergeParsedArtifacts([gi1, gi2])!
    const insights = (merged.data.nodes ?? []).filter((n) => n.type === 'Insight')
    expect(insights).toHaveLength(2)
  })
})

describe('GI+KG combine CIL id dedup', () => {
  it('merges GI Person and KG Entity with the same person: id', () => {
    const gi = parseArtifact('a.gi.json', {
      episode_id: 'ep',
      nodes: [
        { id: 'episode:x', type: 'Episode', properties: { title: 'E' } },
        { id: 'person:alice', type: 'Person', properties: { name: 'Alice' } },
      ],
      edges: [],
    })
    const kg = parseArtifact('a.kg.json', {
      episode_id: 'ep',
      extraction: { model_version: 'v1' },
      nodes: [
        { id: 'kg:episode:x', type: 'Episode', properties: { title: 'E' } },
        { id: 'person:alice', type: 'Entity', properties: { name: 'Alice', kind: 'person' } },
      ],
      edges: [],
    })
    const m = combineGiKgParsedArtifacts(gi, kg)!
    const withAlice = (m.data.nodes ?? []).filter((n) => /person:alice$/.test(String(n.id ?? '')))
    expect(withAlice).toHaveLength(1)
  })
})

// ── mergeGiKgFromArtifactArrays ──

describe('mergeGiKgFromArtifactArrays', () => {
  it('returns null when either array is empty', () => {
    expect(mergeGiKgFromArtifactArrays([], [kgArt()])).toBeNull()
    expect(mergeGiKgFromArtifactArrays([giArt()], [])).toBeNull()
  })

  it('combines single GI + single KG', () => {
    const result = mergeGiKgFromArtifactArrays([giArt()], [kgArt()])!
    expect(result.kind).toBe('both')
  })

  it('merges multiple GI then combines with KG', () => {
    const result = mergeGiKgFromArtifactArrays([giArt('ep1'), giArt('ep2')], [kgArt()])!
    expect(result.kind).toBe('both')
    expect(result.name).toContain('2 GI')
  })
})

// ── buildDisplayArtifact ──

describe('buildDisplayArtifact', () => {
  it('returns null for empty inputs', () => {
    expect(buildDisplayArtifact([], [])).toBeNull()
  })

  it('returns single GI as-is', () => {
    const gi = giArt()
    expect(buildDisplayArtifact([gi], [])).toBe(gi)
  })

  it('returns single KG as-is', () => {
    const kg = kgArt()
    expect(buildDisplayArtifact([], [kg])).toBe(kg)
  })

  it('merges two GI without KG', () => {
    const result = buildDisplayArtifact([giArt('ep1'), giArt('ep2')], [])!
    expect(result.kind).toBe('gi')
    expect(result.name).toContain('Merged')
  })

  it('combines GI + KG', () => {
    const result = buildDisplayArtifact([giArt()], [kgArt()])!
    expect(result.kind).toBe('both')
  })
})

// ── RFC-097 v3.0 typed Organization + MENTIONS family preservation ──

describe('RFC-097 v3.0 typed Organization dedup + MENTIONS family preservation', () => {
  it('deduplicates Organization nodes with the same name across episodes (KG merge)', () => {
    /** Parallel of the Person dedup test for v3.0 Organization (the first-
     * class typed node introduced in chunk 3). Without explicit handling
     * in ``DEDUP_TYPES``, the same Org referenced in N episodes would
     * surface as N nodes in the merged graph instead of one canonical
     * Organization with N MENTIONS_ORG edges.
     */
    const kg1 = parseArtifact('ep1.kg.json', {
      episode_id: 'ep1',
      extraction: { model_version: 'v1' },
      nodes: [
        { id: 'kg:episode:aaa', type: 'Episode', properties: { title: 'Ep1' } },
        { id: 'kg:org:aaa:acme', type: 'Organization', properties: { name: 'Acme Corp' } },
      ],
      edges: [{ type: 'MENTIONS_ORG', from: 'kg:org:aaa:acme', to: 'kg:episode:aaa' }],
    })
    const kg2 = parseArtifact('ep2.kg.json', {
      episode_id: 'ep2',
      extraction: { model_version: 'v1' },
      nodes: [
        { id: 'kg:episode:bbb', type: 'Episode', properties: { title: 'Ep2' } },
        { id: 'kg:org:bbb:acme', type: 'Organization', properties: { name: 'Acme Corp' } },
      ],
      edges: [{ type: 'MENTIONS_ORG', from: 'kg:org:bbb:acme', to: 'kg:episode:bbb' }],
    })
    const merged = mergeParsedArtifacts([kg1, kg2])!
    expect(merged).not.toBeNull()

    const acmeNodes = (merged.data.nodes ?? []).filter(
      (n) =>
        n.type === 'Organization' &&
        (n.properties as Record<string, unknown>)?.name === 'Acme Corp',
    )
    expect(acmeNodes).toHaveLength(1)

    // Both episodes' MENTIONS_ORG edges now point at the single canonical
    // Organization id (the typed-edge family is preserved through merge).
    const winnerId = String(acmeNodes[0].id)
    const orgEdges = (merged.data.edges ?? []).filter(
      (e) => e.type === 'MENTIONS_ORG' && String(e.from) === winnerId,
    )
    expect(orgEdges).toHaveLength(2)
  })

  it('preserves typed MENTIONS_PERSON edges through the merge', () => {
    const gi1 = parseArtifact('ep1.gi.json', {
      episode_id: 'ep1',
      nodes: [
        { id: 'episode:ep1', type: 'Episode', properties: { title: 'Ep1' } },
        { id: 'person:ada', type: 'Person', properties: { name: 'Ada Lovelace' } },
        {
          id: 'insight:1',
          type: 'Insight',
          properties: {
            text: 'About Ada',
            insight_type: 'claim',
            position_hint: 0.25,
          },
        },
      ],
      edges: [{ type: 'MENTIONS_PERSON', from: 'insight:1', to: 'person:ada' }],
    })
    const gi2 = parseArtifact('ep2.gi.json', {
      episode_id: 'ep2',
      nodes: [
        { id: 'episode:ep2', type: 'Episode', properties: { title: 'Ep2' } },
        { id: 'person:ada-2', type: 'Person', properties: { name: 'Ada Lovelace' } },
        {
          id: 'insight:2',
          type: 'Insight',
          properties: {
            text: 'More about Ada',
            insight_type: 'observation',
            position_hint: 0.7,
          },
        },
      ],
      edges: [{ type: 'MENTIONS_PERSON', from: 'insight:2', to: 'person:ada-2' }],
    })
    const merged = mergeParsedArtifacts([gi1, gi2])!
    expect(merged).not.toBeNull()

    // Person deduped to one canonical node.
    const people = (merged.data.nodes ?? []).filter(
      (n) =>
        n.type === 'Person' &&
        (n.properties as Record<string, unknown>)?.name === 'Ada Lovelace',
    )
    expect(people).toHaveLength(1)
    const winnerId = String(people[0].id)

    // Both typed MENTIONS_PERSON edges point at the canonical Person.
    const typedMentions = (merged.data.edges ?? []).filter(
      (e) => e.type === 'MENTIONS_PERSON' && String(e.to) === winnerId,
    )
    expect(typedMentions).toHaveLength(2)

    // insight_type + position_hint properties pass through unchanged.
    const insights = (merged.data.nodes ?? []).filter((n) => n.type === 'Insight')
    expect(insights).toHaveLength(2)
    const itypes = insights.map(
      (n) => (n.properties as Record<string, unknown>)?.insight_type,
    )
    expect(itypes.sort()).toEqual(['claim', 'observation'])
  })

  it('mid-migration corpus: typed MENTIONS_PERSON + legacy MENTIONS coexist after merge', () => {
    const gi1 = parseArtifact('ep1.gi.json', {
      episode_id: 'ep1',
      nodes: [
        { id: 'episode:ep1', type: 'Episode', properties: { title: 'Ep1' } },
        { id: 'person:linus', type: 'Person', properties: { name: 'Linus' } },
        { id: 'insight:typed', type: 'Insight', properties: { text: 'typed' } },
      ],
      edges: [{ type: 'MENTIONS_PERSON', from: 'insight:typed', to: 'person:linus' }],
    })
    const gi2 = parseArtifact('ep2.gi.json', {
      episode_id: 'ep2',
      nodes: [
        { id: 'episode:ep2', type: 'Episode', properties: { title: 'Ep2' } },
        { id: 'person:linus-2', type: 'Person', properties: { name: 'Linus' } },
        { id: 'insight:legacy', type: 'Insight', properties: { text: 'legacy' } },
      ],
      edges: [{ type: 'MENTIONS', from: 'insight:legacy', to: 'person:linus-2' }],
    })
    const merged = mergeParsedArtifacts([gi1, gi2])!
    expect(merged).not.toBeNull()

    // Both edge types coexist post-merge — no silent collapse.
    const types = (merged.data.edges ?? []).map((e) => e.type).sort()
    expect(types).toContain('MENTIONS_PERSON')
    expect(types).toContain('MENTIONS')
  })
})
