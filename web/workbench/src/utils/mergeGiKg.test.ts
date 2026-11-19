import { describe, expect, it } from 'vitest'
import type { ParsedArtifact } from '../types/artifact'
import { buildDisplayArtifact, mergeParsedArtifacts } from './mergeGiKg'
import { parseArtifact } from './parsing'

// ── fixtures ──

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

/** A KG artifact with extra Topic nodes (ids shared across artifacts dedupe). */
function kgArtWith(episodeId: string, extra: string[]): ParsedArtifact {
  const art = kgArt(episodeId)
  const nodes = [...(art.data.nodes ?? []), ...extra.map((x) => ({ id: x, type: 'Topic' as const }))]
  return parseArtifact(`${episodeId}.kg.json`, { ...art.data, nodes })
}

// ── mergeParsedArtifacts ──

describe('mergeParsedArtifacts', () => {
  it('returns null for fewer than 2 artifacts', () => {
    expect(mergeParsedArtifacts([kgArt()])).toBeNull()
    expect(mergeParsedArtifacts([])).toBeNull()
  })

  it('returns null for mixed kinds', () => {
    expect(mergeParsedArtifacts([kgArt(), { ...kgArt('ep2'), kind: 'gi' }])).toBeNull()
  })

  it('merges two KG artifacts', () => {
    const a = kgArt('ep1')
    const b = kgArt('ep2')
    const merged = mergeParsedArtifacts([a, b])!
    expect(merged.kind).toBe('kg')
    expect(merged.name).toContain('Merged KG')
    expect((merged.data.nodes ?? []).length).toBeGreaterThanOrEqual(a.nodes)
  })

  it('deduplicates nodes with same id', () => {
    const merged = mergeParsedArtifacts([kgArtWith('ep1', ['shared']), kgArtWith('ep2', ['shared'])])!
    expect((merged.data.nodes ?? []).filter((n) => String(n.id) === 'shared')).toHaveLength(1)
  })

  it('deduplicates edges', () => {
    const a = kgArt('ep1')
    const merged = mergeParsedArtifacts([a, a])!
    expect((merged.data.edges ?? []).filter((e) => e.type === 'MENTIONS')).toHaveLength(1)
  })

  it('aggregates sourceCorpusRelPathByEpisodeId for a multi-file merge', () => {
    const p1 = 'feeds/show/run_a/metadata/e1.kg.json'
    const p2 = 'feeds/show/run_b/metadata/e2.kg.json'
    const a = parseArtifact('e1.kg.json', kgArt('ep1').data, p1)
    const b = parseArtifact('e2.kg.json', kgArt('ep2').data, p2)
    const merged = mergeParsedArtifacts([a, b])!
    expect(merged.sourceCorpusRelPath).toBeNull()
    expect(merged.sourceCorpusRelPathByEpisodeId).toEqual({ ep1: p1, ep2: p2 })
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

})

// ── buildDisplayArtifact ──

describe('buildDisplayArtifact', () => {
  it('returns null for no artifacts', () => {
    expect(buildDisplayArtifact([])).toBeNull()
  })

  it('returns a single KG as is', () => {
    const kg = kgArt()
    expect(buildDisplayArtifact([kg])).toBe(kg)
  })

  it('merges several KG', () => {
    const result = buildDisplayArtifact([kgArt('ep1'), kgArt('ep2')])!
    expect(result.kind).toBe('kg')
    expect(result.name).toContain('Merged')
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

})
