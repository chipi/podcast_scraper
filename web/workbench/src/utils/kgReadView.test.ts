import { describe, expect, it } from 'vitest'

import type { ArtifactData } from '../types/artifact'
import { isKgV3, kgReadView } from './kgReadView'
import { parseArtifact } from './parsing'

function v2(): ArtifactData {
  return {
    schema_version: '2.0',
    episode_id: 'ep-1',
    extraction: { transcript_ref: 'transcripts/ep-1.txt' },
    nodes: [
      { id: 'podcast:p1', type: 'Podcast', properties: { title: 'Show' } },
      {
        id: 'episode:ep-1',
        type: 'Episode',
        properties: { title: 'One', podcast_id: 'p1', publish_date: '2025-07-03' },
      },
      { id: 'person:ana', type: 'Person', properties: { name: 'Ana' } },
    ],
    edges: [
      { type: 'HAS_EPISODE', from: 'podcast:p1', to: 'episode:ep-1' },
      { type: 'HOSTS', from: 'person:ana', to: 'podcast:p1' },
    ],
  } as ArtifactData
}

/** The same KG as R3b stores it (kg_as_v3 without adapted_from and the root episode_id). */
function v3(): ArtifactData {
  return {
    schema_version: '3.0',
    item_id: 'ep-1',
    content_type: 'podcast_episode',
    extraction: { document_ref: 'transcripts/ep-1.txt', variant: 'raw' },
    nodes: [
      { id: 'podcast:p1', type: 'Source', properties: { title: 'Show', source_kind: 'podcast' } },
      {
        id: 'episode:ep-1',
        type: 'ContentItem',
        properties: { title: 'One', feed_id: 'p1', published_at: '2025-07-03', source_id: 'podcast:p1' },
      },
      { id: 'person:ana', type: 'Person', properties: { name: 'Ana' } },
    ],
    edges: [
      { type: 'HAS_ITEM', from: 'podcast:p1', to: 'episode:ep-1' },
      { type: 'CONTRIBUTES_TO', from: 'person:ana', to: 'podcast:p1', properties: { role: 'host' } },
    ],
  } as unknown as ArtifactData
}

type Loose = Record<string, unknown>

describe('kgReadView: a KG 3.0 file reads like its KG 2.x form', () => {
  it('returns a 2.x file as is', () => {
    const a = v2()
    expect(isKgV3(a)).toBe(false)
    expect(kgReadView(a)).toBe(a)
  })

  it('never mutates the 3.0 file', () => {
    const a = v3()
    const before = structuredClone(a)
    kgReadView(a)
    expect(a).toEqual(before)
  })

  it('fills every 2.x name', () => {
    const old = v2()
    const view = kgReadView(v3()) as Loose & ArtifactData
    expect(view.episode_id).toBe('ep-1')
    expect((view.nodes ?? []).map((n) => n.type).sort()).toEqual((old.nodes ?? []).map((n) => n.type).sort())
    expect((view.edges ?? []).map((e) => e.type).sort()).toEqual((old.edges ?? []).map((e) => e.type).sort())
    expect((view.extraction as Loose).transcript_ref).toBe('transcripts/ep-1.txt')
    const ep = (view.nodes ?? []).find((n) => n.type === 'Episode')?.properties as Loose
    expect(ep).toMatchObject({ podcast_id: 'p1', publish_date: '2025-07-03' })
  })

  it('parseArtifact gives the 2.x answer for a 3.0 KG file', () => {
    const a = parseArtifact('ep-1.kg.json', v3())
    const b = parseArtifact('ep-1.kg.json', v2())
    expect(a.kind).toBe('kg')
    expect(a.episodeId).toBe(b.episodeId)
    expect(a.nodeTypes).toEqual(b.nodeTypes)
  })
})
