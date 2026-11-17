import { describe, expect, it } from 'vitest'

import type { ArtifactData } from '../types/artifact'
import { giReadView, isGiV4 } from './giReadView'
import { parseArtifact } from './parsing'

/** One episode in GI 3.1, with the Podcast node and HAS_EPISODE edge the edge pass adds. */
function v3(): ArtifactData {
  return {
    schema_version: '3.1',
    episode_id: 'ep-1',
    nodes: [
      { id: 'podcast:the-show', type: 'Podcast', properties: { title: 'The Show' } },
      {
        id: 'episode:ep-1',
        type: 'Episode',
        properties: { title: 'One', publish_date: '2025-07-03T00:00:00', podcast_id: 'p05' },
      },
      {
        id: 'insight:1',
        type: 'Insight',
        properties: { text: 'A claim', episode_id: 'ep-1', speaker: 'Ana' },
      },
      {
        id: 'quote:1',
        type: 'Quote',
        properties: {
          text: 'said it',
          episode_id: 'ep-1',
          speaker_id: 'person:ana',
          speaker_name: 'Ana',
          transcript_ref: 'transcripts/ep-1.txt',
          timestamp_start_ms: 1000,
          timestamp_end_ms: 4000,
        },
      },
    ],
    edges: [
      { type: 'HAS_EPISODE', from: 'podcast:the-show', to: 'episode:ep-1' },
      { type: 'SUPPORTED_BY', from: 'insight:1', to: 'quote:1' },
    ],
  } as ArtifactData
}

/** The same episode as R3b writes it (Common ``as_v4_file``): no 3.1 name left. */
function v4(): ArtifactData {
  return {
    schema_version: '4.0',
    item_id: 'ep-1',
    content_type: 'podcast_episode',
    nodes: [
      { id: 'podcast:p05', type: 'Source', properties: { title: 'The Show' } },
      {
        id: 'episode:ep-1',
        type: 'ContentItem',
        properties: {
          title: 'One',
          published_at: '2025-07-03T00:00:00',
          source_id: 'podcast:p05',
          podcast_episode: { podcast_id: 'p05' },
        },
      },
      {
        id: 'insight:1',
        type: 'Insight',
        properties: { text: 'A claim', item_id: 'ep-1', podcast_episode: { speaker: 'Ana' } },
      },
      {
        id: 'quote:1',
        type: 'Quote',
        properties: {
          text: 'said it',
          item_id: 'ep-1',
          document_ref: 'transcripts/ep-1.txt',
          start_ms: 1000,
          end_ms: 4000,
          podcast_episode: { speaker_id: 'person:ana', speaker_name: 'Ana' },
        },
      },
    ],
    edges: [
      { type: 'HAS_ITEM', from: 'podcast:p05', to: 'episode:ep-1' },
      { type: 'SUPPORTED_BY', from: 'insight:1', to: 'quote:1' },
    ],
  } as ArtifactData
}

type Loose = Record<string, unknown>
const props = (art: ArtifactData, id: string): Loose =>
  ((art.nodes ?? []).find((n) => n.id === id)?.properties ?? {}) as Loose

describe('giReadView: a GI 4.0 file reads like its GI 3.1 form', () => {
  it('returns a 3.1 file as is', () => {
    const a = v3()
    expect(isGiV4(a)).toBe(false)
    expect(giReadView(a)).toBe(a)
  })

  it('never mutates the 4.0 file', () => {
    const a = v4()
    const before = structuredClone(a)
    giReadView(a)
    expect(a).toEqual(before)
  })

  it('fills every 3.1 name the 3.1 form has', () => {
    const old = v3()
    const view = giReadView(v4())
    expect((view as Loose).episode_id).toBe('ep-1')
    expect((view.nodes ?? []).map((n) => n.type).sort()).toEqual((old.nodes ?? []).map((n) => n.type).sort())
    expect((view.edges ?? []).map((e) => e.type).sort()).toEqual((old.edges ?? []).map((e) => e.type).sort())
    for (const id of ['episode:ep-1', 'insight:1', 'quote:1']) {
      expect(props(view, id)).toMatchObject(props(old, id))
    }
  })

  it('parseArtifact gives the 3.1 answer for a 4.0 file, and leaves KG files alone', () => {
    const a = parseArtifact('ep-1.gi.json', v4())
    const b = parseArtifact('ep-1.gi.json', v3())
    expect(a.episodeId).toBe(b.episodeId)
    expect(a.nodeTypes).toEqual(b.nodeTypes)
    const kg = { item_id: 'x', nodes: [{ id: 'n', type: 'Source' }], edges: [] } as unknown as ArtifactData
    expect(parseArtifact('ep-1.kg.json', kg).nodeTypes).toEqual({ Source: 1 })
  })
})
