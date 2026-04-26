import { describe, expect, it } from 'vitest'
import type { ParsedArtifact } from '../types/artifact'
import { buildSubjectMentionsTimeline } from './subjectMentionsTimeline'

function art(
  nodes: Array<{ id: string; type: string; properties?: Record<string, unknown> }>,
  edges: Array<{ type: string; from: string; to: string }>,
): ParsedArtifact {
  return {
    name: 'test',
    kind: 'both',
    episodeId: 'merged',
    nodes: nodes.length,
    edges: edges.length,
    nodeTypes: {},
    data: { nodes, edges },
  }
}

describe('buildSubjectMentionsTimeline', () => {
  it('returns empty timeline for null inputs', () => {
    expect(buildSubjectMentionsTimeline(null, 'topic:x')).toEqual({
      months: [],
      total: 0,
      undated: 0,
      episodeCount: 0,
      insightIds: [],
      quoteIds: [],
    })
  })

  it('returns empty timeline when subject node missing', () => {
    const a = art(
      [{ id: 'topic:x', type: 'Topic' }],
      [],
    )
    expect(buildSubjectMentionsTimeline(a, 'topic:nope')).toEqual({
      months: [],
      total: 0,
      undated: 0,
      episodeCount: 0,
      insightIds: [],
      quoteIds: [],
    })
  })

  it('buckets ABOUT-linked insights by linked Episode publish_date', () => {
    const a = art(
      [
        { id: 'topic:x', type: 'Topic' },
        {
          id: 'i:1',
          type: 'Insight',
          properties: { episode_id: 'ep-a' },
        },
        {
          id: 'i:2',
          type: 'Insight',
          properties: { episode_id: 'ep-b' },
        },
        {
          id: 'i:3',
          type: 'Insight',
          properties: { episode_id: 'ep-a' },
        },
        {
          id: 'episode:ep-a',
          type: 'Episode',
          properties: { publish_date: '2024-03-15T12:00:00Z' },
        },
        {
          id: 'episode:ep-b',
          type: 'Episode',
          properties: { publish_date: '2024-05-02T08:00:00Z' },
        },
      ],
      [
        { type: 'ABOUT', from: 'i:1', to: 'topic:x' },
        { type: 'ABOUT', from: 'i:2', to: 'topic:x' },
        { type: 'ABOUT', from: 'i:3', to: 'topic:x' },
      ],
    )
    const t = buildSubjectMentionsTimeline(a, 'topic:x')
    expect(t.months).toEqual([
      { ymd: '2024-03', count: 2 },
      { ymd: '2024-05', count: 1 },
    ])
    expect(t.total).toBe(3)
    expect(t.episodeCount).toBe(2)
    expect(t.insightIds.sort()).toEqual(['i:1', 'i:2', 'i:3'])
    expect(t.quoteIds).toEqual([])
  })

  it('counts SPOKEN_BY quotes for a Person subject', () => {
    const a = art(
      [
        { id: 'person:ada', type: 'Person', properties: { name: 'Ada' } },
        { id: 'q:1', type: 'Quote', properties: { episode_id: 'ep-a' } },
        { id: 'q:2', type: 'Quote', properties: { episode_id: 'ep-a' } },
        {
          id: 'episode:ep-a',
          type: 'Episode',
          properties: { publish_date: '2024-07-04T00:00:00Z' },
        },
      ],
      [
        { type: 'SPOKEN_BY', from: 'q:1', to: 'person:ada' },
        { type: 'SPOKEN_BY', from: 'q:2', to: 'person:ada' },
      ],
    )
    const t = buildSubjectMentionsTimeline(a, 'person:ada')
    expect(t.months).toEqual([{ ymd: '2024-07', count: 2 }])
    expect(t.quoteIds.sort()).toEqual(['q:1', 'q:2'])
    expect(t.insightIds).toEqual([])
  })

  it('counts MENTIONS edges for an Entity subject', () => {
    const a = art(
      [
        { id: 'entity:acme', type: 'Entity', properties: { name: 'ACME', kind: 'org' } },
        { id: 'i:1', type: 'Insight', properties: { episode_id: 'ep-a' } },
        {
          id: 'episode:ep-a',
          type: 'Episode',
          properties: { publish_date: '2024-01-15' },
        },
      ],
      [{ type: 'MENTIONS', from: 'i:1', to: 'entity:acme' }],
    )
    const t = buildSubjectMentionsTimeline(a, 'entity:acme')
    expect(t.months).toEqual([{ ymd: '2024-01', count: 1 }])
  })

  it('counts items whose linked Episode lacks a publish_date as undated', () => {
    const a = art(
      [
        { id: 'topic:x', type: 'Topic' },
        { id: 'i:1', type: 'Insight', properties: { episode_id: 'ep-a' } },
        { id: 'i:2', type: 'Insight', properties: { episode_id: 'ep-missing' } },
        {
          id: 'episode:ep-a',
          type: 'Episode',
          properties: { publish_date: '2024-03-15' },
        },
      ],
      [
        { type: 'ABOUT', from: 'i:1', to: 'topic:x' },
        { type: 'ABOUT', from: 'i:2', to: 'topic:x' },
      ],
    )
    const t = buildSubjectMentionsTimeline(a, 'topic:x')
    expect(t.total).toBe(1)
    expect(t.undated).toBe(1)
    expect(t.months).toEqual([{ ymd: '2024-03', count: 1 }])
    expect(t.insightIds.sort()).toEqual(['i:1', 'i:2'])
  })

  it('handles either edge direction (subject as from or to)', () => {
    const a = art(
      [
        { id: 'topic:x', type: 'Topic' },
        { id: 'i:1', type: 'Insight', properties: { episode_id: 'ep-a' } },
        { id: 'i:2', type: 'Insight', properties: { episode_id: 'ep-a' } },
        {
          id: 'episode:ep-a',
          type: 'Episode',
          properties: { publish_date: '2024-03-15' },
        },
      ],
      [
        { type: 'ABOUT', from: 'i:1', to: 'topic:x' },
        { type: 'RELATED_TO', from: 'topic:x', to: 'i:2' },
      ],
    )
    const t = buildSubjectMentionsTimeline(a, 'topic:x')
    expect(t.total).toBe(2)
    expect(t.insightIds.sort()).toEqual(['i:1', 'i:2'])
  })
})
