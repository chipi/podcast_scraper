import { afterEach, describe, expect, it } from 'vitest'

import { graphElementDecorators, graphNodeFilters } from '../extensions'
import type { ArtifactData } from '../types/artifact'
import { applyGraphFilters, defaultFilterState, parseArtifact, toCytoElements } from './parsing'

/** A small KG: one episode, two topics. */
function kg() {
  const data = {
    episode_id: 'e1',
    nodes: [
      { id: 'episode:e1', type: 'Episode', properties: { title: 'E1' } },
      { id: 'topic:a', type: 'Topic', properties: { label: 'A' } },
      { id: 'topic:b', type: 'Topic', properties: { label: 'B' } },
    ],
    edges: [
      { type: 'MENTIONS', from: 'topic:a', to: 'episode:e1' },
      { type: 'MENTIONS', from: 'topic:b', to: 'episode:e1' },
    ],
  } as unknown as ArtifactData
  return parseArtifact('e1.kg.json', data)
}

afterEach(() => {
  graphElementDecorators.reset()
  graphNodeFilters.reset()
})

describe('graphElementDecorators', () => {
  it('adds data and classes to a node element and render-only edges to the canvas', () => {
    const seen: boolean[] = []
    graphElementDecorators.register({
      id: 'd',
      node: (raw) => (raw.type === 'Topic' ? { data: { badge: `b:${String(raw.id)}` }, classes: ['ext-topic'] } : null),
      edges: (_art, drawn, options) => {
        seen.push(options.aggregated)
        // `drawn` answers from the canvas's node set: an id nothing drew is false.
        return drawn('not-drawn')
          ? []
          : [{ data: { id: 'ext-edge', source: 'x', target: 'y', edgeType: 'EXT' } }]
      },
    })
    const els = toCytoElements(kg(), { enableAggregatedEdges: true })
    const topics = els.filter((e) => e.data.type === 'Topic')
    expect(topics.length).toBe(2)
    for (const t of topics) {
      expect(String(t.data.badge)).toMatch(/^b:/)
      expect(String(t.classes)).toContain('ext-topic')
    }
    expect(els.find((e) => e.data.type === 'Episode')?.classes ?? '').not.toContain('ext-topic')
    expect(els.some((e) => e.data.id === 'ext-edge')).toBe(true)
    expect(seen).toEqual([true])
  })
})

describe('graphNodeFilters', () => {
  it('drops the nodes an extension filter refuses, before the type filter', () => {
    graphNodeFilters.register({ id: 'f', keep: (n) => !String(n.id).endsWith('topic:b') })
    const art = kg()
    const state = defaultFilterState(art)!
    const kept = (applyGraphFilters(art, state).data.nodes ?? []).map((n) => String(n.id))
    expect(kept.some((id) => id.endsWith('topic:a'))).toBe(true)
    expect(kept.some((id) => id.endsWith('topic:b'))).toBe(false)
  })

  it('keeps every node when nothing is registered', () => {
    const art = kg()
    const state = defaultFilterState(art)!
    expect((applyGraphFilters(art, state).data.nodes ?? []).length).toBe(3)
  })
})
