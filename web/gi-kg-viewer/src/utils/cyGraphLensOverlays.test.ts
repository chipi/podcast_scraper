// @vitest-environment happy-dom
/**
 * Tier 5C/5D — cyGraphLensOverlays unit tests (harden follow-up).
 *
 * Uses a hand-rolled MockCore that satisfies the narrow slice of the
 * Cytoscape Core interface these apply functions actually touch
 * (batch, nodes(selector), $id, add, edges(selector), remove).
 * Cheaper than booting a real Cytoscape instance in happy-dom and
 * covers every branch we care about.
 */
import { describe, expect, it } from 'vitest'
import {
  applyCoGuestEdges,
  applyConsensusEdges,
  applyCredibilityBorder,
  applyVelocityHalo,
  clearCoGuestEdges,
  clearConsensusEdges,
  clearCredibilityBorder,
  clearVelocityHalo,
  COGUEST_EDGE_CLASS,
  CONSENSUS_EDGE_CLASS,
} from './cyGraphLensOverlays'

interface NodeStub {
  id: () => string
  data: (key?: string) => unknown
  addClass: (cls: string) => void
  removeClass: (cls: string) => void
  empty: () => boolean
  classes: string[]
  _data: Record<string, unknown>
}

interface EdgeStub {
  id: () => string
  data: (key?: string) => unknown
  classes: string[]
  _data: Record<string, unknown>
}

function makeNode(id: string, data: Record<string, unknown> = {}): NodeStub {
  const stub: NodeStub = {
    classes: [],
    _data: { id, ...data },
    id: () => id,
    data: (key?: string) => (key ? stub._data[key] : stub._data),
    addClass(cls: string) {
      if (!this.classes.includes(cls)) this.classes.push(cls)
    },
    removeClass(cls: string) {
      const i = this.classes.indexOf(cls)
      if (i >= 0) this.classes.splice(i, 1)
    },
    empty: () => false,
  }
  return stub
}

function makeCore(nodes: NodeStub[]) {
  const edges: EdgeStub[] = []
  const nodesById = new Map<string, NodeStub>(nodes.map((n) => [n.id(), n]))
  const core = {
    _addedEdges: [] as EdgeStub[],
    _removedEdges: [] as EdgeStub[],
    batch(fn: () => void) {
      fn()
    },
    nodes(selector?: string) {
      // Return a collection with the bulk operations apply/clear fns use.
      // Selector is ignored — helpers walk all nodes or query by id via $id.
      void selector
      return {
        forEach(fn: (n: NodeStub) => void) {
          for (const n of nodes) fn(n)
        },
        removeClass(cls: string) {
          for (const n of nodes) n.removeClass(cls)
        },
        addClass(cls: string) {
          for (const n of nodes) n.addClass(cls)
        },
      }
    },
    $id(id: string) {
      const hit = nodesById.get(id)
      if (hit) return hit
      // Cytoscape returns a stub-collection with .empty()===true when the id
      // is not found.
      return {
        id: () => '',
        data: () => undefined,
        addClass: () => {
          /* no-op */
        },
        removeClass: () => {
          /* no-op */
        },
        empty: () => true,
      } as NodeStub
    },
    add(def: { data: Record<string, unknown>; classes?: string }) {
      const eid = String(def.data.id)
      const stub: EdgeStub = {
        id: () => eid,
        data: (key?: string) => (key ? def.data[key] : def.data),
        classes: def.classes ? def.classes.split(/\s+/).filter(Boolean) : [],
        _data: def.data,
      }
      edges.push(stub)
      core._addedEdges.push(stub)
      return stub
    },
    edges(selector: string) {
      // Simple `.class-name` matching only — enough for the apply/clear helpers.
      const cls = selector.startsWith('.') ? selector.slice(1) : selector
      const matching = edges.filter((e) => e.classes.includes(cls))
      return {
        forEach(fn: (e: EdgeStub) => void) {
          for (const e of matching) fn(e)
        },
      }
    },
    remove(e: EdgeStub) {
      const i = edges.indexOf(e)
      if (i >= 0) edges.splice(i, 1)
      core._removedEdges.push(e)
    },
  }
  return core
}

describe('applyVelocityHalo (Tier 5C-1)', () => {
  it('is a no-op when envelope is null', () => {
    const core = makeCore([makeNode('g:topic:x')])
    // @ts-expect-error MockCore satisfies the runtime interface, not the strict Cytoscape type.
    applyVelocityHalo(core, null)
    expect(core.nodes().forEach).toBeDefined()
  })

  it('paints velocity-up when velocity_last_over_6mo >= 1.15', () => {
    const t1 = makeNode('g:topic:rising')
    const core = makeCore([t1])
    // @ts-expect-error MockCore
    applyVelocityHalo(core, { topics: [{ topic_id: 'topic:rising', velocity_last_over_6mo: 2.0 }] })
    expect(t1.classes).toContain('velocity-up')
  })

  it('paints velocity-down when velocity_last_over_6mo <= 0.85', () => {
    const t1 = makeNode('g:topic:cooling')
    const core = makeCore([t1])
    // @ts-expect-error MockCore
    applyVelocityHalo(core, {
      topics: [{ topic_id: 'topic:cooling', velocity_last_over_6mo: 0.5 }],
    })
    expect(t1.classes).toContain('velocity-down')
  })

  it('paints velocity-steady in the neutral 0.85..1.15 band', () => {
    const t1 = makeNode('g:topic:steady')
    const core = makeCore([t1])
    // @ts-expect-error MockCore
    applyVelocityHalo(core, {
      topics: [{ topic_id: 'topic:steady', velocity_last_over_6mo: 1.0 }],
    })
    expect(t1.classes).toContain('velocity-steady')
  })

  it('skips rows with non-finite velocity', () => {
    const t1 = makeNode('g:topic:x')
    const core = makeCore([t1])
    // @ts-expect-error MockCore
    applyVelocityHalo(core, {
      topics: [{ topic_id: 'topic:x', velocity_last_over_6mo: Number.NaN }],
    })
    expect(t1.classes).toEqual([])
  })

  it('clearVelocityHalo removes every previously-assigned velocity class', () => {
    const t1 = makeNode('g:topic:x')
    t1.addClass('velocity-up')
    t1.addClass('velocity-down')
    t1.addClass('velocity-steady')
    const core = makeCore([t1])
    // @ts-expect-error MockCore
    clearVelocityHalo(core)
    expect(t1.classes).toEqual([])
  })
})

describe('applyCredibilityBorder (Tier 5C-2)', () => {
  it('rate >= 0.7 → credibility-high', () => {
    const p = makeNode('g:person:a')
    const core = makeCore([p])
    // @ts-expect-error MockCore
    applyCredibilityBorder(core, { persons: [{ person_id: 'person:a', rate: 0.85 }] })
    expect(p.classes).toContain('credibility-high')
  })

  it('0.4 <= rate < 0.7 → credibility-med', () => {
    const p = makeNode('g:person:b')
    const core = makeCore([p])
    // @ts-expect-error MockCore
    applyCredibilityBorder(core, { persons: [{ person_id: 'person:b', rate: 0.5 }] })
    expect(p.classes).toContain('credibility-med')
  })

  it('rate < 0.4 → credibility-low', () => {
    const p = makeNode('g:person:c')
    const core = makeCore([p])
    // @ts-expect-error MockCore
    applyCredibilityBorder(core, { persons: [{ person_id: 'person:c', rate: 0.1 }] })
    expect(p.classes).toContain('credibility-low')
  })

  it('is a no-op when envelope is null', () => {
    const p = makeNode('g:person:x')
    const core = makeCore([p])
    // @ts-expect-error MockCore
    applyCredibilityBorder(core, null)
    expect(p.classes).toEqual([])
  })

  it('clearCredibilityBorder removes every credibility class', () => {
    const p = makeNode('g:person:x')
    p.addClass('credibility-high')
    p.addClass('credibility-low')
    const core = makeCore([p])
    // @ts-expect-error MockCore
    clearCredibilityBorder(core)
    expect(p.classes).toEqual([])
  })
})

describe('applyConsensusEdges (Tier 5D-1)', () => {
  it('adds one edge per consensus row when both persons exist', () => {
    const a = makeNode('g:person:a')
    const b = makeNode('g:person:b')
    const core = makeCore([a, b])
    // @ts-expect-error MockCore
    applyConsensusEdges(core, {
      consensus: [{ topic_id: 'topic:x', person_a_id: 'person:a', person_b_id: 'person:b' }],
    })
    expect(core._addedEdges.length).toBe(1)
    expect(core._addedEdges[0]?.classes).toContain(CONSENSUS_EDGE_CLASS)
  })

  it('dedupes when the same pair corroborates on the same topic twice', () => {
    const a = makeNode('g:person:a')
    const b = makeNode('g:person:b')
    const core = makeCore([a, b])
    // @ts-expect-error MockCore
    applyConsensusEdges(core, {
      consensus: [
        { topic_id: 'topic:x', person_a_id: 'person:a', person_b_id: 'person:b' },
        { topic_id: 'topic:x', person_a_id: 'person:b', person_b_id: 'person:a' }, // same pair swapped
      ],
    })
    expect(core._addedEdges.length).toBe(1)
  })

  it('emits distinct edges for the same pair on different topics', () => {
    const a = makeNode('g:person:a')
    const b = makeNode('g:person:b')
    const core = makeCore([a, b])
    // @ts-expect-error MockCore
    applyConsensusEdges(core, {
      consensus: [
        { topic_id: 'topic:x', person_a_id: 'person:a', person_b_id: 'person:b' },
        { topic_id: 'topic:y', person_a_id: 'person:a', person_b_id: 'person:b' },
      ],
    })
    expect(core._addedEdges.length).toBe(2)
  })

  it('skips rows where a person is not in the graph slice', () => {
    const a = makeNode('g:person:a')
    const core = makeCore([a])
    // @ts-expect-error MockCore
    applyConsensusEdges(core, {
      consensus: [{ topic_id: 'topic:x', person_a_id: 'person:a', person_b_id: 'person:missing' }],
    })
    expect(core._addedEdges.length).toBe(0)
  })

  it('clearConsensusEdges removes every consensus edge', () => {
    const a = makeNode('g:person:a')
    const b = makeNode('g:person:b')
    const core = makeCore([a, b])
    // @ts-expect-error MockCore
    applyConsensusEdges(core, {
      consensus: [{ topic_id: 'topic:x', person_a_id: 'person:a', person_b_id: 'person:b' }],
    })
    // @ts-expect-error MockCore
    clearConsensusEdges(core)
    expect(core._removedEdges.length).toBe(1)
  })
})

describe('applyCoGuestEdges (Tier 5D-2)', () => {
  it('adds edges only when episode_count >= threshold (default 2)', () => {
    const a = makeNode('g:person:a')
    const b = makeNode('g:person:b')
    const c = makeNode('g:person:c')
    const core = makeCore([a, b, c])
    // @ts-expect-error MockCore
    applyCoGuestEdges(core, {
      pairs: [
        { person_a_id: 'person:a', person_b_id: 'person:b', episode_count: 3 },
        { person_a_id: 'person:a', person_b_id: 'person:c', episode_count: 1 }, // below threshold
      ],
    })
    expect(core._addedEdges.length).toBe(1)
    expect(core._addedEdges[0]?.classes).toContain(COGUEST_EDGE_CLASS)
  })

  it('honours a custom minEpisodeCount override', () => {
    const a = makeNode('g:person:a')
    const b = makeNode('g:person:b')
    const core = makeCore([a, b])
    // @ts-expect-error MockCore
    applyCoGuestEdges(
      core,
      { pairs: [{ person_a_id: 'person:a', person_b_id: 'person:b', episode_count: 2 }] },
      5,
    )
    expect(core._addedEdges.length).toBe(0)
  })

  it('dedupes when the same pair appears in both orderings', () => {
    const a = makeNode('g:person:a')
    const b = makeNode('g:person:b')
    const core = makeCore([a, b])
    // @ts-expect-error MockCore
    applyCoGuestEdges(core, {
      pairs: [
        { person_a_id: 'person:a', person_b_id: 'person:b', episode_count: 3 },
        { person_a_id: 'person:b', person_b_id: 'person:a', episode_count: 5 },
      ],
    })
    expect(core._addedEdges.length).toBe(1)
  })

  it('stores episode_count as edge weight (for width mapping)', () => {
    const a = makeNode('g:person:a')
    const b = makeNode('g:person:b')
    const core = makeCore([a, b])
    // @ts-expect-error MockCore
    applyCoGuestEdges(core, {
      pairs: [{ person_a_id: 'person:a', person_b_id: 'person:b', episode_count: 7 }],
    })
    expect(core._addedEdges[0]?._data.weight).toBe(7)
  })

  it('clearCoGuestEdges removes every co-guest edge', () => {
    const a = makeNode('g:person:a')
    const b = makeNode('g:person:b')
    const core = makeCore([a, b])
    // @ts-expect-error MockCore
    applyCoGuestEdges(core, {
      pairs: [{ person_a_id: 'person:a', person_b_id: 'person:b', episode_count: 3 }],
    })
    // @ts-expect-error MockCore
    clearCoGuestEdges(core)
    expect(core._removedEdges.length).toBe(1)
  })
})
