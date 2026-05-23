import { describe, expect, it } from 'vitest'
import type { EdgeSingular } from 'cytoscape'
import {
  giKgCoseEdgeElasticity,
  giKgCoseIdealEdgeLength,
  giKgCoseLayoutOptionsCompact,
  giKgCoseLayoutOptionsMain,
  giKgCoseLayoutOptionsMainFallback,
  giKgCoseNodeRepulsionFromData,
  giKgCoseNumIterCapped,
  isIntraTopicClusterEdgeParents,
  isTopicClusterParentId,
  REDRAW_DEBOUNCE_INTERNAL_MS,
  RECENTER_SAFETY_TAIL_TIMINGS_MS,
  redrawDebounceMs,
} from './cyCoseLayoutOptions'

describe('isTopicClusterParentId', () => {
  it('is true for tc: ids', () => {
    expect(isTopicClusterParentId('tc:foo')).toBe(true)
    expect(isTopicClusterParentId('  tc:x  ')).toBe(true)
  })

  it('is false for other parents', () => {
    expect(isTopicClusterParentId('')).toBe(false)
    expect(isTopicClusterParentId('g:topic')).toBe(false)
    expect(isTopicClusterParentId(undefined)).toBe(false)
  })
})

describe('isIntraTopicClusterEdgeParents', () => {
  it('is true when both parents match the same tc:', () => {
    expect(isIntraTopicClusterEdgeParents('tc:a', 'tc:a')).toBe(true)
  })

  it('is false when parents differ or are not tc:', () => {
    expect(isIntraTopicClusterEdgeParents('tc:a', 'tc:b')).toBe(false)
    expect(isIntraTopicClusterEdgeParents(undefined, 'tc:a')).toBe(false)
    expect(isIntraTopicClusterEdgeParents('g:x', 'g:x')).toBe(false)
  })
})

describe('giKgCoseNodeRepulsionFromData', () => {
  it('uses higher repulsion for TopicCluster compounds', () => {
    const main = giKgCoseNodeRepulsionFromData('TopicCluster', undefined, 'main')
    const base = giKgCoseNodeRepulsionFromData('Topic', undefined, 'main')
    expect(main).toBeGreaterThan(base)
  })

  it('uses lower repulsion for members under tc:', () => {
    const member = giKgCoseNodeRepulsionFromData('Topic', 'tc:cl', 'main')
    const base = giKgCoseNodeRepulsionFromData('Topic', undefined, 'main')
    expect(member).toBeLessThan(base)
  })
})

describe('giKgCoseLayoutOptionsMain', () => {
  it('includes nestingFactor for cross-graph edges', () => {
    const o = giKgCoseLayoutOptionsMain()
    expect(o.name).toBe('cose')
    expect(o.nestingFactor).toBe(1.52)
    expect(o.numIter).toBe(2500)
    expect(typeof o.nodeRepulsion).toBe('function')
    expect(typeof o.idealEdgeLength).toBe('function')
    expect(typeof o.edgeElasticity).toBe('function')
  })
})

function mockEdge(
  sourceParent: string | null,
  targetParent: string | null,
  edgeType: string,
): EdgeSingular {
  return {
    data: (k: string) => (k === 'edgeType' ? edgeType : undefined),
    source: () => ({ data: (k: string) => (k === 'parent' ? sourceParent : undefined) }),
    target: () => ({ data: (k: string) => (k === 'parent' ? targetParent : undefined) }),
  } as unknown as EdgeSingular
}

describe('giKgCoseIdealEdgeLength', () => {
  it('prioritises intra-topic-cluster length over semantic ABOUT', () => {
    const edge = mockEdge('tc:x', 'tc:x', 'ABOUT')
    expect(giKgCoseIdealEdgeLength(edge, 'main')).toBe(36)
  })

  it('uses semantic ABOUT length outside tc: clusters', () => {
    const edge = mockEdge(null, null, 'ABOUT')
    expect(giKgCoseIdealEdgeLength(edge, 'main')).toBe(80)
  })
})

describe('giKgCoseEdgeElasticity', () => {
  it('uses default elasticity inside tc: clusters', () => {
    const edge = mockEdge('tc:x', 'tc:x', 'ABOUT')
    expect(giKgCoseEdgeElasticity(edge, 'main')).toBe(100)
  })

  it('uses semantic elasticity for ABOUT outside tc:', () => {
    const edge = mockEdge(null, null, 'ABOUT')
    expect(giKgCoseEdgeElasticity(edge, 'main')).toBe(200)
  })
})

describe('giKgCoseLayoutOptionsMainFallback', () => {
  it('uses flat repulsion/edge length (no per-node callbacks)', () => {
    const o = giKgCoseLayoutOptionsMainFallback()
    expect(o.name).toBe('cose')
    expect(o.nestingFactor).toBe(1.52)
    expect(o.numIter).toBe(2500)
    expect(typeof o.nodeRepulsion).toBe('function')
    expect(typeof o.idealEdgeLength).toBe('function')
    expect((o.nodeRepulsion as () => number)()).toBe(880_000)
    expect((o.idealEdgeLength as () => number)()).toBe(96)
  })
})

describe('giKgCoseLayoutOptionsCompact', () => {
  it('matches minimap gravity and disables animation', () => {
    const o = giKgCoseLayoutOptionsCompact()
    expect(o.gravity).toBe(0.32)
    expect(o.animate).toBe(false)
    expect(o.nestingFactor).toBe(1.52)
  })
})

describe('#767-A — giKgCoseNumIterCapped + numIter override', () => {
  // Default ``MAIN.numIter`` is 2500 per cyCoseLayoutOptions.ts. The cap
  // formula is ``Math.min(2500, 200 + 8 × N)`` — keep these tests pinned
  // to that formula so any future re-tune of the cap is explicit.

  it('returns the cap for small graphs (well below 2500)', () => {
    expect(giKgCoseNumIterCapped(50)).toBe(600) // 200 + 8×50
    expect(giKgCoseNumIterCapped(100)).toBe(1000) // 200 + 8×100
    expect(giKgCoseNumIterCapped(200)).toBe(1800) // 200 + 8×200
  })

  it('saturates at the default 2500 for large graphs', () => {
    expect(giKgCoseNumIterCapped(290)).toBe(2500) // 200 + 8×290 = 2520 → clamped
    expect(giKgCoseNumIterCapped(500)).toBe(2500)
    expect(giKgCoseNumIterCapped(2000)).toBe(2500)
  })

  it('production-shaped corpus (270 nodes) gets a moderate cap', () => {
    expect(giKgCoseNumIterCapped(270)).toBe(2360) // 200 + 8×270
  })

  it('defends against bad input — non-positive / NaN returns the default', () => {
    expect(giKgCoseNumIterCapped(0)).toBe(2500)
    expect(giKgCoseNumIterCapped(-5)).toBe(2500)
    expect(giKgCoseNumIterCapped(NaN)).toBe(2500)
  })

  it('giKgCoseLayoutOptionsMain accepts a numIter override', () => {
    const def = giKgCoseLayoutOptionsMain()
    expect(def.numIter).toBe(2500)

    const capped = giKgCoseLayoutOptionsMain(1800)
    expect(capped.numIter).toBe(1800)
  })

  it('giKgCoseLayoutOptionsMainFallback accepts a numIter override too', () => {
    const def = giKgCoseLayoutOptionsMainFallback()
    expect(def.numIter).toBe(2500)

    const capped = giKgCoseLayoutOptionsMainFallback(1000)
    expect(capped.numIter).toBe(1000)
  })
})

describe('#767-B — redrawDebounceMs (behavior contract)', () => {
  // The constant + branch pin the before/after for the bypass: prior
  // code always used 150 ms; the new path returns 0 ms whenever an FSM
  // envelope is pending. If a future change reintroduces the slack, the
  // first assertion goes red.

  it('returns 0 ms when an FSM envelope is pending (handoff in flight)', () => {
    expect(redrawDebounceMs(true)).toBe(0)
  })

  it('returns the internal-cascade debounce (150 ms) when no handoff is pending', () => {
    expect(redrawDebounceMs(false)).toBe(REDRAW_DEBOUNCE_INTERNAL_MS)
    expect(REDRAW_DEBOUNCE_INTERNAL_MS).toBe(150)
  })

  it('saves at least 150 ms per cross-surface handoff click vs the always-debounce baseline', () => {
    // The "before this fix" baseline was a flat 150 ms regardless of
    // handoff state. Pin the contrast so the optimisation's user-visible
    // delta is locked in code.
    const before = 150 // legacy: setTimeout(redraw, 150)
    const after = redrawDebounceMs(true)
    expect(before - after).toBeGreaterThanOrEqual(150)
  })
})

describe('#767-C — RECENTER_SAFETY_TAIL_TIMINGS_MS (behavior contract)', () => {
  // ``animateCameraToFocusedNode`` arms one ``setTimeout`` per entry in
  // this list. The schedule is the three-anchor [400, 900, 1800] tail;
  // #787 attempted to trim to [400] for perceived-latency savings but
  // firefox-mac Tier-2 production-shaped specs (``e2e/handoff-production/``)
  // surfaced a regression — local canvas-resize settle is past 400 ms,
  // so the late timers were catching missed recenters that linux-CI
  // never needed. Schedule restored.

  it('matches the three-anchor schedule [400, 900, 1800]', () => {
    expect(RECENTER_SAFETY_TAIL_TIMINGS_MS).toEqual([400, 900, 1800])
  })

  it('is a readonly tuple — accidental push to the array would not compile', () => {
    // Type-level guarantee (``readonly number[]``); also pin the runtime
    // shape so the constant is the single source of truth for the schedule.
    expect(Array.isArray(RECENTER_SAFETY_TAIL_TIMINGS_MS)).toBe(true)
    expect(RECENTER_SAFETY_TAIL_TIMINGS_MS.length).toBeLessThanOrEqual(3)
  })
})
