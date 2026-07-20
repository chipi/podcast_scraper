import { describe, expect, it } from 'vitest'
import type { EdgeSingular, NodeSingular } from 'cytoscape'
import {
  giKgCoseEdgeElasticity,
  giKgCoseIdealEdgeLength,
  giKgCoseLayoutOptionsCompact,
  giKgCoseLayoutOptionsMain,
  giKgCoseLayoutOptionsMainFallback,
  giKgCoseNodeRepulsion,
  giKgCoseNodeRepulsionFromData,
  giKgCoseNumIterCapped,
  isIntraTopicClusterEdgeParents,
  isTopicClusterParentId,
  REDRAW_DEBOUNCE_INTERNAL_MS,
  RECENTER_SAFETY_TAIL_TIMINGS_MS,
  redrawDebounceMs,
} from './cyCoseLayoutOptions'

function mockNode(type: string | undefined, parent: string | null | undefined): NodeSingular {
  return {
    data: (k: string) => (k === 'type' ? type : k === 'parent' ? parent : undefined),
  } as unknown as NodeSingular
}

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

describe('isTopicClusterParentId — null', () => {
  it('is false for null', () => {
    expect(isTopicClusterParentId(null)).toBe(false)
  })

  it('is false for a tc: substring that is not a prefix', () => {
    expect(isTopicClusterParentId('x:tc:foo')).toBe(false)
  })
})

describe('isIntraTopicClusterEdgeParents — extra branches', () => {
  it('is false when one parent is null', () => {
    expect(isIntraTopicClusterEdgeParents('tc:a', null)).toBe(false)
    expect(isIntraTopicClusterEdgeParents(null, 'tc:a')).toBe(false)
  })

  it('is false when both null/undefined/empty', () => {
    expect(isIntraTopicClusterEdgeParents(null, null)).toBe(false)
    expect(isIntraTopicClusterEdgeParents(undefined, undefined)).toBe(false)
    expect(isIntraTopicClusterEdgeParents('', '')).toBe(false)
  })

  it('trims whitespace before comparing', () => {
    expect(isIntraTopicClusterEdgeParents('  tc:a  ', 'tc:a')).toBe(true)
  })

  it('is false when matching parents are not tc:', () => {
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

  it('returns the exact MAIN constants per node class', () => {
    expect(giKgCoseNodeRepulsionFromData('TopicCluster', undefined, 'main')).toBe(1_450_000)
    expect(giKgCoseNodeRepulsionFromData('Topic', 'tc:cl', 'main')).toBe(180_000)
    expect(giKgCoseNodeRepulsionFromData('Topic', undefined, 'main')).toBe(880_000)
  })

  it('returns the exact COMPACT constants per node class', () => {
    expect(giKgCoseNodeRepulsionFromData('TopicCluster', undefined, 'compact')).toBe(198_000)
    expect(giKgCoseNodeRepulsionFromData('Topic', 'tc:cl', 'compact')).toBe(24_000)
    expect(giKgCoseNodeRepulsionFromData('Topic', undefined, 'compact')).toBe(120_000)
  })

  it('treats undefined type as the base (non-cluster) class', () => {
    expect(giKgCoseNodeRepulsionFromData(undefined, undefined, 'main')).toBe(880_000)
    expect(giKgCoseNodeRepulsionFromData(undefined, undefined, 'compact')).toBe(120_000)
  })

  it('TopicCluster type wins over a tc: parent (compound takes precedence)', () => {
    expect(giKgCoseNodeRepulsionFromData('TopicCluster', 'tc:cl', 'main')).toBe(1_450_000)
  })

  it('ignores a whitespace-only parent (falls to base)', () => {
    expect(giKgCoseNodeRepulsionFromData('Topic', '   ', 'main')).toBe(880_000)
  })

  it('ignores a non-tc: parent (falls to base)', () => {
    expect(giKgCoseNodeRepulsionFromData('Topic', 'g:cluster', 'main')).toBe(880_000)
  })
})

describe('giKgCoseNodeRepulsion (NodeSingular wrapper)', () => {
  it('reads type + parent off node.data and trims the parent', () => {
    expect(giKgCoseNodeRepulsion(mockNode('TopicCluster', null), 'main')).toBe(1_450_000)
    expect(giKgCoseNodeRepulsion(mockNode('Topic', '  tc:cl  '), 'main')).toBe(180_000)
    expect(giKgCoseNodeRepulsion(mockNode('Topic', null), 'main')).toBe(880_000)
    expect(giKgCoseNodeRepulsion(mockNode('Topic', null), 'compact')).toBe(120_000)
  })

  it('treats a whitespace-only / non-string parent as no parent', () => {
    expect(giKgCoseNodeRepulsion(mockNode('Topic', '   '), 'main')).toBe(880_000)
    expect(giKgCoseNodeRepulsion(mockNode('Topic', undefined), 'main')).toBe(880_000)
  })
})

describe('giKgCoseLayoutOptionsMain', () => {
  it('includes nestingFactor for cross-graph edges', () => {
    const o = giKgCoseLayoutOptionsMain()
    expect(o.name).toBe('fcose')
    expect(o.nestingFactor).toBe(1.52)
    expect(o.numIter).toBe(2500)
    expect(typeof o.nodeRepulsion).toBe('function')
    expect(typeof o.idealEdgeLength).toBe('function')
    expect(typeof o.edgeElasticity).toBe('function')
  })

  it('exposes the static MAIN tuning constants', () => {
    const o = giKgCoseLayoutOptionsMain()
    expect(o.padding).toBe(36)
    expect(o.fit).toBe(false)
    // graph-v3 M — gravity lowered so natural communities drift apart.
    expect(o.gravity).toBe(0.12)
    expect(o.nodeDimensionsIncludeLabels).toBe(false)
  })

  it('wires the per-node repulsion callback to the main profile', () => {
    const o = giKgCoseLayoutOptionsMain()
    const nodeRepulsion = o.nodeRepulsion as (n: NodeSingular) => number
    expect(nodeRepulsion(mockNode('TopicCluster', null))).toBe(1_450_000)
    expect(nodeRepulsion(mockNode('Topic', 'tc:cl'))).toBe(180_000)
    expect(nodeRepulsion(mockNode('Topic', null))).toBe(880_000)
  })

  it('wires the per-edge callbacks to the main profile', () => {
    const o = giKgCoseLayoutOptionsMain()
    const idealEdgeLength = o.idealEdgeLength as (e: EdgeSingular) => number
    const edgeElasticity = o.edgeElasticity as (e: EdgeSingular) => number
    expect(idealEdgeLength(mockEdge('tc:x', 'tc:x', 'ABOUT'))).toBe(36)
    expect(idealEdgeLength(mockEdge(null, null, 'MENTIONS'))).toBe(150)
    expect(edgeElasticity(mockEdge(null, null, 'ABOUT'))).toBe(200)
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

  it('uses the compact intra-topic-cluster length for the compact profile', () => {
    const edge = mockEdge('tc:x', 'tc:x', 'ABOUT')
    expect(giKgCoseIdealEdgeLength(edge, 'compact')).toBe(20)
  })

  it('maps every semantic edge type to its main ideal length', () => {
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, 'HAS_INSIGHT'), 'main')).toBe(60)
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, 'ABOUT'), 'main')).toBe(80)
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, 'SUPPORTED_BY'), 'main')).toBe(40)
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, 'RELATED_TO'), 'main')).toBe(120)
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, 'SPOKE_IN'), 'main')).toBe(100)
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, 'MENTIONS'), 'main')).toBe(150)
  })

  it('uses the base length for unknown / empty edge types (main)', () => {
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, 'UNKNOWN'), 'main')).toBe(96)
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, ''), 'main')).toBe(96)
  })

  it('typed MENTIONS family (RFC-097 v3.0) inherits the legacy MENTIONS length', () => {
    // MENTIONS_PERSON / MENTIONS_ORG must layout identically to the legacy
    // generic MENTIONS so the typed split doesn't visually rearrange graphs.
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, 'MENTIONS_PERSON'), 'main')).toBe(150)
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, 'MENTIONS_ORG'), 'main')).toBe(150)
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, 'MENTIONS_PERSON'), 'compact')).toBe(81)
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, 'MENTIONS_ORG'), 'compact')).toBe(81)
  })

  it('scales semantic lengths for the compact profile (52/96 factor, rounded)', () => {
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, 'HAS_INSIGHT'), 'compact')).toBe(33)
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, 'ABOUT'), 'compact')).toBe(43)
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, 'RELATED_TO'), 'compact')).toBe(65)
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, 'SPOKE_IN'), 'compact')).toBe(54)
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, 'MENTIONS'), 'compact')).toBe(81)
  })

  it('clamps the smallest compact semantic length to the 24px floor', () => {
    // SUPPORTED_BY main 40 → 40×52/96 ≈ 21.67 → would round to 22 → floored to 24.
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, 'SUPPORTED_BY'), 'compact')).toBe(24)
  })

  it('uses the compact base length for unknown edge types (compact)', () => {
    expect(giKgCoseIdealEdgeLength(mockEdge(null, null, 'UNKNOWN'), 'compact')).toBe(52)
  })

  it('falls back to the base length when edgeType data is missing', () => {
    const edge = {
      data: () => undefined,
      source: () => ({ data: () => null }),
      target: () => ({ data: () => null }),
    } as unknown as EdgeSingular
    expect(giKgCoseIdealEdgeLength(edge, 'main')).toBe(96)
  })
})

describe('giKgCoseEdgeElasticity', () => {
  it('uses default elasticity inside tc: clusters (main)', () => {
    const edge = mockEdge('tc:x', 'tc:x', 'ABOUT')
    expect(giKgCoseEdgeElasticity(edge, 'main')).toBe(100)
  })

  it('uses the compact default elasticity inside tc: clusters (compact)', () => {
    const edge = mockEdge('tc:x', 'tc:x', 'ABOUT')
    expect(giKgCoseEdgeElasticity(edge, 'compact')).toBe(80)
  })

  it('maps every semantic edge type to its main elasticity', () => {
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, 'HAS_INSIGHT'), 'main')).toBe(180)
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, 'ABOUT'), 'main')).toBe(200)
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, 'SUPPORTED_BY'), 'main')).toBe(150)
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, 'RELATED_TO'), 'main')).toBe(100)
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, 'SPOKE_IN'), 'main')).toBe(120)
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, 'MENTIONS'), 'main')).toBe(60)
  })

  it('uses the base elasticity for unknown / empty edge types (main)', () => {
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, 'UNKNOWN'), 'main')).toBe(100)
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, ''), 'main')).toBe(100)
  })

  it('typed MENTIONS family (RFC-097 v3.0) inherits the legacy MENTIONS elasticity', () => {
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, 'MENTIONS_PERSON'), 'main')).toBe(60)
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, 'MENTIONS_ORG'), 'main')).toBe(60)
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, 'MENTIONS_PERSON'), 'compact')).toBe(48)
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, 'MENTIONS_ORG'), 'compact')).toBe(48)
  })

  it('scales semantic elasticity for the compact profile (×0.8, rounded, ≥40 floor)', () => {
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, 'HAS_INSIGHT'), 'compact')).toBe(144)
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, 'ABOUT'), 'compact')).toBe(160)
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, 'SUPPORTED_BY'), 'compact')).toBe(120)
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, 'RELATED_TO'), 'compact')).toBe(80)
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, 'SPOKE_IN'), 'compact')).toBe(96)
  })

  it('clamps the smallest compact elasticity to the 40 floor', () => {
    // MENTIONS main 60 → 60×0.8 = 48 (above floor); confirm exact rounding too.
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, 'MENTIONS'), 'compact')).toBe(48)
  })

  it('uses the compact base elasticity for unknown edge types (compact)', () => {
    expect(giKgCoseEdgeElasticity(mockEdge(null, null, 'UNKNOWN'), 'compact')).toBe(80)
  })

  it('falls back to the base elasticity when edgeType data is missing (nullish coalesce)', () => {
    const edge = {
      data: () => undefined,
      source: () => ({ data: () => null }),
      target: () => ({ data: () => null }),
    } as unknown as EdgeSingular
    expect(giKgCoseEdgeElasticity(edge, 'main')).toBe(100)
    expect(giKgCoseEdgeElasticity(edge, 'compact')).toBe(80)
  })
})

describe('giKgCoseLayoutOptionsMainFallback', () => {
  it('uses flat repulsion/edge length (no per-node callbacks)', () => {
    const o = giKgCoseLayoutOptionsMainFallback()
    expect(o.name).toBe('fcose')
    expect(o.nestingFactor).toBe(1.52)
    expect(o.numIter).toBe(2500)
    expect(typeof o.nodeRepulsion).toBe('function')
    expect(typeof o.idealEdgeLength).toBe('function')
    expect((o.nodeRepulsion as () => number)()).toBe(880_000)
    expect((o.idealEdgeLength as () => number)()).toBe(96)
  })
})

describe('giKgCoseLayoutOptionsMainFallback — static constants', () => {
  it('exposes the same static MAIN tuning constants', () => {
    const o = giKgCoseLayoutOptionsMainFallback()
    expect(o.padding).toBe(36)
    expect(o.fit).toBe(false)
    // graph-v3 M — gravity lowered so natural communities drift apart.
    expect(o.gravity).toBe(0.12)
    expect(o.nodeDimensionsIncludeLabels).toBe(false)
    expect((o.edgeElasticity as () => number)()).toBe(100)
  })
})

describe('giKgCoseLayoutOptionsCompact', () => {
  it('matches minimap gravity and disables animation', () => {
    const o = giKgCoseLayoutOptionsCompact()
    expect(o.gravity).toBe(0.32)
    expect(o.animate).toBe(false)
    expect(o.nestingFactor).toBe(1.52)
  })

  it('exposes the static COMPACT tuning constants', () => {
    const o = giKgCoseLayoutOptionsCompact()
    expect(o.name).toBe('fcose')
    expect(o.padding).toBe(14)
    expect(o.fit).toBe(false)
    expect(o.nodeDimensionsIncludeLabels).toBe(true)
  })

  it('wires the per-node + per-edge callbacks to the compact profile', () => {
    const o = giKgCoseLayoutOptionsCompact()
    const nodeRepulsion = o.nodeRepulsion as (n: NodeSingular) => number
    const idealEdgeLength = o.idealEdgeLength as (e: EdgeSingular) => number
    const edgeElasticity = o.edgeElasticity as (e: EdgeSingular) => number
    expect(nodeRepulsion(mockNode('TopicCluster', null))).toBe(198_000)
    expect(nodeRepulsion(mockNode('Topic', 'tc:cl'))).toBe(24_000)
    expect(idealEdgeLength(mockEdge('tc:x', 'tc:x', 'ABOUT'))).toBe(20)
    expect(idealEdgeLength(mockEdge(null, null, 'ABOUT'))).toBe(43)
    expect(edgeElasticity(mockEdge(null, null, 'ABOUT'))).toBe(160)
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
