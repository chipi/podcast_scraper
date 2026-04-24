import { describe, expect, it } from 'vitest'

import {
  scaleWeighted,
  weightedEdgeOpacity,
  weightedEdgeStyle,
  weightedEdgeWidth,
} from './cyEdgeWeight'

/**
 * Minimal ``EdgeSingular`` stub: the helpers only call ``ele.data()`` so we
 * don't drag Cytoscape into the test runtime. ``as unknown as …`` keeps
 * TypeScript from demanding the whole EdgeSingular surface.
 */
function fakeEdge(data: Record<string, unknown>): import('cytoscape').EdgeSingular {
  return {
    data: (key?: string) => (key === undefined ? data : (data as Record<string, unknown>)[key]),
  } as unknown as import('cytoscape').EdgeSingular
}

describe('scaleWeighted', () => {
  it('linearly interpolates within the domain', () => {
    expect(scaleWeighted(0.5, { min: 0, max: 1 }, { min: 0, max: 10 })).toBeCloseTo(5)
  })

  it('clamps below-domain values to range.min', () => {
    expect(scaleWeighted(-1, { min: 0, max: 1 }, { min: 2, max: 8 })).toBe(2)
  })

  it('clamps above-domain values to range.max', () => {
    expect(scaleWeighted(99, { min: 0, max: 1 }, { min: 2, max: 8 })).toBe(8)
  })

  it('returns range.min when domain is degenerate', () => {
    expect(scaleWeighted(5, { min: 1, max: 1 }, { min: 0.3, max: 0.9 })).toBe(0.3)
  })
})

describe('weightedEdgeStyle', () => {
  const opts = {
    propertyPath: 'properties.confidence',
    domain: { min: 0.25, max: 1.0 },
    range: { min: 0.35, max: 1.0 },
    fallback: 0.5,
  }

  it('reads nested property and scales', () => {
    const callback = weightedEdgeStyle(opts)
    const raw = 0.625 // midpoint of [0.25, 1.0]
    const edge = fakeEdge({ properties: { confidence: raw } })
    // Should land at the midpoint of [0.35, 1.0] = 0.675
    expect(callback(edge)).toBeCloseTo(0.675)
  })

  it('returns fallback when property is missing', () => {
    const callback = weightedEdgeStyle(opts)
    expect(callback(fakeEdge({}))).toBe(0.5)
    expect(callback(fakeEdge({ properties: {} }))).toBe(0.5)
    expect(callback(fakeEdge({ properties: { confidence: null } }))).toBe(0.5)
  })

  it('parses stringified numeric properties', () => {
    // Cytoscape sometimes round-trips data through JSON that coerces
    // numbers to strings; the helper should still map them.
    const callback = weightedEdgeStyle(opts)
    const edge = fakeEdge({ properties: { confidence: '0.625' } })
    expect(callback(edge)).toBeCloseTo(0.675)
  })

  it('returns fallback for non-finite / non-parseable values', () => {
    const callback = weightedEdgeStyle(opts)
    expect(callback(fakeEdge({ properties: { confidence: 'abc' } }))).toBe(0.5)
    expect(callback(fakeEdge({ properties: { confidence: NaN } }))).toBe(0.5)
    expect(callback(fakeEdge({ properties: { confidence: Infinity } }))).toBe(0.5)
  })

  it('resolves deeper property paths', () => {
    const callback = weightedEdgeStyle({ ...opts, propertyPath: 'a.b.c' })
    // Domain [0.25, 1.0], range [0.35, 1.0], raw 0.5 → t = 0.333… →
    // 0.35 + 0.333 * 0.65 ≈ 0.567
    expect(callback(fakeEdge({ a: { b: { c: 0.5 } } }))).toBeCloseTo(0.567, 2)
  })
})

describe('weightedEdgeOpacity (#664 convenience)', () => {
  const callback = weightedEdgeOpacity()

  it('maps floor-cosine 0.25 to opacity 0.35', () => {
    expect(callback(fakeEdge({ properties: { confidence: 0.25 } }))).toBeCloseTo(0.35)
  })

  it('maps max-cosine 1.0 to opacity 1.0', () => {
    expect(callback(fakeEdge({ properties: { confidence: 1.0 } }))).toBeCloseTo(1.0)
  })

  it('scales intermediate values monotonically', () => {
    const low = callback(fakeEdge({ properties: { confidence: 0.4 } }))
    const mid = callback(fakeEdge({ properties: { confidence: 0.6 } }))
    const high = callback(fakeEdge({ properties: { confidence: 0.85 } }))
    expect(low).toBeLessThan(mid)
    expect(mid).toBeLessThan(high)
  })

  it('falls back to 1.0 for legacy edges without confidence (no visual regression)', () => {
    expect(callback(fakeEdge({}))).toBe(1)
    expect(callback(fakeEdge({ properties: {} }))).toBe(1)
  })
})

describe('weightedEdgeWidth', () => {
  it('scales base width in [0.75x, 1.5x] over the confidence range', () => {
    const callback = weightedEdgeWidth(2)
    expect(callback(fakeEdge({ properties: { confidence: 0.25 } }))).toBeCloseTo(1.5)
    expect(callback(fakeEdge({ properties: { confidence: 1.0 } }))).toBeCloseTo(3.0)
  })

  it('falls back to base width for edges without confidence', () => {
    const callback = weightedEdgeWidth(2)
    expect(callback(fakeEdge({}))).toBe(2)
  })
})
