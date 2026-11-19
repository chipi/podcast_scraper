import { describe, expect, it } from 'vitest'
import {
  computeRadialPositions,
  radialRingRadii,
} from './cyRadialLayout'

describe('radialRingRadii (RFC-080 V4)', () => {
  it('uses default 120 inner / 240 outer when no max-radius is given', () => {
    const r = radialRingRadii()
    expect(r.r1).toBe(120)
    expect(r.r2).toBe(240)
  })

  it('grows the inner radius to 2.5× the largest ring-1 node radius (V5 interaction)', () => {
    // 60px node → radius 30 → r1 = max(120, 30 * 2.5) = max(120, 75) = 120
    // (still bounded by floor)
    expect(radialRingRadii({ maxRing1NodeRadius: 30 }).r1).toBe(120)
    // 60px node radius (max V5 Topic): r1 = max(120, 60 * 2.5) = 150
    expect(radialRingRadii({ maxRing1NodeRadius: 60 }).r1).toBe(150)
  })

  it('keeps outer radius proportional via outerRingFactor (default 2)', () => {
    expect(radialRingRadii({ baseR1: 200 }).r2).toBe(400)
    expect(radialRingRadii({ baseR1: 200, outerRingFactor: 3 }).r2).toBe(600)
  })

  it('clamps negative maxRing1NodeRadius to 0 (defensive)', () => {
    expect(radialRingRadii({ maxRing1NodeRadius: -50 }).r1).toBe(120)
  })
})

describe('computeRadialPositions (RFC-080 V4)', () => {
  it('places the centre node at (0, 0)', () => {
    const out = computeRadialPositions('c', [], [])
    expect(out.positions.c).toEqual({ x: 0, y: 0 })
  })

  it('distributes ring 1 nodes uniformly around the inner circle', () => {
    const out = computeRadialPositions('c', ['n1', 'n2', 'n3', 'n4'], [])
    // 4 nodes at 90-degree spacing around r1=120.
    expect(out.positions.n1.x).toBeCloseTo(120)
    expect(out.positions.n1.y).toBeCloseTo(0)
    expect(out.positions.n2.x).toBeCloseTo(0)
    expect(out.positions.n2.y).toBeCloseTo(120)
    expect(out.positions.n3.x).toBeCloseTo(-120)
    expect(out.positions.n3.y).toBeCloseTo(0)
    expect(out.positions.n4.x).toBeCloseTo(0)
    expect(out.positions.n4.y).toBeCloseTo(-120)
  })

  it('places ring 2 nodes on a larger circle (default 2× ring 1 radius)', () => {
    const out = computeRadialPositions('c', [], ['x'])
    // Single ring-2 node lands at (r2, 0).
    expect(out.positions.x.x).toBeCloseTo(240)
    expect(out.positions.x.y).toBeCloseTo(0)
  })

  it('adapts ring radii to V5 node sizes (composes with V4×V5 interaction)', () => {
    const out = computeRadialPositions('c', ['big'], [], { maxRing1NodeRadius: 60 })
    // r1 grows to max(120, 60 * 2.5) = 150 — node at (150, 0).
    expect(out.r1).toBe(150)
    expect(out.positions.big.x).toBeCloseTo(150)
  })

  it('skips a ring-2 entry that is also in ring 1 (de-dup)', () => {
    const out = computeRadialPositions('c', ['shared'], ['shared', 'far'])
    // 'shared' placed in ring 1 only; 'far' is the only ring 2 node.
    expect(out.positions.shared.x).toBeCloseTo(120) // ring 1 radius
    expect(out.positions.far.x).toBeCloseTo(240) // ring 2 radius
  })

  it('skips ring entries equal to the centre id', () => {
    const out = computeRadialPositions('c', ['c', 'a'], ['c', 'b'])
    // Centre stays at (0, 0); 'a' is the only ring-1 node; 'b' the only ring-2 node.
    expect(out.positions.c).toEqual({ x: 0, y: 0 })
    expect(out.positions.a.x).toBeCloseTo(120)
    expect(out.positions.b.x).toBeCloseTo(240)
  })

  it('returns a position only for centre + supplied ring nodes (callers hide everything else)', () => {
    const out = computeRadialPositions('c', ['a'], ['b'])
    expect(Object.keys(out.positions).sort()).toEqual(['a', 'b', 'c'])
  })

  it('handles empty input (just the centre at origin)', () => {
    const out = computeRadialPositions('c', [], [])
    expect(out.positions).toEqual({ c: { x: 0, y: 0 } })
  })

  it('produces deterministic angular ordering across re-runs', () => {
    const a = computeRadialPositions('c', ['n1', 'n2', 'n3'], [])
    const b = computeRadialPositions('c', ['n1', 'n2', 'n3'], [])
    expect(a.positions).toEqual(b.positions)
  })
})
