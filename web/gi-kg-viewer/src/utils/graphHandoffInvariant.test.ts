/**
 * Unit tests for the self-healing invariant predicate (T2a).
 *
 * Pure-function tests — no Vue, no Pinia, no Cytoscape. Run as part of the
 * standard Vitest suite. If a future refactor breaks the set-difference
 * computation or the reconcile-action decision rules, these go red first.
 */

import { describe, expect, it } from 'vitest'
import {
  computeNodeIdSetDifference,
  decideReconcileAction,
  MAX_TARGETED_RECONCILE_MISSING,
} from './graphHandoffInvariant'

describe('computeNodeIdSetDifference', () => {
  it('returns empty arrays when sets are identical', () => {
    const { missing, extra } = computeNodeIdSetDifference(
      ['a', 'b', 'c'],
      ['a', 'b', 'c'],
    )
    expect(missing).toEqual([])
    expect(extra).toEqual([])
  })

  it('returns missing when expected has IDs not in actual', () => {
    const { missing, extra } = computeNodeIdSetDifference(['a', 'b', 'c'], ['a'])
    expect(missing.sort()).toEqual(['b', 'c'])
    expect(extra).toEqual([])
  })

  it('returns extra when actual has IDs not in expected', () => {
    const { missing, extra } = computeNodeIdSetDifference(['a'], ['a', 'b', 'c'])
    expect(missing).toEqual([])
    expect(extra.sort()).toEqual(['b', 'c'])
  })

  it('returns both missing and extra in mixed cases', () => {
    const { missing, extra } = computeNodeIdSetDifference(
      ['a', 'b', 'c'],
      ['b', 'c', 'd', 'e'],
    )
    expect(missing).toEqual(['a'])
    expect(extra.sort()).toEqual(['d', 'e'])
  })

  it('handles empty inputs', () => {
    expect(computeNodeIdSetDifference([], [])).toEqual({ missing: [], extra: [] })
    expect(computeNodeIdSetDifference(['a'], [])).toEqual({
      missing: ['a'],
      extra: [],
    })
    expect(computeNodeIdSetDifference([], ['a'])).toEqual({
      missing: [],
      extra: ['a'],
    })
  })

  it('accepts Sets directly', () => {
    const result = computeNodeIdSetDifference(
      new Set(['a', 'b']),
      new Set(['b', 'c']),
    )
    expect(result.missing).toEqual(['a'])
    expect(result.extra).toEqual(['c'])
  })

  it('deduplicates when given iterables with duplicates', () => {
    const result = computeNodeIdSetDifference(
      ['a', 'a', 'b', 'b'],
      ['a', 'a'],
    )
    expect(result.missing).toEqual(['b'])
    expect(result.extra).toEqual([])
  })

  it('handles realistic graph node IDs', () => {
    const expected = [
      'g:episode:ep1',
      '__unified_ep__:e1',
      'g:topic:alpha',
      'g:topic:beta',
    ]
    const actual = [
      'g:episode:ep1',
      'g:topic:alpha',
      // missing __unified_ep__:e1 and g:topic:beta
    ]
    const { missing, extra } = computeNodeIdSetDifference(expected, actual)
    expect(missing.sort()).toEqual(['__unified_ep__:e1', 'g:topic:beta'])
    expect(extra).toEqual([])
  })
})

describe('decideReconcileAction', () => {
  it("returns 'ok' when no missing IDs", () => {
    expect(decideReconcileAction([], false)).toBe('ok')
    expect(decideReconcileAction([], true)).toBe('ok')
  })

  it("returns 'reconcile' for small missing.length on first attempt", () => {
    expect(decideReconcileAction(['a'], false)).toBe('reconcile')
    expect(decideReconcileAction(['a', 'b', 'c'], false)).toBe('reconcile')
  })

  it("returns 'accept-divergence' if alreadyRetried=true", () => {
    expect(decideReconcileAction(['a'], true)).toBe('accept-divergence')
    expect(decideReconcileAction(['a', 'b', 'c'], true)).toBe('accept-divergence')
  })

  it("returns 'accept-divergence' when missing.length >= MAX_TARGETED_RECONCILE_MISSING", () => {
    const lots = Array.from(
      { length: MAX_TARGETED_RECONCILE_MISSING },
      (_, i) => `id-${i}`,
    )
    expect(decideReconcileAction(lots, false)).toBe('accept-divergence')
    expect(decideReconcileAction([...lots, 'extra'], false)).toBe(
      'accept-divergence',
    )
  })

  it("returns 'reconcile' at MAX_TARGETED_RECONCILE_MISSING - 1 boundary", () => {
    const justUnderLimit = Array.from(
      { length: MAX_TARGETED_RECONCILE_MISSING - 1 },
      (_, i) => `id-${i}`,
    )
    expect(decideReconcileAction(justUnderLimit, false)).toBe('reconcile')
  })

  it('exposes MAX_TARGETED_RECONCILE_MISSING as a documented constant', () => {
    // Pin the documented value so a refactor adjusting the threshold has
    // to update the test deliberately (and any downstream docs).
    expect(MAX_TARGETED_RECONCILE_MISSING).toBe(20)
  })
})
