import { describe, expect, it } from 'vitest'
import {
  computeEpisodeTimelinePositions,
  deterministicJitter,
  weightedMeanXFromEpisodes,
} from './cyTimelineLayout'

const GEOM = { canvasWidth: 1000, canvasMidY: 500, jitterRange: 0 }

describe('deterministicJitter (RFC-080 V3)', () => {
  it('returns the same value for the same id (stable across re-runs)', () => {
    const a = deterministicJitter('node-a', 40)
    const b = deterministicJitter('node-a', 40)
    expect(a).toBe(b)
  })

  it('returns different values for different ids', () => {
    const a = deterministicJitter('alpha', 40)
    const b = deterministicJitter('beta', 40)
    expect(a).not.toBe(b)
  })

  it('stays within ±range', () => {
    for (const id of ['x', 'longer-id', 'episode:zzz', 't:1']) {
      const v = deterministicJitter(id, 40)
      expect(v).toBeGreaterThanOrEqual(-40)
      expect(v).toBeLessThanOrEqual(40)
    }
  })
})

describe('computeEpisodeTimelinePositions (RFC-080 V3)', () => {
  describe('quantile axis (default)', () => {
    it('places the i-th sorted episode at canvasWidth * i / (N-1)', () => {
      const episodes = [
        { id: 'e1', dateMs: 100 },
        { id: 'e2', dateMs: 200 },
        { id: 'e3', dateMs: 300 },
      ]
      const out = computeEpisodeTimelinePositions(episodes, GEOM)
      expect(out.positions.e1.x).toBeCloseTo(0)
      expect(out.positions.e2.x).toBeCloseTo(500)
      expect(out.positions.e3.x).toBeCloseTo(1000)
    })

    it('absorbs an outlier without compressing the cluster (quantile robustness)', () => {
      // 3 dates close together + 1 far in the future. Linear would
      // squash the early three into the leftmost ~3% of canvas; quantile
      // distributes them evenly because it ignores absolute spacing.
      const episodes = [
        { id: 'e1', dateMs: 1 },
        { id: 'e2', dateMs: 2 },
        { id: 'e3', dateMs: 3 },
        { id: 'outlier', dateMs: 1_000_000 },
      ]
      const out = computeEpisodeTimelinePositions(episodes, GEOM)
      // Quantile spacing: 0, 333.3, 666.6, 1000.
      expect(out.positions.e1.x).toBeCloseTo(0)
      expect(out.positions.e2.x).toBeCloseTo(333.33, 1)
      expect(out.positions.e3.x).toBeCloseTo(666.67, 1)
      expect(out.positions.outlier.x).toBeCloseTo(1000)
    })

    it('breaks ties on duplicate dates by id (alphabetical, stable)', () => {
      const episodes = [
        { id: 'b', dateMs: 100 },
        { id: 'a', dateMs: 100 },
        { id: 'c', dateMs: 200 },
      ]
      const out = computeEpisodeTimelinePositions(episodes, GEOM)
      // Sorted: a (date 100), b (date 100), c (date 200) → x = 0, 500, 1000
      expect(out.positions.a.x).toBeCloseTo(0)
      expect(out.positions.b.x).toBeCloseTo(500)
      expect(out.positions.c.x).toBeCloseTo(1000)
    })
  })

  describe('linear axis (opt-in)', () => {
    it('preserves date intervals via canvasWidth * (date - min) / (max - min)', () => {
      const episodes = [
        { id: 'e1', dateMs: 100 },
        { id: 'e2', dateMs: 200 },
        { id: 'e3', dateMs: 300 },
      ]
      const out = computeEpisodeTimelinePositions(episodes, GEOM, 'linear')
      expect(out.positions.e1.x).toBeCloseTo(0)
      expect(out.positions.e2.x).toBeCloseTo(500)
      expect(out.positions.e3.x).toBeCloseTo(1000)
    })

    it('compresses tight clusters when an outlier dominates max (the documented trade-off)', () => {
      const episodes = [
        { id: 'e1', dateMs: 1 },
        { id: 'e2', dateMs: 2 },
        { id: 'e3', dateMs: 3 },
        { id: 'outlier', dateMs: 1_000_000 },
      ]
      const out = computeEpisodeTimelinePositions(episodes, GEOM, 'linear')
      // e1..e3 collapse near x=0; outlier hits canvasWidth.
      expect(out.positions.e1.x).toBeCloseTo(0)
      expect(out.positions.e2.x).toBeCloseTo(0, 0)
      expect(out.positions.e3.x).toBeCloseTo(0, 0)
      expect(out.positions.outlier.x).toBeCloseTo(1000)
    })
  })

  describe('missing-date episodes', () => {
    it('parks undated episodes at canvasMidY with x = missingDateParkOffset (no silent rightmost)', () => {
      const episodes = [
        { id: 'e1', dateMs: 100 },
        { id: 'noDate', dateMs: null },
      ]
      const out = computeEpisodeTimelinePositions(episodes, GEOM)
      expect(out.positions.noDate.x).toBe(-60)
      expect(out.missingDateIds).toEqual(['noDate'])
    })

    it('parks ALL episodes at the missing spot when none have dates', () => {
      const episodes = [
        { id: 'a', dateMs: null },
        { id: 'b', dateMs: null },
      ]
      const out = computeEpisodeTimelinePositions(episodes, GEOM)
      expect(out.positions.a.x).toBe(-60)
      expect(out.positions.b.x).toBe(-60)
      expect(out.missingDateIds).toEqual(['a', 'b'])
    })

    it('honours custom missingDateParkOffset', () => {
      const out = computeEpisodeTimelinePositions(
        [{ id: 'x', dateMs: null }],
        { ...GEOM, missingDateParkOffset: -200 },
      )
      expect(out.positions.x.x).toBe(-200)
    })
  })

  it('applies deterministic y-jitter when jitterRange > 0', () => {
    const a = computeEpisodeTimelinePositions(
      [{ id: 'e1', dateMs: 100 }],
      { canvasWidth: 1000, canvasMidY: 500, jitterRange: 40 },
    )
    const b = computeEpisodeTimelinePositions(
      [{ id: 'e1', dateMs: 100 }],
      { canvasWidth: 1000, canvasMidY: 500, jitterRange: 40 },
    )
    expect(a.positions.e1.y).toBe(b.positions.e1.y)
    expect(Math.abs(a.positions.e1.y - 500)).toBeLessThanOrEqual(40)
  })
})

describe('weightedMeanXFromEpisodes (RFC-080 V3)', () => {
  it('returns the mean of supplied x positions', () => {
    expect(weightedMeanXFromEpisodes([100, 200, 300])).toBe(200)
  })

  it('returns null when no episodes are connected', () => {
    expect(weightedMeanXFromEpisodes([])).toBeNull()
  })

  it('handles a single connected episode', () => {
    expect(weightedMeanXFromEpisodes([42])).toBe(42)
  })
})
