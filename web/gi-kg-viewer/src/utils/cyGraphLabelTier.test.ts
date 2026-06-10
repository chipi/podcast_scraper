import type { Core } from 'cytoscape'
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  GRAPH_LABEL_ZOOM_NONE_MAX,
  GRAPH_LABEL_ZOOM_SHORT_MAX,
  prefersReducedMotionQuery,
  syncGraphLabelTierClasses,
} from './cyGraphLabelTier'

const LABEL_TIER_CLASSES = 'graph-label-tier-none graph-label-tier-short graph-label-tier-full'

/**
 * Minimal stub of the Cytoscape `Core`/`NodeCollection` surface touched by
 * `syncGraphLabelTierClasses`: `zoom()`, `batch(fn)`, and
 * `nodes().removeClass()/addClass()`.
 */
function makeCore(zoom: number) {
  const removeClass = vi.fn().mockReturnThis()
  const addClass = vi.fn().mockReturnThis()
  const nodes = { removeClass, addClass }
  const core = {
    zoom: vi.fn(() => zoom),
    nodes: vi.fn(() => nodes),
    batch: vi.fn((fn: () => void) => fn()),
  }
  return { core: core as unknown as Core, raw: core, nodes, removeClass, addClass }
}

describe('cyGraphLabelTier', () => {
  it('exports zoom breakpoints aligned with VIEWER_GRAPH_SPEC §3.5', () => {
    expect(GRAPH_LABEL_ZOOM_NONE_MAX).toBe(0.5)
    expect(GRAPH_LABEL_ZOOM_SHORT_MAX).toBe(1.0)
  })

  describe('prefersReducedMotionQuery', () => {
    afterEach(() => {
      vi.unstubAllGlobals()
      vi.restoreAllMocks()
    })

    it('returns a boolean', () => {
      expect(typeof prefersReducedMotionQuery()).toBe('boolean')
    })

    it('returns false when window is undefined (SSR)', () => {
      vi.stubGlobal('window', undefined)
      expect(prefersReducedMotionQuery()).toBe(false)
    })

    it('returns false when matchMedia is not a function', () => {
      vi.stubGlobal('window', {})
      expect(prefersReducedMotionQuery()).toBe(false)
    })

    it('returns true when the reduce media query matches', () => {
      const matchMedia = vi.fn(() => ({ matches: true }))
      vi.stubGlobal('window', { matchMedia })
      expect(prefersReducedMotionQuery()).toBe(true)
      expect(matchMedia).toHaveBeenCalledWith('(prefers-reduced-motion: reduce)')
    })

    it('returns false when the reduce media query does not match', () => {
      vi.stubGlobal('window', { matchMedia: vi.fn(() => ({ matches: false })) })
      expect(prefersReducedMotionQuery()).toBe(false)
    })
  })

  describe('syncGraphLabelTierClasses', () => {
    it('assigns the none tier below the none-max breakpoint', () => {
      const { core, nodes, removeClass, addClass } = makeCore(0.49)
      syncGraphLabelTierClasses(core)
      expect(removeClass).toHaveBeenCalledWith(LABEL_TIER_CLASSES)
      expect(addClass).toHaveBeenCalledWith('graph-label-tier-none')
      void nodes
    })

    it('assigns the short tier exactly at the none-max boundary (inclusive)', () => {
      const { core, addClass } = makeCore(GRAPH_LABEL_ZOOM_NONE_MAX)
      syncGraphLabelTierClasses(core)
      expect(addClass).toHaveBeenCalledWith('graph-label-tier-short')
    })

    it('assigns the short tier between the breakpoints', () => {
      const { core, addClass } = makeCore(0.75)
      syncGraphLabelTierClasses(core)
      expect(addClass).toHaveBeenCalledWith('graph-label-tier-short')
    })

    it('assigns the short tier exactly at the short-max boundary (inclusive)', () => {
      const { core, addClass } = makeCore(GRAPH_LABEL_ZOOM_SHORT_MAX)
      syncGraphLabelTierClasses(core)
      expect(addClass).toHaveBeenCalledWith('graph-label-tier-short')
    })

    it('assigns the full tier just above the short-max boundary', () => {
      const { core, addClass } = makeCore(1.01)
      syncGraphLabelTierClasses(core)
      expect(addClass).toHaveBeenCalledWith('graph-label-tier-full')
    })

    it('assigns the full tier at high zoom', () => {
      const { core, addClass } = makeCore(5)
      syncGraphLabelTierClasses(core)
      expect(addClass).toHaveBeenCalledWith('graph-label-tier-full')
    })

    it('removes all tier classes before adding, inside a batch', () => {
      const { core, raw, removeClass, addClass } = makeCore(0.75)
      syncGraphLabelTierClasses(core)
      expect(raw.batch).toHaveBeenCalledTimes(1)
      expect(removeClass).toHaveBeenCalledWith(LABEL_TIER_CLASSES)
      // removeClass must run before addClass so exactly one tier remains.
      expect(removeClass.mock.invocationCallOrder[0]).toBeLessThan(
        addClass.mock.invocationCallOrder[0],
      )
    })
  })
})
