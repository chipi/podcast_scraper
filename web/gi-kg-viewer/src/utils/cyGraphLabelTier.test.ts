import { describe, expect, it } from 'vitest'
import { GRAPH_LABEL_ZOOM_NONE_MAX, GRAPH_LABEL_ZOOM_SHORT_MAX, prefersReducedMotionQuery } from './cyGraphLabelTier'

describe('cyGraphLabelTier', () => {
  it('exports zoom breakpoints aligned with GRAPH-VISUAL-STYLING WIP §3.5', () => {
    expect(GRAPH_LABEL_ZOOM_NONE_MAX).toBe(0.5)
    expect(GRAPH_LABEL_ZOOM_SHORT_MAX).toBe(1.0)
  })

  it('prefersReducedMotionQuery returns a boolean', () => {
    expect(typeof prefersReducedMotionQuery()).toBe('boolean')
  })
})
