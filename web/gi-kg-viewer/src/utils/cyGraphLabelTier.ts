/**
 * Zoom-driven graph label tiers (WIP §3.5). Shared by main GraphCanvas and rail minimap preview.
 */
import type { Core } from 'cytoscape'

/** Below this zoom, labels are hidden (`graph-label-tier-none`). */
export const GRAPH_LABEL_ZOOM_NONE_MAX = 0.5

/** Through this zoom (inclusive), short-tier rules apply (`graph-label-tier-short`). */
export const GRAPH_LABEL_ZOOM_SHORT_MAX = 1.0

const LABEL_TIER_CLASSES =
  'graph-label-tier-none graph-label-tier-short graph-label-tier-full' as const

export type GraphLabelTierClass =
  | 'graph-label-tier-none'
  | 'graph-label-tier-short'
  | 'graph-label-tier-full'

/** `prefers-reduced-motion: reduce` for Cytoscape transition toggles (SSR-safe). */
export function prefersReducedMotionQuery(): boolean {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return false
  }
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

/** Assign exactly one of the mutually exclusive `graph-label-tier-*` classes on every node. */
export function syncGraphLabelTierClasses(core: Core): void {
  const z = core.zoom()
  let tier: GraphLabelTierClass
  if (z < GRAPH_LABEL_ZOOM_NONE_MAX) {
    tier = 'graph-label-tier-none'
  } else if (z <= GRAPH_LABEL_ZOOM_SHORT_MAX) {
    tier = 'graph-label-tier-short'
  } else {
    tier = 'graph-label-tier-full'
  }
  core.batch(() => {
    core.nodes().removeClass(LABEL_TIER_CLASSES)
    core.nodes().addClass(tier)
  })
}
