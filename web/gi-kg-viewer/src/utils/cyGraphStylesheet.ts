/**
 * Shared Cytoscape stylesheet for GI/KG merged graphs (main canvas + rail mini preview).
 */
import type { EdgeSingular, NodeSingular } from 'cytoscape'
import { graphNodeFill } from './colors'
import { weightedEdgeOpacity, weightedEdgeWidth } from './cyEdgeWeight'

export function cytoscapeNodeLabelColorFromTheme(): string {
  try {
    const v = getComputedStyle(document.documentElement)
      .getPropertyValue('--ps-canvas-foreground')
      .trim()
    if (v) return v
  } catch {
    /* ignore */
  }
  return '#e5e8eb'
}

/** Canvas fill — used as label halo so text does not blend into edge strokes. */
export function cytoscapeNodeLabelHaloColorFromTheme(): string {
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue('--ps-canvas').trim()
    if (v) return v
  } catch {
    /* ignore */
  }
  return '#111418'
}

const VISUAL_TYPES = [
  'Episode',
  'Insight',
  'Quote',
  'Speaker',
  'Topic',
  'Entity_person',
  'Entity_organization',
  'Podcast',
] as const

/** Tier 1–2 (WIP §3.5): show truncated labels in the medium-zoom band. */
const LABEL_SHORT_TIER_TYPES = [
  'Insight',
  'Topic',
  'TopicCluster',
  'Entity_person',
  'Entity_organization',
] as const

/** Main profile node diameters (WIP §3.1), before compact scale. */
const NODE_DIAMETER_MAIN_PX: Record<string, number> = {
  Insight: 44,
  Topic: 40,
  TopicCluster: 48,
  Entity_person: 34,
  Entity_organization: 26,
  Quote: 22,
  Speaker: 18,
  Episode: 18,
  Podcast: 18,
}

function scaledNodeSize(type: string, compact: boolean): number {
  const base = NODE_DIAMETER_MAIN_PX[type] ?? 18
  if (!compact) {
    return base
  }
  return Math.max(12, Math.round((base * 14) / 18))
}

function graphNodeOpacity(ele: NodeSingular): number {
  const raw = ele.data('recencyWeight')
  const rw = Number(raw)
  /** Parser clamps `recencyWeight` to [0.4, 1]; keep the same floor here (WIP §4.1). */
  const recency = Number.isFinite(rw) ? Math.max(0.4, Math.min(1, rw)) : 1
  if (ele.hasClass('graph-dimmed')) {
    return 0.4 * recency
  }
  if (ele.hasClass('graph-neighbour')) {
    return 0.85 * recency
  }
  if (ele.hasClass('graph-focused')) {
    return recency
  }
  return recency
}

function graphEdgeOpacity(ele: EdgeSingular): number {
  if (ele.hasClass('graph-edge-dimmed')) {
    return 0.2
  }
  if (ele.hasClass('graph-edge-neighbour')) {
    return 0.9
  }
  return 0.6
}

function insightBackgroundOpacity(ele: NodeSingular): number {
  const c = Number(ele.data('confidenceOpacity'))
  return Number.isFinite(c) ? Math.max(0.35, Math.min(1, c)) : 0.85
}

/**
 * Topic hub border from normalized `degreeHeat` (0–1). **0** when there is no heat
 * signal so expand / search-hit / selection rings are not visually crowded (UXS-004).
 */
export function topicDegreeHeatBorderWidthPx(degreeHeat: unknown, compact: boolean): number {
  const heat = Number(degreeHeat)
  const h = Number.isFinite(heat) ? Math.max(0, Math.min(1, heat)) : 0
  if (h <= 0) {
    return 0
  }
  const base = 1 + h * 3
  return compact ? Math.max(1, base * (14 / 18)) : base
}

function topicBorderWidthForHeat(ele: NodeSingular, compact: boolean): number {
  return topicDegreeHeatBorderWidthPx(ele.data('degreeHeat'), compact)
}

/** Where to put the caption relative to the node body (Cytoscape `text-*` props). */
export type GiKgNodeLabelPlacement = 'side' | 'above' | 'below'

/**
 * Half of the label's horizontal extent in model space (used with `text-halign: center`).
 * Caps at half of `text-max-width`; approximates wrapped lines so long labels still reserve
 * ~full wrap width. Short labels get a smaller half-width so `text-margin-x` stays tight.
 */
export function estimateLabelHalfWidthPx(
  label: string,
  maxWrapPx: number,
  avgCharPx: number,
  minHalfPx = 10,
): number {
  const t = label.trim()
  if (!t) {
    return minHalfPx
  }
  const approxLineCount = Math.max(1, Math.ceil((t.length * avgCharPx) / maxWrapPx))
  const lineVisualWidth =
    approxLineCount <= 1 ? Math.min(maxWrapPx, t.length * avgCharPx) : maxWrapPx
  const half = Math.min(maxWrapPx / 2, lineVisualWidth / 2)
  return Math.max(minHalfPx, half)
}

/**
 * `text-margin-x` for side labels: label **center** sits at node center + this value.
 * Need margin ≥ r + (rendered half-width) + gap so the label bbox clears the disc.
 */
export function sideLabelTextMarginX(
  nodeBodyWidthPx: number,
  label: string,
  maxWrapPx: number,
  gapPx: number,
  avgCharPx: number,
): number {
  const t = label.trim()
  if (!t) {
    return 0
  }
  const r = nodeBodyWidthPx / 2
  const halfW = estimateLabelHalfWidthPx(label, maxWrapPx, avgCharPx)
  /** Extra px: outline + background padding make bbox wider than char estimate (e2e: disc vs label rect). */
  return Math.round(r + halfW + gapPx + 4)
}

/** Cytoscape style callback — append after `buildGiKgCyStylesheet` for `nodeLabelPlacement: side`. */
export function cytoscapeSideLabelMarginXCallback(compact: boolean): (ele: NodeSingular) => number {
  const maxWrapPx = compact ? 72 : 140
  const gapPx = compact ? 2 : 3
  const avgCharPx = compact ? 4.75 : 5.5
  return (ele: NodeSingular) => {
    const w = ele.width()
    const useShort = ele.hasClass('graph-label-tier-short')
    const shortL = String(ele.data('shortLabel') ?? '')
    const fullL = String(ele.data('label') ?? '')
    const label = useShort && shortL.trim() ? shortL : fullL
    return sideLabelTextMarginX(w, label, maxWrapPx, gapPx, avgCharPx)
  }
}

function nodeLabelOffsetStyle(
  placement: GiKgNodeLabelPlacement,
  _nw: number,
  nh: number,
  compact: boolean,
): {
  'text-valign': 'top' | 'center' | 'bottom'
  'text-halign': 'left' | 'center' | 'right'
  'text-margin-x': number
  'text-margin-y': number
} {
  const halfH = Math.round(nh / 2)
  const lift = compact ? 4 : 6
  switch (placement) {
    case 'above':
      return {
        'text-valign': 'top',
        'text-halign': 'center',
        'text-margin-x': 0,
        /** Negative: anchor at top edge moves up so the block clears the disc. */
        'text-margin-y': -(halfH + lift),
      }
    case 'below':
      return {
        'text-valign': 'bottom',
        'text-halign': 'center',
        'text-margin-x': 0,
        'text-margin-y': halfH + lift,
      }
    case 'side':
      return {
        'text-valign': 'center',
        'text-halign': 'center',
        /** Real offset comes from `cytoscapeSideLabelMarginXCallback` (per-label width). */
        'text-margin-x': 0,
        'text-margin-y': 0,
      }
  }
}

function labelShortTierSelector(): string {
  return LABEL_SHORT_TIER_TYPES.map((t) => `node.graph-label-tier-short[type = "${t}"]`).join(', ')
}

export function buildGiKgCyStylesheet(options?: {
  /** Main graph only — yellow ring for search hits. */
  includeSearchHit?: boolean
  /** Slightly smaller nodes for embedded previews. */
  compact?: boolean
  /** Default `side` (label to the right of the disc, no bbox/disc overlap). */
  nodeLabelPlacement?: GiKgNodeLabelPlacement
  /** When true, skip opacity transition (matches `prefers-reduced-motion: reduce`). */
  prefersReducedMotion?: boolean
}): Record<string, unknown>[] {
  const compact = Boolean(options?.compact)
  const nw = scaledNodeSize('Episode', compact)
  const nh = scaledNodeSize('Episode', compact)
  const fs = compact ? '7px' : '9px'
  const tw = compact ? '72px' : '140px'
  const halo = cytoscapeNodeLabelHaloColorFromTheme()
  const placement: GiKgNodeLabelPlacement = options?.nodeLabelPlacement ?? 'side'
  const labelPos = nodeLabelOffsetStyle(placement, nw, nh, compact)
  const outlineW = compact ? 2 : 2.75
  const reducedMotion = Boolean(options?.prefersReducedMotion)
  const transitionStyle: Record<string, unknown> = reducedMotion
    ? {}
    : {
        'transition-property': 'opacity, border-opacity, text-opacity',
        'transition-duration': 0.15,
      }

  const style: Record<string, unknown>[] = [
    {
      selector: 'node',
      style: {
        label: 'data(label)',
        'font-size': fs,
        'text-wrap': 'wrap',
        'text-max-width': tw,
        ...labelPos,
        'text-outline-width': outlineW,
        'text-outline-color': halo,
        'text-outline-opacity': 1,
        'text-background-color': halo,
        'text-background-opacity': 0.82,
        'text-background-padding': compact ? '2px' : '3px',
        'text-background-shape': 'roundrectangle',
        'min-zoomed-font-size': compact ? 5 : 6,
        'background-color': '#868e96',
        color: cytoscapeNodeLabelColorFromTheme(),
        width: nw,
        height: nh,
        'border-width': 0,
        opacity: (ele: NodeSingular) => graphNodeOpacity(ele),
        ...transitionStyle,
      },
    },
    {
      selector: 'node:selected',
      style: {
        'border-width': compact ? 2 : 3,
        'border-color': '#228be6',
        'border-opacity': 1,
      },
    },
    {
      selector: 'edge',
      style: {
        width: compact ? 1 : 1.5,
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': 'var(--ps-muted)',
        'line-color': 'var(--ps-muted)',
        'line-style': 'solid',
        label: '',
        'font-size': compact ? '6px' : '8px',
        color: '#495057',
        'text-outline-width': compact ? 1.5 : 2,
        'text-outline-color': halo,
        'text-outline-opacity': 1,
        opacity: (ele: EdgeSingular) => graphEdgeOpacity(ele),
        ...transitionStyle,
      },
    },
  ]

  for (const t of VISUAL_TYPES) {
    const side = scaledNodeSize(t, compact)
    style.push({
      selector: `node[type = "${t}"]`,
      style: {
        'background-color': graphNodeFill(t),
        width: side,
        height: side,
      },
    })
  }

  style.push({
    selector: 'node[type = "Insight"]',
    style: {
      'background-opacity': (ele: NodeSingular) => insightBackgroundOpacity(ele),
    },
  })

  style.push({
    selector: 'node[type = "Topic"]',
    style: {
      'border-width': (ele: NodeSingular) => topicBorderWidthForHeat(ele, compact),
      'border-color': 'var(--ps-kg)',
      'border-opacity': 0.55,
    },
  })

  style.push({
    selector: 'node[type = "Topic"]:selected',
    style: {
      'border-width': compact ? 2 : 3,
      'border-color': '#228be6',
      'border-opacity': 1,
    },
  })

  const tcPad = compact ? '14px' : '18px'
  style.push({
    selector: 'node[type = "TopicCluster"]',
    style: {
      'background-color': 'var(--ps-kg)',
      'background-opacity': 0.06,
      'border-width': compact ? 1.25 : 1.5,
      'border-style': 'dashed',
      'border-color': 'var(--ps-kg)',
      'border-opacity': 0.4,
      shape: 'roundrectangle',
      padding: tcPad,
    },
  })

  style.push({
    selector: 'node[type = "Insight"], node[type = "Topic"]',
    style: {
      'shadow-blur': compact ? 6 : 8,
      'shadow-color': 'var(--ps-border)',
      'shadow-offset-x': 0,
      'shadow-offset-y': 2,
      'shadow-opacity': 0.6,
    },
  })

  style.push({
    selector: 'node[type = "Topic"].graph-topic-heat-high',
    style: {
      'shadow-blur': 12,
      'shadow-color': 'var(--ps-kg)',
      'shadow-offset-x': 0,
      'shadow-offset-y': 2,
      'shadow-opacity': 0.5,
    },
  })

  style.push({
    selector: 'node.graph-label-tier-none',
    style: {
      'text-opacity': 0,
    },
  })
  style.push({
    selector: 'node.graph-label-tier-short',
    style: {
      'text-opacity': 0,
    },
  })
  style.push({
    selector: labelShortTierSelector(),
    style: {
      label: 'data(shortLabel)',
      'text-opacity': 1,
    },
  })
  style.push({
    selector: 'node.graph-label-tier-full',
    style: {
      label: 'data(label)',
      'text-opacity': 1,
    },
  })

  const edgeStrokes: { selector: string; style: Record<string, unknown> }[] = [
    {
      selector: 'edge[edgeType = "HAS_INSIGHT"]',
      style: {
        width: compact ? 1.5 : 2,
        'line-color': 'var(--ps-primary)',
        'line-style': 'solid',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': 'var(--ps-primary)',
      },
    },
    {
      // #664 + #656-foundation: ABOUT edges now carry
      // ``properties.confidence`` (cosine similarity, 0.25–1.0). Map it to
      // line-opacity + width so strong "about X" edges stand out and weak
      // ones fade. Legacy edges without the property fall back to uniform
      // rendering via the helpers' defaults.
      selector: 'edge[edgeType = "ABOUT"]',
      style: {
        width: weightedEdgeWidth(compact ? 1.5 : 2),
        'line-color': 'var(--ps-gi)',
        'line-style': 'solid',
        'line-opacity': weightedEdgeOpacity(),
        'target-arrow-shape': 'none',
      },
    },
    {
      selector: 'edge[edgeType = "SUPPORTED_BY"]',
      style: {
        width: compact ? 0.85 : 1,
        'line-color': 'var(--ps-muted)',
        'line-style': 'dashed',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': 'var(--ps-muted)',
      },
    },
    {
      selector: 'edge[edgeType = "RELATED_TO"]',
      style: {
        width: compact ? 1 : 1,
        'line-color': 'var(--ps-kg)',
        'line-style': 'solid',
        'target-arrow-shape': 'none',
      },
    },
    {
      selector: 'edge[edgeType = "MENTIONS"]',
      style: {
        width: compact ? 1 : 1,
        'line-color': 'var(--ps-muted)',
        'line-style': 'dotted',
        'target-arrow-shape': 'none',
      },
    },
    {
      selector: 'edge[edgeType = "SPOKE_IN"]',
      style: {
        width: compact ? 1.5 : 2,
        'line-color': 'var(--ps-primary)',
        'line-style': 'solid',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': 'var(--ps-primary)',
      },
    },
    {
      selector: 'edge[edgeType = "HAS_MEMBER"]',
      style: {
        width: compact ? 1.25 : 1.5,
        'line-color': 'var(--ps-kg)',
        'line-style': 'solid',
        'target-arrow-shape': 'none',
        'line-opacity': 0.6,
      },
    },
  ]
  for (const r of edgeStrokes) {
    style.push(r)
  }

  /** Explicit fallback when `edgeType` is missing or `(unknown)` after canonicalisation. */
  style.push({
    selector: 'edge[edgeType = "(unknown)"]',
    style: {
      width: compact ? 1 : 1.5,
      'curve-style': 'bezier',
      'line-color': 'var(--ps-muted)',
      'line-style': 'solid',
      'target-arrow-shape': 'triangle',
      'target-arrow-color': 'var(--ps-muted)',
    },
  })

  style.push({
    selector: 'edge[edgeType = "_tc_cohesion"]',
    style: {
      'line-opacity': 0,
      'target-arrow-shape': 'none',
      width: 0,
      label: '',
      events: 'no',
    },
  })

  style.push({
    selector: 'edge.graph-edge-dimmed',
    style: {
      opacity: 0.2,
    },
  })
  style.push({
    selector: 'edge.graph-edge-neighbour',
    style: {
      opacity: 0.9,
    },
  })

  /** Eligible Topic / Person / Entity for plain dbl-click cross-episode expand. */
  const expandEligibleRing = compact ? 1.5 : 2
  const expandSeedRing = compact ? 2.25 : 3
  style.push({
    selector: 'node.graph-expand-eligible',
    style: {
      'border-width': expandEligibleRing,
      'border-style': 'solid',
      'border-color': '#14b8a6',
      'border-opacity': 0.92,
    },
  })
  style.push({
    selector: 'node.graph-expand-seed',
    style: {
      'border-width': expandSeedRing,
      'border-style': 'solid',
      'border-color': '#748ffc',
      'border-opacity': 0.95,
    },
  })

  if (options?.includeSearchHit) {
    style.push({
      selector: 'node.search-hit',
      style: {
        'border-width': 4,
        'border-color': '#fab005',
        'border-opacity': 0.9,
        'z-index': 10,
      },
    })
  }

  return style
}
