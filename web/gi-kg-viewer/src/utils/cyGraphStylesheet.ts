/**
 * Shared Cytoscape stylesheet for GI/KG merged graphs (main canvas + rail mini preview).
 */
import type { NodeSingular } from 'cytoscape'
import { graphNodeFill } from './colors'

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
    const label = String(ele.data('label') ?? '')
    return sideLabelTextMarginX(w, label, maxWrapPx, gapPx, avgCharPx)
  }
}

function nodeLabelOffsetStyle(
  placement: GiKgNodeLabelPlacement,
  nw: number,
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

export function buildGiKgCyStylesheet(options?: {
  /** Main graph only — yellow ring for search hits. */
  includeSearchHit?: boolean
  /** Slightly smaller nodes for embedded previews. */
  compact?: boolean
  /** Default `side` (label to the right of the disc, no bbox/disc overlap). */
  nodeLabelPlacement?: GiKgNodeLabelPlacement
}): Record<string, unknown>[] {
  const compact = Boolean(options?.compact)
  const nw = compact ? 14 : 18
  const nh = compact ? 14 : 18
  const fs = compact ? '7px' : '9px'
  const tw = compact ? '72px' : '140px'
  const halo = cytoscapeNodeLabelHaloColorFromTheme()
  const placement: GiKgNodeLabelPlacement = options?.nodeLabelPlacement ?? 'side'
  const labelPos = nodeLabelOffsetStyle(placement, nw, nh, compact)
  const outlineW = compact ? 2 : 2.75

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
        'target-arrow-color': '#adb5bd',
        'line-color': '#adb5bd',
        label: 'data(label)',
        'font-size': compact ? '6px' : '8px',
        color: '#495057',
        'text-outline-width': compact ? 1.5 : 2,
        'text-outline-color': halo,
        'text-outline-opacity': 1,
      },
    },
  ]

  for (const t of VISUAL_TYPES) {
    style.push({
      selector: `node[type = "${t}"]`,
      style: {
        'background-color': graphNodeFill(t),
      },
    })
  }

  style.push({
    selector: 'node[type = "TopicCluster"]',
    style: {
      'background-color': graphNodeFill('TopicCluster'),
      'background-opacity': 0.22,
      'border-width': compact ? 1.5 : 2,
      'border-style': 'dashed',
      'border-color': '#5f3dc4',
      'border-opacity': 0.9,
      shape: 'roundrectangle',
      /** Tighter than generic nodes so merged topic clusters read as one tight region. */
      padding: compact ? '3px' : '6px',
    },
  })

  if (options?.includeSearchHit) {
    const hitStyle: Record<string, unknown> = {
      'border-width': 4,
      'border-color': '#fab005',
      'border-opacity': 0.9,
      width: 24,
      height: 24,
      'z-index': 10,
    }
    style.push({
      selector: 'node.search-hit',
      style: hitStyle,
    })
  }

  return style
}
