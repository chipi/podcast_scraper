/**
 * Shared Cytoscape stylesheet for GI/KG merged graphs (main canvas + rail mini preview).
 */
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

export function buildGiKgCyStylesheet(options?: {
  /** Main graph only — yellow ring for search hits. */
  includeSearchHit?: boolean
  /** Slightly smaller nodes for embedded previews. */
  compact?: boolean
}): Record<string, unknown>[] {
  const compact = Boolean(options?.compact)
  const nw = compact ? 14 : 18
  const nh = compact ? 14 : 18
  const fs = compact ? '7px' : '9px'
  const tw = compact ? '72px' : '140px'

  const style: Record<string, unknown>[] = [
    {
      selector: 'node',
      style: {
        label: 'data(label)',
        'font-size': fs,
        'text-wrap': 'wrap',
        'text-max-width': tw,
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

  if (options?.includeSearchHit) {
    style.push({
      selector: 'node.search-hit',
      style: {
        'border-width': 4,
        'border-color': '#fab005',
        'border-opacity': 0.9,
        width: 24,
        height: 24,
        'z-index': 10,
      },
    })
  }

  return style
}
