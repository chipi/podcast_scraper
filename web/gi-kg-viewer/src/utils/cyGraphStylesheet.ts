/**
 * Shared Cytoscape stylesheet for GI/KG merged graphs (main canvas + rail mini preview).
 *
 * This file defines every base-layer visual rule the graph applies —
 * type-based shape + colour, edge width tiers, confidence opacity, temporal
 * recency fade, theme-region tints, degree-heat sizing, bridge ring, and
 * enricher-lens class selectors. See ``docs/guides/GRAPH_VISUALIZATION_GUIDE.md``
 * for the prose reference explaining WHY each rule exists and WHEN it
 * fires. That guide is the authoritative "what does this dial do" reference;
 * keep changes here in sync with the guide's node/edge encoding sections.
 */
import type { EdgeSingular, NodeSingular } from 'cytoscape'
import { graphNodeFill } from './colors'
import { aboutConfidenceOpacity, aboutConfidenceWidth } from './cyEdgeWeight'
import { THEME_REGION_PALETTE } from './themeRegionPalette'

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

/** Canvas fill — used as label halo so text does not blend into edge strokes.
 *  graph-v3 E — prefer --ps-graph-canvas (the graph-scoped darker token) so
 *  halos match the canvas exactly instead of leaving a ~3-shade lighter
 *  rectangle behind each label. Falls through to --ps-canvas on light theme
 *  (where the graph-canvas var aliases --ps-canvas). */
export function cytoscapeNodeLabelHaloColorFromTheme(): string {
  try {
    const root = document.documentElement
    const g = getComputedStyle(root).getPropertyValue('--ps-graph-canvas').trim()
    if (g) return g
    const v = getComputedStyle(root).getPropertyValue('--ps-canvas').trim()
    if (v) return v
  } catch {
    /* ignore */
  }
  return '#0a0d10'
}

/**
 * Cytoscape's stylesheet parser does not resolve CSS `var(--…)` references —
 * it expects literal color values. Resolve at build time via
 * `getComputedStyle()` (matches the live theme), with a hex fallback for SSR
 * / jsdom / pre-mount contexts where the document hasn't applied tokens yet.
 *
 * Default fallbacks track the dark-theme palette in `src/styles/tokens.css`.
 */
const PS_TOKEN_FALLBACKS: Record<string, string> = {
  '--ps-canvas': '#111418',
  '--ps-graph-canvas': '#0a0d10',
  '--ps-canvas-foreground': '#e5e8eb',
  '--ps-border': '#404854',
  '--ps-muted': '#8f99a8',
  '--ps-primary': '#4c90f0',
  '--ps-warning': '#ec9a3c',
  '--ps-gi': '#7dd3a0',
  '--ps-kg': '#c4a8ff',
}

export function resolveThemeColor(varName: string, fallback?: string): string {
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue(varName).trim()
    if (v) return v
  } catch {
    /* ignore */
  }
  return fallback ?? PS_TOKEN_FALLBACKS[varName] ?? '#868e96'
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

/** Main profile node diameters (WIP §3.1), before compact scale.
 *
 *  graph-v3 tier 6-1 (2026-07-17) — explicit 4-tier hierarchy so the
 *  graph reads as "knowledge nodes with plumbing connectors" rather than
 *  "everyone is medium-sized":
 *
 *    Compound     TopicCluster 48   (must dominate; wraps others)
 *    Value        Insight 44, Topic 40   (the thinking + what it's about)
 *    Container    Episode 22 (aspect 1.35 → 30w × 22h card)
 *    Plumbing     Entity_person, Entity_organization, Speaker,
 *                 Podcast, Quote   → 12 flat dots
 *
 *  Plumbing dots preserve degreeHeat scaling (V5), so a hub Entity_person
 *  with heat=1 still reads at ~18px — small enough to recede, large
 *  enough that important connectors don't disappear. Prior sizes at 18-34
 *  for KG connectors made the whole canvas feel same-weight; this puts
 *  Insights + Topics on top of a low-visual-weight connectivity fabric.
 *
 *  Previous:
 *    Insight 44 · Topic 40 · TopicCluster 48 · Entity_person 34 ·
 *    Entity_organization 26 · Quote 22 · Speaker 18 · Episode 22 · Podcast 18 */
const NODE_DIAMETER_MAIN_PX: Record<string, number> = {
  Insight: 44,
  Topic: 40,
  TopicCluster: 48,
  Episode: 22,
  Entity_person: 12,
  Entity_organization: 12,
  Quote: 12,
  Speaker: 12,
  Podcast: 12,
}

/** graph-v3 H+I — width multiplier for rectangular shapes so they read as
 *  cards (wider than tall). Height stays at the base diameter. Unlisted
 *  types default to 1 (uniform width = height). Extended to
 *  Entity_organization so institutional nodes get the same card treatment
 *  as Episode. */
const NODE_ASPECT_W: Record<string, number> = {
  Episode: 1.35,
  Entity_organization: 1.35,
}

function scaledNodeSize(type: string, compact: boolean): number {
  const base = NODE_DIAMETER_MAIN_PX[type] ?? 18
  if (!compact) {
    return base
  }
  return Math.max(12, Math.round((base * 14) / 18))
}

function scaledNodeWidth(type: string, compact: boolean): number {
  const h = scaledNodeSize(type, compact)
  const wScale = NODE_ASPECT_W[type] ?? 1
  return Math.round(h * wScale)
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

/* graph-v3 B — read theme once per callback (cheap; dataset lookup, not
   getComputedStyle). Explicit ``data-theme`` wins; ``auto`` falls back to
   matchMedia so OS-preference users get the right opacity too. */
function isLightThemeActive(): boolean {
  try {
    const explicit = document.documentElement.dataset.theme
    if (explicit === 'light') return true
    if (explicit === 'dark') return false
    return window.matchMedia?.('(prefers-color-scheme: light)').matches ?? false
  } catch {
    return false
  }
}

function graphEdgeOpacity(ele: EdgeSingular): number {
  if (ele.hasClass('graph-edge-dimmed')) {
    return 0.2
  }
  if (ele.hasClass('graph-edge-neighbour')) {
    return 0.9
  }
  /* graph-v3 B — default edges recede so colour + type carry weight,
     neighbour/focus (0.9) still pop. Light theme uses a higher default
     because darker edge colours on a near-white canvas wash out at 0.3. */
  return isLightThemeActive() ? 0.5 : 0.3
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
  /**
   * RFC-080 V5 — when true, Topic + Episode `width`/`height` become
   * `mapData(degreeHeat, 0, 1, …)` so high-degree nodes are visually
   * larger. Off by default (RFC rollout: validate on staging corpus
   * before promoting). The existing `degreeHeat` border signal stays as
   * a secondary cue.
   */
  enableNodeSizeByDegree?: boolean
  /**
   * PRD-033 FR5.1 — main graph only. When true, nodes carrying the
   * ``context-relevant`` class (their episode is relevant to the active search
   * context) get an emphasis ring + size bump, so the canvas reflects retrieval
   * relevance, not only degree. Toggled at runtime via the ``context-relevant``
   * class (see ``GraphCanvas.applyContextEmphasis``).
   */
  includeContextEmphasis?: boolean
}): Record<string, unknown>[] {
  const compact = Boolean(options?.compact)
  const nw = scaledNodeSize('Episode', compact)
  const nh = scaledNodeSize('Episode', compact)
  const fs = compact ? '7px' : '9px'
  const tw = compact ? '72px' : '140px'
  const halo = cytoscapeNodeLabelHaloColorFromTheme()
  // Resolve theme tokens once. Cytoscape's stylesheet parser does not
  // resolve CSS `var(--…)` — using literals avoids the "invalid property"
  // warnings while still tracking the active theme.
  const psMuted = resolveThemeColor('--ps-muted')
  const psPrimary = resolveThemeColor('--ps-primary')
  const psWarning = resolveThemeColor('--ps-warning')
  const psGi = resolveThemeColor('--ps-gi')
  const psKg = resolveThemeColor('--ps-kg')
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
      // Theme-cluster membership (co-occurrence "Theme") — a teal ring on the topic
      // node, matching the player pill ring (--lp-theme #7dd3c0). A ring, not a
      // compound parent, so it coexists with the semantic cluster boxes. node:selected
      // is declared after, so the blue selection ring wins while a node is selected.
      selector: 'node.theme-member',
      style: {
        'border-width': compact ? 2 : 3,
        'border-color': '#7dd3c0',
        'border-opacity': 0.95,
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
        'target-arrow-color': psMuted,
        'line-color': psMuted,
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

  /* graph-v3 F + tier 6-1a — semantic shape per node type. Operator UX
     call 2026-07-17: **circles for big green knowledge, squares for
     smaller purple concepts** — Insight becomes ellipse (was diamond)
     and Topic becomes round-rectangle (was ellipse). The visual grammar
     now reads:
       Insight (green ellipse, big)     — the thinking (circles = knowledge)
       Topic   (purple round-rect, med) — what it's about (squares = concepts)
       Podcast → hexagon (container-of-episodes, distinct silhouette)
       Episode → round-rectangle (aspect 1.35 → card feel; distinct
                                  from Topic's square via aspect)
       Quote → round-tag (attributed snippet)
       Speaker → round-diamond (small speech-act marker)
       Entity_organization → round-rectangle (institution; small dot after tier 6-1)
     Person + Entity_person stay ellipse (humans-as-circles default).
     TopicCluster keeps its existing `roundrectangle` compound shape. */
  const shapeByType: Partial<Record<(typeof VISUAL_TYPES)[number], string>> = {
    Topic: 'round-rectangle',
    Insight: 'ellipse',
    Podcast: 'hexagon',
    Episode: 'round-rectangle',
    Quote: 'round-tag',
    Speaker: 'round-diamond',
    Entity_organization: 'round-rectangle',
  }

  const sizeByDegree = Boolean(options?.enableNodeSizeByDegree)
  for (const t of VISUAL_TYPES) {
    const h = scaledNodeSize(t, compact)
    const w = scaledNodeWidth(t, compact)
    const baseStyle: Record<string, unknown> = {
      'background-color': graphNodeFill(t),
      width: w,
      height: h,
    }
    const shape = shapeByType[t]
    if (shape) baseStyle.shape = shape
    style.push({
      selector: `node[type = "${t}"]`,
      style: baseStyle,
    })
    // RFC-080 V5 + graph-v3 J: scale hub types (Topic, Episode,
    // Entity_person, Entity_organization) by `degreeHeat`. GraphCanvas'
    // applyTopicDegreeHeat writes the field post-layout for the same
    // set. The specialization rule (`[type][degreeHeat]`) only fires
    // for elements carrying the field, so Quote / Speaker / Insight /
    // Podcast stay at their fixed base. Aspect ratio preserved by
    // scaling width and height independently against per-type base.
    //
    // graph-v3 tier 6-1b (2026-07-17) — plumbing types (Entity_person,
    // Entity_organization) use a tighter min so a low-heat plumbing node
    // never drops below ~10px. 12 base × 0.7 = 8.4px was borderline
    // invisible on DPR-2; 0.85 × 12 = 10.2px stays legible while still
    // shrinking the least-connected connectors relative to hubs.
    // Value tier (Topic + Episode) keeps the wider 0.7–1.5 range so
    // degree emphasis stays dramatic.
    if (
      sizeByDegree &&
      (t === 'Topic' ||
        t === 'Episode' ||
        t === 'Entity_person' ||
        t === 'Entity_organization')
    ) {
      const isPlumbing = t === 'Entity_person' || t === 'Entity_organization'
      const loScale = isPlumbing ? 0.85 : 0.7
      const wLo = Math.round(w * loScale)
      const wHi = Math.round(w * 1.5)
      const hLo = Math.round(h * loScale)
      const hHi = Math.round(h * 1.5)
      style.push({
        selector: `node[type = "${t}"][degreeHeat]`,
        style: {
          width: `mapData(degreeHeat, 0, 1, ${wLo}, ${wHi})`,
          height: `mapData(degreeHeat, 0, 1, ${hLo}, ${hHi})`,
        },
      })
    }
  }

  style.push({
    selector: 'node[type = "Insight"]',
    style: {
      'background-opacity': (ele: NodeSingular) => insightBackgroundOpacity(ele),
    },
  })

  // RFC-080 V2 — Insight grounding + confidence tier hooks. These rules
  // are always present in the stylesheet but only fire when the
  // matching class is assigned at element-build time
  // (`toCytoElements`). Tier classes are stylesheet hooks for future
  // selectors (e.g. "hide low-confidence" filters); they don't override
  // the existing `confidenceOpacity` mapping that drives
  // `background-opacity`. The ungrounded dashed border draws attention
  // to `grounded: false` insights without disturbing opacity. The
  // `.search-hit` selector pushed later in this stylesheet wins on
  // border styling when both apply (same-property override by source
  // order).
  style.push({
    selector: 'node[type = "Insight"].insight-ungrounded',
    style: {
      'border-width': compact ? 1.25 : 1.5,
      'border-style': 'dashed',
      'border-color': psWarning,
      'border-opacity': 0.7,
    },
  })

  style.push({
    selector: 'node[type = "Topic"]',
    style: {
      'border-width': (ele: NodeSingular) => topicBorderWidthForHeat(ele, compact),
      'border-color': psKg,
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
      'background-color': psKg,
      'background-opacity': 0.06,
      'border-width': compact ? 1.25 : 1.5,
      'border-style': 'dashed',
      'border-color': psKg,
      'border-opacity': 0.4,
      shape: 'roundrectangle',
      padding: tcPad,
    },
  })

  /* graph-v3 tier 8-1 — SuperTheme nodes.
     Big soft filled circles labelled by super_theme_label; the whole point
     of top-down mode is to make these bubbles the reader's first surface.
     Colour: teal (--ps-kg) at higher opacity than TopicCluster so they
     stand out as tier-0 anchors. Size: large (~ 3× a normal Topic). */
  style.push({
    selector: 'node[type = "SuperTheme"]',
    style: {
      'background-color': psKg,
      'background-opacity': 0.22,
      'border-width': compact ? 2 : 2.5,
      'border-style': 'solid',
      'border-color': psKg,
      'border-opacity': 0.85,
      shape: 'ellipse',
      width: compact ? 100 : 140,
      height: compact ? 100 : 140,
      label: 'data(label)',
      'text-valign': 'center',
      'text-halign': 'center',
      color: '#fff',
      'font-size': compact ? 12 : 14,
      'font-weight': 600,
      'text-outline-color': psKg,
      'text-outline-width': 2,
      'text-outline-opacity': 0.85,
      'text-wrap': 'wrap',
      'text-max-width': compact ? 84 : 116,
    },
  })
  /* Synthetic inter-super-theme links — dashed light lines just for
     layout structure, not analytic edges. */
  style.push({
    selector: 'edge[edgeType = "_topdown_link"], edge[type = "_topdown_link"]',
    style: {
      'line-color': psKg,
      'line-opacity': 0.2,
      'line-style': 'dashed',
      width: 1,
      'curve-style': 'straight',
      'target-arrow-shape': 'none',
    },
  })

  // Cytoscape 3.x core does not support `shadow-*` style properties (would
  // emit "shadow-blur: 8 is invalid" warnings on every load). The depth cue
  // they were intended to provide is already carried by border-color +
  // border-width on the same selectors above; the rules were never
  // visually applied, so they have been removed.

  style.push({
    selector: 'node.graph-label-tier-none',
    style: {
      'text-opacity': 0,
    },
  })

  /* graph-v3 tier 6-2 — zoom-gated Insight + Quote visibility. Applied
     by `syncGraphNodeVisibilityTierClasses` when zoom drops below
     GRAPH_NODE_ZOOM_INSIGHT_MIN. Hides the body + labels + disables
     interaction so hovering / clicking passes through to Topics /
     Persons underneath — the initial fit-all view stops being 400+
     green dots and reads as a topic-and-episode map. Above the
     threshold the class is removed and Insights / Quotes fade back in
     (opacity transition already declared on the base `node` rule). */
  style.push({
    selector: 'node.graph-node-zoom-hidden',
    style: {
      opacity: 0,
      'text-opacity': 0,
      events: 'no',
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

  /* graph-v3 G — widen the semantic-tier width spread so "same-thickness"
     lines stop dominating the read. Structural chunky (2.75px), descriptive
     confidence-mapped (0.75×–1.5× base), evidence + discovery thinner
     (0.75px) so they recede into the canvas hum. Compact profile scales
     proportionally. Aggregates keep their existing mapData(weight) so the
     chunkiest render at 5px. */
  const edgeStrokes: { selector: string; style: Record<string, unknown> }[] = [
    {
      selector: 'edge[edgeType = "HAS_INSIGHT"]',
      style: {
        width: compact ? 2 : 2.75,
        'line-color': psPrimary,
        'line-style': 'solid',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': psPrimary,
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
        width: aboutConfidenceWidth(compact ? 1.5 : 2),
        'line-color': psGi,
        'line-style': 'solid',
        'line-opacity': aboutConfidenceOpacity(),
        'target-arrow-shape': 'none',
      },
    },
    {
      selector: 'edge[edgeType = "SUPPORTED_BY"]',
      style: {
        width: compact ? 0.7 : 0.85,
        'line-color': psMuted,
        'line-style': 'dashed',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': psMuted,
      },
    },
    {
      selector: 'edge[edgeType = "RELATED_TO"]',
      style: {
        width: compact ? 0.7 : 0.85,
        'line-color': psKg,
        'line-style': 'solid',
        'target-arrow-shape': 'none',
      },
    },
    {
      selector: 'edge[edgeType = "MENTIONS"]',
      style: {
        width: compact ? 0.7 : 0.85,
        'line-color': psMuted,
        'line-style': 'dotted',
        'target-arrow-shape': 'none',
      },
    },
    {
      selector: 'edge[edgeType = "SPOKE_IN"]',
      style: {
        width: compact ? 2 : 2.75,
        'line-color': psPrimary,
        'line-style': 'solid',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': psPrimary,
      },
    },
    {
      selector: 'edge[edgeType = "HAS_MEMBER"]',
      style: {
        width: compact ? 1.5 : 2,
        'line-color': psKg,
        'line-style': 'solid',
        'target-arrow-shape': 'none',
        'line-opacity': 0.6,
      },
    },
    // RFC-097 v2 two-tier edge contract — visual classes (chunk 8):
    //   evidentiary  → SUPPORTED_BY        (dashed, muted; the grounding contract)
    //   descriptive  → ABOUT / MENTIONS_*  (solid gi-color; confidence-mapped opacity)
    //   discovery    → MENTIONS            (dotted, muted; co-occurrence only)
    //   structural   → HAS_INSIGHT / SPOKE_IN / HAS_EPISODE / HAS_MEMBER  (solid primary, arrow)
    //   attribution  → SPOKEN_BY           (solid warning-tone accent, arrow)
    // ABOUT, SUPPORTED_BY, MENTIONS, HAS_INSIGHT, SPOKE_IN, HAS_MEMBER predate v2;
    // the new selectors below add MENTIONS_PERSON, MENTIONS_ORG, HAS_EPISODE, SPOKEN_BY.
    {
      // descriptive: same visual class as ABOUT (gi color, confidence opacity).
      // Insight → Person edge, so add an arrow to make the direction explicit.
      selector: 'edge[edgeType = "MENTIONS_PERSON"]',
      style: {
        width: aboutConfidenceWidth(compact ? 1.5 : 2),
        'line-color': psGi,
        'line-style': 'solid',
        'line-opacity': aboutConfidenceOpacity(),
        'target-arrow-shape': 'triangle',
        'target-arrow-color': psGi,
      },
    },
    {
      // descriptive: same visual class as ABOUT / MENTIONS_PERSON.
      selector: 'edge[edgeType = "MENTIONS_ORG"]',
      style: {
        width: aboutConfidenceWidth(compact ? 1.5 : 2),
        'line-color': psGi,
        'line-style': 'solid',
        'line-opacity': aboutConfidenceOpacity(),
        'target-arrow-shape': 'triangle',
        'target-arrow-color': psGi,
      },
    },
    {
      // structural: Podcast → Episode, same visual class as HAS_INSIGHT.
      selector: 'edge[edgeType = "HAS_EPISODE"]',
      style: {
        width: compact ? 2 : 2.75,
        'line-color': psPrimary,
        'line-style': 'solid',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': psPrimary,
      },
    },
    {
      // attribution: Quote → Person. Distinctive warning-tone accent to set apart
      // from structural primary-tone edges; carries an arrow because direction
      // matters (the Quote is spoken BY the Person).
      selector: 'edge[edgeType = "SPOKEN_BY"]',
      style: {
        width: compact ? 1 : 1.25,
        'line-color': psWarning,
        'line-style': 'solid',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': psWarning,
        'line-opacity': 0.85,
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
      'line-color': psMuted,
      'line-style': 'solid',
      'target-arrow-shape': 'triangle',
      'target-arrow-color': psMuted,
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

  // RFC-080 V1 — render-only aggregated edges (Episode→Topic ABOUT_AGG,
  // Episode→Person SPOKE_IN_AGG). These selectors are always present in
  // the stylesheet but only fire when `toCytoElements` was called with
  // `enableAggregatedEdges: true` (so production builds without the
  // lens enabled get no behaviour change). Width maps `data(weight)` —
  // a count of contributing per-Insight edges — into a 1.5px..5px
  // range. Mapping is per-element, not normalised against the whole
  // graph; tightening to a slice-relative max is in #667's open
  // questions.
  const aboutAggBaseWidth = compact ? 1 : 1.5
  const aboutAggMaxWidth = compact ? 4 : 5
  style.push({
    selector: 'edge.graph-edge-about-agg',
    style: {
      width: `mapData(weight, 1, 12, ${aboutAggBaseWidth}, ${aboutAggMaxWidth})`,
      'line-color': psGi,
      'line-style': 'solid',
      'line-opacity': 0.85,
      'target-arrow-shape': 'none',
      // Slightly higher z so the chunky aggregate sits over the
      // per-Insight ABOUT fan when both are visible (e.g. tier-full
      // before #667 ships the tier gating).
      'z-index': 5,
    },
  })
  style.push({
    selector: 'edge.graph-edge-spoke-in-agg',
    style: {
      width: `mapData(weight, 1, 8, ${aboutAggBaseWidth}, ${aboutAggMaxWidth})`,
      'line-color': psPrimary,
      'line-style': 'solid',
      'line-opacity': 0.85,
      'target-arrow-shape': 'triangle',
      'target-arrow-color': psPrimary,
      'z-index': 5,
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

  /* graph-v3 K — bridge nodes (high betweenness centrality). Distinct
     dashed rose border so bridging entities stand out visually.
     Ordered BEFORE search-hit / selection / expand rules so those
     interaction-state borders still win when both apply. */
  const bridgeRing = compact ? 1.5 : 2
  style.push({
    selector: 'node.graph-bridge',
    style: {
      'border-width': bridgeRing,
      'border-style': 'dashed',
      'border-color': '#f472b6',
      'border-opacity': 0.85,
    },
  })

  /* graph-v3 U — theme-cluster region underlay tint. Cytoscape
     `underlay-*` renders a translucent disc BEHIND the node body, so
     type colour + shape + size stay 100% legible on top. High padding
     + low opacity makes adjacent same-region nodes' underlays overlap
     into blob-like regions (InfraNodus feel).
     Classes `theme-region-0..7` are assigned by GraphCanvas'
     applyThemeRegionClasses via a stable hash of the cluster's
     `graph_compound_parent_id`, so the same thc:… id always paints the
     same colour across sessions. Palette: 8 pastel hues evenly spaced
     around HSL, saturation ~45%, lightness ~65% — at 0.14 opacity
     they read as soft coloured mist against the darker canvas.
     Selectors ordered before interaction-state rules so search-hit /
     selection borders still win visually. */
  // graph-v3 tier 5A harden fix #2: palette imported from the shared util
  // so legend + graph render identical hex for the same `thc:...` id.
  // Drifting them independently would silently mismatch — see the comment
  // in themeRegionPalette.ts.
  const underlayOpacity = compact ? 0.1 : 0.14
  const underlayPadding = compact ? 8 : 14
  THEME_REGION_PALETTE.forEach((hex, i) => {
    style.push({
      selector: `node.theme-region-${i}`,
      style: {
        'underlay-color': hex,
        'underlay-opacity': underlayOpacity,
        'underlay-padding': underlayPadding,
        'underlay-shape': 'ellipse',
      },
    })
  })
  /* graph-v3 tier 7-4 — Person community regions. Same palette + shape
     as theme regions but slightly higher opacity (Person nodes are
     smaller than Topic hubs, so the underlay needs to read a beat more
     strongly to be visible as a group tint). Classes person-region-0..7
     assigned by applyPersonCommunityRegions from
     guest_coappearance.communities[]. */
  const personUnderlayOpacity = compact ? 0.14 : 0.18
  const personUnderlayPadding = compact ? 6 : 10
  THEME_REGION_PALETTE.forEach((hex, i) => {
    style.push({
      selector: `node.person-region-${i}`,
      style: {
        'underlay-color': hex,
        'underlay-opacity': personUnderlayOpacity,
        'underlay-padding': personUnderlayPadding,
        'underlay-shape': 'ellipse',
      },
    })
  })

  /* graph-v3 Tier 5C-1 — velocity halo. Bright coloured border on
     Topic + Person nodes when the temporal_velocity envelope classes
     them rising / cooling / steady. Uses the same palette as the
     Digest / Trending Topics trend arrows (`utils/trend.ts`) so the
     app tells one story about "is this topic hot right now".
     Border-width bump above the type default so the halo reads at
     mid-zoom without needing a label. */
  const velocityBorder = compact ? 1.5 : 2.25
  const velocityColours = { up: '#22c55e', down: '#ef4444', steady: '#f59e0b' }
  ;(['up', 'down', 'steady'] as const).forEach((dir) => {
    style.push({
      selector: `node.velocity-${dir}`,
      style: {
        'border-width': velocityBorder,
        'border-style': 'solid',
        'border-color': velocityColours[dir],
        'border-opacity': dir === 'steady' ? 0.7 : 0.9,
      },
    })
  })

  /* graph-v3 Tier 5C-2 — Person credibility border. Reads the
     grounding_rate envelope (Insights supported by a Person that are
     grounded / total). High = solid green, medium = solid amber, low
     = dashed red. Direct analog of `.insight-ungrounded` for Persons. */
  const credibilityBorder = compact ? 1.5 : 2
  style.push({
    selector: 'node.credibility-high',
    style: {
      'border-width': credibilityBorder,
      'border-style': 'solid',
      'border-color': '#22c55e',
      'border-opacity': 0.85,
    },
  })
  style.push({
    selector: 'node.credibility-med',
    style: {
      'border-width': credibilityBorder,
      'border-style': 'solid',
      'border-color': '#f59e0b',
      'border-opacity': 0.8,
    },
  })
  style.push({
    selector: 'node.credibility-low',
    style: {
      'border-width': credibilityBorder,
      'border-style': 'dashed',
      'border-color': '#ef4444',
      'border-opacity': 0.8,
    },
  })

  /* graph-v3 Tier 5D-1 — consensus edges (topic_consensus). Thin
     bright-green arcs between two Persons who corroborate on a Topic.
     No arrow (symmetric). Distinct hue from HAS_INSIGHT primary blue
     so the overlay doesn't blend into structural edges. */
  style.push({
    selector: 'edge.lens-consensus-edge',
    style: {
      width: compact ? 1 : 1.5,
      'line-color': '#22c55e',
      'line-style': 'solid',
      'line-opacity': 0.65,
      'target-arrow-shape': 'none',
      'curve-style': 'unbundled-bezier',
      'z-index': 4,
    },
  })

  /* graph-v3 Tier 5D-2 — co-guest edges (guest_coappearance). Dotted
     amber arcs between Persons who share ≥2 episodes. Weight-scaled
     width so a pair with 5 shared episodes reads chunkier than a
     pair with 2. Muted opacity so the base graph still dominates. */
  style.push({
    selector: 'edge.lens-coguest-edge',
    style: {
      width: `mapData(weight, 2, 10, ${compact ? 0.7 : 1}, ${compact ? 2 : 3})`,
      'line-color': '#f59e0b',
      'line-style': 'dotted',
      'line-opacity': 0.55,
      'target-arrow-shape': 'none',
      'curve-style': 'unbundled-bezier',
      'z-index': 3,
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

  if (options?.includeContextEmphasis) {
    // PRD-033 FR5.1 — class selector (specificity > the node[type=…] size rules)
    // so relevant nodes are visibly larger + ringed while a search context is active.
    style.push({
      selector: 'node.context-relevant',
      style: {
        width: 50,
        height: 50,
        'border-width': 3,
        'border-color': psPrimary,
        'border-opacity': 1,
        'z-index': 11,
      },
    })
  }

  return style
}
