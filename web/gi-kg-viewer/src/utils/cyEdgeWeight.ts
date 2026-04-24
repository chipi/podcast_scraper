/**
 * Pre-#656 foundation: reusable data-driven edge-style helpers for
 * Cytoscape ABOUT / RELATED_TO / SUPPORTED_BY edges.
 *
 * Background
 *   Edge selectors in ``cyGraphStylesheet.ts`` are static — every ABOUT
 *   edge renders with ``width: 2, line-opacity: 1``. #664 shipped
 *   ``properties.confidence`` on ABOUT edges (cosine similarity, 0.25–1.0)
 *   but there was no standard way to consume it in the graph.
 *
 *   #656 Stage C (confidence-weighted edges) needs that mapping. Rather
 *   than inline the ``data(...)`` callback at the feature site, expose a
 *   single helper used for every weighted edge type we'll add over time.
 *
 * Shape
 *   ``weightedEdgeOpacity`` and ``weightedEdgeWidth`` are callback
 *   factories matching Cytoscape's selector-style API (``function(ele)``).
 *   They read ``ele.data('properties')?.<key>`` — matching the edge
 *   schema where ABOUT's confidence lives under
 *   ``properties.confidence`` (see ``gi/pipeline.py`` edge-build).
 *
 *   Back-compat: when the property is absent / malformed, the callback
 *   returns ``fallback`` — so legacy ``gi.json`` files from corpora built
 *   before #664 render with today's uniform weight (no regression).
 */
import type { EdgeSingular } from 'cytoscape'

export interface EdgeWeightRange {
  /** Minimum observed or expected value for the property (clamped below). */
  min: number
  /** Maximum observed or expected value (clamped above). */
  max: number
}

export interface WeightedStyleOptions {
  /**
   * Dotted path inside ``ele.data()``. Examples: ``properties.confidence``,
   * ``properties.weight``. Dots are resolved left-to-right.
   */
  propertyPath: string
  /** Domain of the raw property value. */
  domain: EdgeWeightRange
  /** Range to map into (opacity 0..1 / width in px). */
  range: EdgeWeightRange
  /** Value returned when the property is missing / non-finite. */
  fallback: number
}

function readPath(obj: unknown, path: string): unknown {
  if (obj == null) return undefined
  const parts = path.split('.')
  let current: unknown = obj
  for (const part of parts) {
    if (current == null || typeof current !== 'object') return undefined
    current = (current as Record<string, unknown>)[part]
  }
  return current
}

function coerceFinite(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) return v
  if (typeof v === 'string' && v.trim() !== '') {
    const n = Number(v)
    if (Number.isFinite(n)) return n
  }
  return null
}

/**
 * Linear-interpolate a raw value in ``domain`` into ``range``, clamping
 * inputs that fall outside the domain so a stray cosine of -0.5 does not
 * produce a negative opacity.
 */
export function scaleWeighted(
  raw: number,
  domain: EdgeWeightRange,
  range: EdgeWeightRange,
): number {
  if (domain.max <= domain.min) return range.min
  const clamped = Math.min(domain.max, Math.max(domain.min, raw))
  const t = (clamped - domain.min) / (domain.max - domain.min)
  return range.min + t * (range.max - range.min)
}

/**
 * Build a Cytoscape ``function(ele)`` callback that reads
 * ``ele.data(propertyPath)`` and maps it to a numeric style output.
 *
 * Suitable for ``line-opacity``, ``width``, etc. Missing / unparseable
 * properties fall back to ``fallback`` so legacy data continues to
 * render uniformly.
 */
export function weightedEdgeStyle(
  options: WeightedStyleOptions,
): (ele: EdgeSingular) => number {
  const { propertyPath, domain, range, fallback } = options
  return (ele) => {
    const raw = coerceFinite(readPath(ele.data(), propertyPath))
    if (raw == null) return fallback
    return scaleWeighted(raw, domain, range)
  }
}

/**
 * Convenience: opacity helper for confidence-style [0, 1] properties.
 * Maps ``properties.<key>`` cosine (clamped to [0.25, 1.0] — the
 * #664 floor) to opacity in [0.35, 1.0]. Weak edges remain visible
 * enough to read but clearly de-emphasized.
 */
export function weightedEdgeOpacity(
  propertyPath: string = 'properties.confidence',
  fallback: number = 1,
): (ele: EdgeSingular) => number {
  return weightedEdgeStyle({
    propertyPath,
    domain: { min: 0.25, max: 1.0 },
    range: { min: 0.35, max: 1.0 },
    fallback,
  })
}

/**
 * Convenience: width helper. Scales ``properties.<key>`` in [0.25, 1.0]
 * to [baseWidth * 0.75, baseWidth * 1.5] — so the heaviest edges read
 * ~2× the lightest without overwhelming node badges.
 */
export function weightedEdgeWidth(
  baseWidth: number,
  propertyPath: string = 'properties.confidence',
): (ele: EdgeSingular) => number {
  return weightedEdgeStyle({
    propertyPath,
    domain: { min: 0.25, max: 1.0 },
    range: { min: baseWidth * 0.75, max: baseWidth * 1.5 },
    fallback: baseWidth,
  })
}
