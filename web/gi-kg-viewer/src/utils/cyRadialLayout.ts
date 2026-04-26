/**
 * RFC-080 V4 — radial focus mode position math.
 *
 * Pure functions only — no Cytoscape import — so the math is unit-
 * testable without a browser. The Cytoscape wiring (enter / exit / state
 * snapshot / event listeners) lives in GraphCanvas and consumes these
 * helpers.
 *
 * Geometry:
 *   - Centre node sits at (0, 0).
 *   - Ring 1 nodes (1-hop neighbours of the centre) are placed on a
 *     circle of radius `r1`.
 *   - Ring 2 nodes (2-hop) sit on a circle of radius `r2 = 2 * r1`.
 *   - Ring radii adapt to the visual radius of the largest ring-1 node
 *     so V5 (node-size-by-degree) doesn't make adjacent nodes overlap.
 *   - Anything beyond ring 2 is excluded from the position set; the
 *     caller hides those nodes.
 *   - Angular spacing within a ring is uniform: the i-th of N nodes
 *     gets `angle = 2π * i / N` (deterministic; stable across re-runs).
 */

export interface RadialPosition {
  x: number
  y: number
}

export interface RadialPositions {
  /** Centre node position — always (0, 0). */
  centre: RadialPosition
  /** Ring radii actually used (may be larger than `baseR1` if adapted). */
  r1: number
  r2: number
  /** Map of node id → computed (x, y) for centre + ring 1 + ring 2. */
  positions: Record<string, RadialPosition>
}

export interface RadialLayoutOptions {
  /**
   * Floor for the inner ring radius. The actual `r1` is the maximum
   * of this value and `maxRing1NodeRadius * 2.5` so big nodes never
   * overlap their neighbours on the ring.
   */
  baseR1?: number
  /**
   * Outer ring radius factor. `r2 = r1 * outerRingFactor`. Default
   * 2 keeps the documented two-ring proportion.
   */
  outerRingFactor?: number
  /**
   * Largest visual radius among ring-1 nodes (in px). Used by
   * `radialRingRadii` to grow `r1` so adjacent ring-1 nodes don't
   * overlap. Pass 0 (default) when nodes are uniformly small / V5 is
   * off.
   */
  maxRing1NodeRadius?: number
}

const DEFAULT_BASE_R1 = 120
const DEFAULT_OUTER_FACTOR = 2

/**
 * Compute adaptive ring radii. The inner radius scales with the largest
 * ring-1 node radius so V5-scaled high-degree Topics (up to 60px) don't
 * overlap their ring neighbours. The outer is a fixed multiple so the
 * two-ring layout reads consistently.
 */
export function radialRingRadii(opts?: RadialLayoutOptions): { r1: number; r2: number } {
  const baseR1 = opts?.baseR1 ?? DEFAULT_BASE_R1
  const factor = opts?.outerRingFactor ?? DEFAULT_OUTER_FACTOR
  const maxRadius = Math.max(0, opts?.maxRing1NodeRadius ?? 0)
  // Empirically: 2.5x the node radius gives ~30% gap between adjacent
  // nodes on a ring with up to ~12 elements; matches RFC-080 V4 spec.
  const r1 = Math.max(baseR1, maxRadius * 2.5)
  return { r1, r2: r1 * factor }
}

/**
 * Compute (x, y) positions for a radial layout centred on `centreId`,
 * with the supplied 1-hop and 2-hop node id sets. Returns the full
 * position map plus the ring radii actually used (so the caller can
 * size the viewport / fit padding appropriately).
 *
 * Node ids that appear in BOTH `ring1Ids` and `ring2Ids` are placed in
 * ring 1 only — callers should pre-filter to the canonical hop count.
 * The centre id, if also present in either ring, stays at the centre.
 */
export function computeRadialPositions(
  centreId: string,
  ring1Ids: string[],
  ring2Ids: string[],
  opts?: RadialLayoutOptions,
): RadialPositions {
  const { r1, r2 } = radialRingRadii(opts)
  const positions: Record<string, RadialPosition> = {}
  positions[centreId] = { x: 0, y: 0 }

  // De-dup: if a node sits in ring1, skip it from ring2.
  const ring1 = ring1Ids.filter((id) => id !== centreId)
  const ring1Set = new Set(ring1)
  const ring2 = ring2Ids.filter((id) => id !== centreId && !ring1Set.has(id))

  for (let i = 0; i < ring1.length; i += 1) {
    const angle = (2 * Math.PI * i) / Math.max(1, ring1.length)
    positions[ring1[i]!] = { x: r1 * Math.cos(angle), y: r1 * Math.sin(angle) }
  }
  for (let i = 0; i < ring2.length; i += 1) {
    const angle = (2 * Math.PI * i) / Math.max(1, ring2.length)
    positions[ring2[i]!] = { x: r2 * Math.cos(angle), y: r2 * Math.sin(angle) }
  }
  return { centre: { x: 0, y: 0 }, r1, r2, positions }
}

/**
 * Snapshot describes everything the caller needs to restore the graph
 * to its pre-radial state. Saved by enterRadialMode and consumed by
 * exitRadialMode in GraphCanvas. Held in pinia so a second enter
 * doesn't accidentally double-snapshot a partially-restored state.
 */
export interface RadialSnapshot {
  /** Per-node `{x, y}` captured before any positions were rewritten. */
  positions: Record<string, RadialPosition>
  /** Per-node `display` style — typically `'element'` or `'none'`. */
  displays: Record<string, string>
  /** Per-edge `display` style. */
  edgeDisplays: Record<string, string>
  /** The centre node id at enter time (used to drive a11y announcement on exit). */
  centreId: string
}
