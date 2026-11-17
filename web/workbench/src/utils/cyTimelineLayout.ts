/**
 * RFC-080 V3 — timeline layout position math.
 *
 * Pure functions only — no Cytoscape import — so the math is unit-
 * testable without a browser. The Cytoscape integration (preset apply,
 * `publishDate` wiring on Episode node data, layout-cycle entry, missing-
 * date marker) lives in GraphCanvas and consumes these helpers.
 *
 * Geometry:
 *   - Episodes form the spine on the horizontal axis.
 *   - x is computed via *quantile mapping* by default — sort dates and
 *     give each episode `x = canvasWidth * i / (N - 1)`. Robust to
 *     podcast back-catalog shape (one recent + many old episodes
 *     wouldn't compress all the old ones into the leftmost pixel).
 *   - x can opt into *linear date mapping* —
 *     `x = canvasWidth * (date - min) / (max - min)`. User-driven
 *     sub-toggle on the bottom bar.
 *   - Topic / Person nodes get x = weighted mean of their connected
 *     episodes' x positions. Bands separate them on y so the spine
 *     stays visually distinct.
 *   - Insights / Quotes ride near their parent Episode (small offset).
 *   - Missing-date episodes park at a single off-screen-leftish spot
 *     with a marker (no silent "rightmost = recent" lie).
 *   - Y-jitter is deterministic (hash of node id) so the layout is
 *     stable across re-runs.
 */

export interface TimelinePosition {
  x: number
  y: number
}

export interface TimelineEpisodeInput {
  id: string
  /** Publish date in **milliseconds since epoch**. `null` parks the episode at the missing-date spot. */
  dateMs: number | null
}

export interface TimelineCanvasGeometry {
  /** Pixel width available for the horizontal date axis. */
  canvasWidth: number
  /** Vertical centre — the episode spine sits on this y line ± jitter. */
  canvasMidY: number
  /** Topic band sits at `canvasMidY + topicBandOffset` (default 100). */
  topicBandOffset?: number
  /** Person band sits at `canvasMidY + personBandOffset` (default -180). */
  personBandOffset?: number
  /** Y-jitter range applied per node (±). */
  jitterRange?: number
  /** Where to park episodes with no date (default `-60` from the leftmost x). */
  missingDateParkOffset?: number
}

export type TimelineAxis = 'quantile' | 'linear'

/**
 * Deterministic 32-bit hash → small float in [-1, 1]. Used for stable
 * y-jitter so the layout is reproducible across re-runs (the user's
 * mental map of "this Topic is here" survives a re-layout).
 */
export function deterministicJitter(id: string, range: number): number {
  let h = 5381
  for (let i = 0; i < id.length; i += 1) {
    // djb2-style hash — fast, no library, deterministic.
    h = (h * 33) ^ id.charCodeAt(i)
    h |= 0 // force int32
  }
  // Normalise into [-1, 1]; map onto ±range.
  const norm = ((h | 0) / 0x7fffffff)
  return Math.max(-range, Math.min(range, norm * range))
}

/**
 * Compute episode x positions.
 *
 * Quantile (default): sort by date; the i-th of N sorted episodes gets
 *   `x = canvasWidth * i / (N - 1)`. Episodes with the same date
 *   alphabetise on id for stable ordering.
 *
 * Linear: `x = canvasWidth * (date - min) / (max - min)` — preserves
 *   metric date intervals at the cost of compressing tight clusters
 *   when one outlier shifts max.
 *
 * Episodes with `dateMs == null` are parked at the leftmost-minus-
 * `missingDateParkOffset` spot (default -60px); callers can render a
 * marker noting the count.
 */
export function computeEpisodeTimelinePositions(
  episodes: TimelineEpisodeInput[],
  geometry: TimelineCanvasGeometry,
  axis: TimelineAxis = 'quantile',
): {
  /** Per-episode (x, y). Includes missing-date episodes at the park spot. */
  positions: Record<string, TimelinePosition>
  /** Ids of episodes with no date — caller surfaces a count to the user. */
  missingDateIds: string[]
} {
  const jitterRange = geometry.jitterRange ?? 40
  const parkOffset = geometry.missingDateParkOffset ?? -60
  const dated = episodes.filter((e) => typeof e.dateMs === 'number' && Number.isFinite(e.dateMs))
  const undated = episodes.filter((e) => !(typeof e.dateMs === 'number' && Number.isFinite(e.dateMs)))
  const positions: Record<string, TimelinePosition> = {}

  if (dated.length === 0) {
    // Nothing dated → park everything at the missing-date spot.
    for (const e of episodes) {
      positions[e.id] = {
        x: parkOffset,
        y: geometry.canvasMidY + deterministicJitter(e.id, jitterRange),
      }
    }
    return { positions, missingDateIds: episodes.map((e) => e.id) }
  }

  if (axis === 'quantile') {
    const sorted = [...dated].sort((a, b) => {
      const da = a.dateMs as number
      const db = b.dateMs as number
      if (da !== db) return da - db
      return a.id < b.id ? -1 : a.id > b.id ? 1 : 0
    })
    const denom = Math.max(1, sorted.length - 1)
    for (let i = 0; i < sorted.length; i += 1) {
      const e = sorted[i]!
      positions[e.id] = {
        x: (geometry.canvasWidth * i) / denom,
        y: geometry.canvasMidY + deterministicJitter(e.id, jitterRange),
      }
    }
  } else {
    let minDate = Infinity
    let maxDate = -Infinity
    for (const e of dated) {
      const d = e.dateMs as number
      if (d < minDate) minDate = d
      if (d > maxDate) maxDate = d
    }
    const span = Math.max(1, maxDate - minDate)
    for (const e of dated) {
      const t = ((e.dateMs as number) - minDate) / span
      positions[e.id] = {
        x: geometry.canvasWidth * t,
        y: geometry.canvasMidY + deterministicJitter(e.id, jitterRange),
      }
    }
  }
  for (const e of undated) {
    positions[e.id] = {
      x: parkOffset,
      y: geometry.canvasMidY + deterministicJitter(e.id, jitterRange),
    }
  }
  return { positions, missingDateIds: undated.map((e) => e.id) }
}

/**
 * Compute x position for a Topic/Person/Entity node from its connected
 * episodes' positions: the weighted mean. Returns `null` when the node
 * has no connected episodes (caller decides where to park orphans).
 */
export function weightedMeanXFromEpisodes(
  connectedEpisodeXs: number[],
): number | null {
  if (connectedEpisodeXs.length === 0) return null
  let sum = 0
  for (const x of connectedEpisodeXs) sum += x
  return sum / connectedEpisodeXs.length
}
