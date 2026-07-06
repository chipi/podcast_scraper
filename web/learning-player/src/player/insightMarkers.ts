/**
 * Insight density markers for the player scrubber (#1140 — the "skip guide").
 *
 * One tick per insight that has a timestamped supporting quote, positioned by its
 * EARLIEST quote. The *distribution* is the point: clusters of ticks show where the
 * substance is, so a listener can skip the fluff. `grounded` drives colour; the
 * insight's `confidence` drives the tick's visual weight (opacity) so a dense run
 * of high-confidence insights reads as "heavy" at a glance.
 *
 * Pure + presentational-agnostic: returns positions + weights; the component paints.
 */
import type { Insight } from '../services/types'

export interface InsightMarker {
  id: string
  /** Seconds into the episode (for click-to-seek and ordering). */
  timeSec: number
  /** 0–100 position along the scrubber. */
  pct: number
  grounded: boolean
  /** 0..1 visual weight from `confidence` (fallback 0.6 when unknown). */
  weight: number
}

/** Earliest supporting-quote start (ms) for an insight — its place on the timeline. */
function insightStartMs(ins: Insight): number | null {
  let best: number | null = null
  for (const q of ins.quotes ?? []) {
    if (typeof q.start_ms === 'number' && Number.isFinite(q.start_ms) && q.start_ms >= 0) {
      best = best === null ? q.start_ms : Math.min(best, q.start_ms)
    }
  }
  return best
}

function clampWeight(confidence: number | null): number {
  if (typeof confidence !== 'number' || !Number.isFinite(confidence)) return 0.6
  return Math.max(0.2, Math.min(1, confidence))
}

/**
 * Markers for the scrubber, sorted by position. Only insights with a timestamped
 * quote are placed. `durationSec <= 0` (unknown length) → no markers.
 */
export function insightScrubberMarkers(
  insights: Insight[] | null | undefined,
  durationSec: number,
): InsightMarker[] {
  if (!(durationSec > 0) || !Array.isArray(insights)) return []
  const out: InsightMarker[] = []
  for (const ins of insights) {
    const ms = insightStartMs(ins)
    if (ms === null) continue
    const timeSec = ms / 1000
    const pct = Math.max(0, Math.min(100, (timeSec / durationSec) * 100))
    out.push({
      id: ins.id,
      timeSec,
      pct,
      grounded: Boolean(ins.grounded),
      weight: clampWeight(ins.confidence),
    })
  }
  return out.sort((a, b) => a.pct - b.pct)
}
