/**
 * Pure helpers for insight ↔ playback alignment (RFC-099 §4/§5). Kept framework-free so the
 * "surfacing now" math and timestamp extraction are unit-tested in isolation.
 */

import type { Insight, Quote, SearchHit, Segment } from '../services/types'

/** Where in the transcript an insight's quote lands, for highlighting + tap-to-reveal. */
export interface GroundedSpan {
  insightId: string
  insightText: string
  insightType: string | null
  /** Verbatim supporting quote (for char-level highlight of the exact phrase, 3.6). */
  quote: string
}

/**
 * Char-level highlight range of a grounded quote inside one segment's text (RFC-102 / PRD-043 FR5).
 * Text-matched (NOT char offsets — those drift across transcript versions). Returns the split
 * `{pre, match, post}` when the quote (or this whole segment, for a multi-segment quote) can be
 * located; `null` when it can't — the caller then underlines the whole segment (safe fallback).
 */
export function quoteHighlight(
  segmentText: string,
  quote: string,
): { pre: string; match: string; post: string } | null {
  const s = segmentText
  const q = quote.trim()
  if (!q || !s) return null
  // Quote sits inside this segment → highlight just the matched phrase (short single-segment quote).
  const idx = s.toLowerCase().indexOf(q.toLowerCase())
  if (idx !== -1) {
    return { pre: s.slice(0, idx), match: s.slice(idx, idx + q.length), post: s.slice(idx + q.length) }
  }
  // This whole segment sits inside the quote → it's all part of the quote (middle of a long quote).
  if (s.trim() && q.toLowerCase().includes(s.trim().toLowerCase())) {
    return { pre: '', match: s, post: '' }
  }
  return null
}

/**
 * Map transcript segment indices → the grounded insight whose supporting quote overlaps them
 * (by timeline, robust to transcript-version char-offset drift). The earliest insight wins a
 * shared segment. Lets the transcript highlight quoted passages and tap through to the claim.
 */
export function groundedSpansBySegment(
  segments: Segment[],
  insights: Insight[],
): Record<number, GroundedSpan> {
  const out: Record<number, GroundedSpan> = {}
  for (const ins of insights) {
    for (const q of ins.quotes) {
      if (q.start_ms == null) continue
      const qStart = q.start_ms / 1000
      const qEnd = (q.end_ms ?? q.start_ms + 8000) / 1000
      for (let i = 0; i < segments.length; i++) {
        const s = segments[i]
        if (s.start < qEnd && s.end > qStart && !(i in out)) {
          out[i] = {
            insightId: ins.id,
            insightText: ins.text,
            insightType: ins.insight_type,
            quote: q.text,
          }
        }
      }
    }
  }
  return out
}

/** Earliest supporting-quote start (seconds) for an insight, or null when untimed. */
export function insightStartSeconds(insight: Insight): number | null {
  let best: number | null = null
  for (const q of insight.quotes) {
    if (q.start_ms != null && (best == null || q.start_ms < best)) best = q.start_ms
  }
  return best == null ? null : best / 1000
}

function quoteContains(q: Quote, tMs: number): boolean {
  if (q.start_ms == null) return false
  const end = q.end_ms ?? q.start_ms + 8000 // assume ~8s when no end marker
  return tMs >= q.start_ms && tMs <= end
}

/**
 * Index of the insight being "spoken" at playback time `t` (seconds) — the one whose
 * supporting quote window contains `t`. Returns -1 when none is active. Picks the latest
 * starting match if several overlap.
 */
export function activeInsightIndex(insights: Insight[], t: number): number {
  const tMs = t * 1000
  let ans = -1
  let bestStart = -1
  insights.forEach((ins, i) => {
    for (const q of ins.quotes) {
      if (quoteContains(q, tMs) && (q.start_ms ?? -1) >= bestStart) {
        bestStart = q.start_ms ?? -1
        ans = i
      }
    }
  })
  return ans
}

/** Best jump-to-moment time (seconds) for a search hit, or null when none is derivable. */
export function hitStartSeconds(hit: SearchHit): number | null {
  const fromMs = (v: unknown): number | null =>
    typeof v === 'number' && Number.isFinite(v) ? v / 1000 : null

  // Transcript hit lift: lifted.quote.timestamp_start_ms.
  const lifted = hit.lifted as { quote?: Record<string, unknown> } | null | undefined
  const lq = lifted?.quote
  if (lq) {
    const s = fromMs(lq['timestamp_start_ms']) ?? fromMs(lq['start_ms'])
    if (s != null) return s
  }
  // Insight hit: first supporting quote with a timestamp.
  for (const sq of hit.supporting_quotes ?? []) {
    const s = fromMs(sq['start_ms']) ?? fromMs(sq['timestamp_start_ms'])
    if (s != null) return s
  }
  // Fallback: metadata timestamp.
  const md = hit.metadata as Record<string, unknown>
  return fromMs(md['timestamp_start_ms']) ?? fromMs(md['start_ms'])
}
