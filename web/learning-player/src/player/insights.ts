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

/**
 * How many distinct MOMENTS in the audio an insight is actually sourced to.
 *
 * Not the same number as the underline count, and that difference is the bug this replaces. Zone D
 * derived its "Sourced to N moments" receipt by counting entries in `groundedSpansBySegment`, which
 * is keyed by TRANSCRIPT SEGMENT — so one quote crossing three segments was reported to the reader
 * as three moments. The receipt exists to say "this claim is anchored in the recording, here is how
 * much"; inflating it by however finely the transcript happened to be chunked is the one thing it
 * must not do, because a grounding receipt nobody can trust is worse than no receipt.
 *
 * A moment is a supporting quote that carries a timestamp AND lands on real transcript. A quote
 * with no `start_ms` is not anchored to anything, and a degenerate window (`end_ms <= start_ms`) is
 * not a point in the audio — the same rule `quoteContains` applies, so the panel and its receipt
 * cannot disagree about what counts.
 */
export function groundedMomentCount(segments: Segment[], insight: Insight): number {
  let n = 0
  for (const q of insight.quotes) {
    if (q.start_ms == null) continue
    if (q.end_ms != null && q.end_ms <= q.start_ms) continue
    const qStart = q.start_ms / 1000
    const qEnd = (q.end_ms ?? q.start_ms + 8000) / 1000
    if (segments.some((s) => s.start < qEnd && s.end > qStart)) n++
  }
  return n
}

/** Earliest supporting-quote start (seconds) for an insight, or null when untimed. */
export function insightStartSeconds(insight: Insight): number | null {
  let best: number | null = null
  for (const q of insight.quotes) {
    if (q.start_ms != null && (best == null || q.start_ms < best)) best = q.start_ms
  }
  return best == null ? null : best / 1000
}

/**
 * How long an insight stays on screen after its quote window ends (ms; #Zone-D rewrite).
 *
 * Appearing exactly in sync with the words is honest, but vanishing the instant the sentence
 * ends is not: a listener whose eyes drop to the artwork a beat late — the common case, since
 * looking down is itself a reaction to hearing something worth a second look — would see nothing.
 * A short linger covers that without drifting into showing an insight before it's spoken, which
 * was tried and explicitly rejected (there is deliberately no symmetric lead-in).
 */
export const INSIGHT_LINGER_MS = 4000

function quoteContains(q: Quote, tMs: number, lingerMs = 0): boolean {
  if (q.start_ms == null) return false
  // A DEGENERATE WINDOW IS NOT A MOMENT IN THE AUDIO (#1978 follow-up).
  //
  // "Authored" quotes carry `start_ms: 0, end_ms: 0` — they are attached to an insight without
  // being located in the recording. Seven of the 36 corpus episodes contain them, including
  // `p09_e04` ("Risk Is a Systems Property"), which is the What's-new hero and therefore the first
  // episode most people will open.
  //
  // Before this guard, `end` resolved to 0 and the window became [0, 0 + lingerMs] — so the insight
  // was "surfacing now" at t=0, before a word had been spoken, and stayed for the whole linger. That
  // is exactly the behaviour rejected when a prototype fell back to showing the first insight, and
  // the linger reintroduced it through a different door. `nextInsightIndex` already skips these via
  // NEXT_LOOKAHEAD_FLOOR_MS; the active check had no equivalent.
  //
  // Zero or negative width means the quote was never placed in time, so it can never be the thing
  // being said right now. A missing `end_ms` is different and still gets the 8s assumption below —
  // that quote HAS a real start, we just do not know where it stops.
  if (q.end_ms != null && q.end_ms <= q.start_ms) return false
  const end = q.end_ms ?? q.start_ms + 8000 // assume ~8s when no end marker
  return tMs >= q.start_ms && tMs <= end + lingerMs
}

/**
 * Index of the insight being "spoken" at playback time `t` (seconds) — the one whose
 * supporting quote window (plus `lingerMs`, default 0 for existing callers) contains `t`.
 * Returns -1 when none is active — including before the first quote starts; there is
 * deliberately no fallback to "the next one" or "the first one", which would put an insight on
 * screen before anything has been said. Picks the latest starting match if several overlap.
 */
export function activeInsightIndex(insights: Insight[], t: number, lingerMs = 0): number {
  const tMs = t * 1000
  let ans = -1
  let bestStart = -1
  insights.forEach((ins, i) => {
    for (const q of ins.quotes) {
      if (quoteContains(q, tMs, lingerMs) && (q.start_ms ?? -1) >= bestStart) {
        bestStart = q.start_ms ?? -1
        ans = i
      }
    }
  })
  return ans
}

/** Look-ahead floor (ms) before an insight counts as "next" — see {@link nextInsightIndex}. */
const NEXT_LOOKAHEAD_FLOOR_MS = 5000

/**
 * Index of the next insight coming up after playback time `t` (seconds), or -1 when none is
 * upcoming. Picked by earliest supporting-quote start ({@link insightStartSeconds}), among
 * insights starting at least `NEXT_LOOKAHEAD_FLOOR_MS` after `t`.
 *
 * The floor exists because of degenerate `0`/`0` (untimed, "authored") quotes in real corpora:
 * without it, a naive ">= t" comparison lets an insight that is already active — or one sitting
 * at the same synthetic 0ms start — win the "next" slot at t≈0, so the panel would preview a
 * moment already playing instead of something actually ahead.
 */
export function nextInsightIndex(insights: Insight[], t: number): number {
  const tMs = t * 1000
  let ans = -1
  let bestStart = Infinity
  insights.forEach((ins, i) => {
    const startSec = insightStartSeconds(ins)
    if (startSec == null) return
    const startMs = startSec * 1000
    if (startMs >= tMs + NEXT_LOOKAHEAD_FLOOR_MS && startMs < bestStart) {
      bestStart = startMs
      ans = i
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
