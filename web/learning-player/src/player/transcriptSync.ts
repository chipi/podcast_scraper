/**
 * Pure transcript-sync helpers (RFC-099 §2). Kept framework-free so the active-segment math
 * and time formatting are unit-tested in isolation; the component owns the <audio> wiring,
 * autoscroll, and idle re-enable (behavioural rules live in the RFC, not here).
 */

import type { Segment } from '../services/types'

/**
 * Index of the active segment for playback time `t` (seconds): the last segment whose
 * `start <= t`. Returns -1 before the first segment starts. Assumes segments are sorted by
 * `start` (the contract guarantees this). O(log n) binary search — fits the 10k+ target.
 */
export function activeSegmentIndex(segments: Segment[], t: number): number {
  let lo = 0
  let hi = segments.length - 1
  let ans = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (segments[mid].start <= t) {
      ans = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }
  return ans
}

/** Format seconds as `m:ss` (or `h:mm:ss` past an hour). Clamps negatives/NaN to `0:00`. */
export function formatTime(seconds: number): string {
  const s = Number.isFinite(seconds) && seconds > 0 ? Math.floor(seconds) : 0
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  const ss = String(sec).padStart(2, '0')
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${ss}`
  return `${m}:${ss}`
}

/** Playback speed options (PRD-039 FR2.2). */
export const PLAYBACK_RATES = [0.75, 1, 1.25, 1.5, 2] as const
