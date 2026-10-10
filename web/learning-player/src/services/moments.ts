/**
 * Load an episode's Moments reel (operator 2026-10-10).
 *
 * Online, the server's answer wins: which moments and how many are server config, tuned without an
 * app release. Offline (or when the request fails), the copy saved with the download — the reel
 * plays on a plane like the rest of a downloaded episode. Neither: an empty reel.
 */
import type { ReelMoment } from '../stores/player'
import { getMoments } from './api'
import { localKnowledgeFor } from './downloads'
import type { Moment } from './types'

export function toReel(moments: Moment[]): ReelMoment[] {
  return moments.map((m) => ({
    insightId: m.insight_id,
    text: m.text,
    speaker: m.speaker,
    startMs: m.start_ms,
    endMs: m.end_ms,
  }))
}

/** Total reel length in seconds. */
export function reelSeconds(moments: ReelMoment[]): number {
  return moments.reduce((sum, m) => sum + (m.endMs - m.startMs) / 1000, 0)
}

/** "3½"-style minutes for a reel's length: whole minutes, a half when closer to one; at least ½. */
export function minutesValue(seconds: number): string {
  const halves = Math.max(1, Math.round(seconds / 30))
  const whole = Math.floor(halves / 2)
  return halves % 2 ? `${whole || ''}½` : String(whole)
}

export async function loadMoments(slug: string): Promise<Moment[]> {
  try {
    return (await getMoments(slug)).moments
  } catch {
    return (await localKnowledgeFor(slug))?.moments ?? []
  }
}
