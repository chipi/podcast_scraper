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

export async function loadMoments(slug: string): Promise<Moment[]> {
  try {
    return (await getMoments(slug)).moments
  } catch {
    return (await localKnowledgeFor(slug))?.moments ?? []
  }
}
