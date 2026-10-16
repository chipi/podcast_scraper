/**
 * #2359: hosts that insert ads when an episode is fetched shift the audio a listener gets from the
 * audio we transcribed. The server measures the shift (`python -m podcast_scraper.audio_sync
 * build-maps`, RFC-129) and sends it with the episode as `audio_time_map`: pairs [transcript_s, offset_s], ascending; from
 * transcript_s on, the served audio runs offset_s seconds later. Every time the API returns is
 * transcript time, so a seek converts with `toStream` and the playhead converts back with
 * `toTranscript`.
 */
export type AudioTimeMap = readonly (readonly [number, number])[]

/** Transcript seconds → seconds in the served audio. */
export function toStream(map: AudioTimeMap | null | undefined, t: number): number {
  let off = 0
  for (const [at, o] of map ?? []) {
    if (t < at) break
    off = o
  }
  return t + off
}

/**
 * Served-audio seconds → transcript seconds. Where an ad was inserted the stream has words the
 * transcript lacks: the answer holds at the point the talk resumes. Where an old ad is gone the
 * stream runs straight on into the later part, so the later offset wins.
 */
export function toTranscript(map: AudioTimeMap | null | undefined, s: number): number {
  if (!map || map.length === 0) return s
  for (let i = map.length - 1; i >= 0; i--) {
    const [at, off] = map[i]
    if (i > 0 && s < at + off) continue
    const t = s - off
    const next = map[i + 1]?.[0]
    return next !== undefined && t > next ? next : t
  }
  return s
}
