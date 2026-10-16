/**
 * One `AudioSync` per episode and stream URL for this app session (RFC-129): the episode page and
 * the reel engine share it, so a checkpoint is fetched once. What fills an ad break changes within
 * hours, so nothing is kept across sessions.
 */

import { getAudioReference, getAudioReferenceLong } from '../services/api'
import { localKnowledgeFor } from '../services/downloads'
import { decodeWindow } from './audioFingerprint'
import { AudioSync, browserStreamAccess } from './audioSync'
import { toStream, type AudioTimeMap } from './audioTimeMap'

const syncs = new Map<string, Promise<AudioSync | null>>()

export function syncFor(slug: string, url: string, fallback: AudioTimeMap | null = null): Promise<AudioSync | null> {
  const key = `${slug}\n${url}`
  let s = syncs.get(key)
  if (!s) {
    // Offline, a downloaded episode's reference comes from its download (RFC-129).
    s = getAudioReference(slug)
      .catch(() => null)
      .then(async (ref) => ref ?? (await localKnowledgeFor(slug).catch(() => null))?.audioReference ?? null)
      .then((ref) =>
        ref && ref.checkpoints.length
          ? new AudioSync(ref, browserStreamAccess(url), fallback, (i) =>
              getAudioReferenceLong(slug, i).then((w) => (w ? decodeWindow(w.frames, ref.scale) : null)),
            )
          : null,
      )
      .catch(() => null)
    syncs.set(key, s)
  }
  return s
}

/** Resolves within `ms`, or gives up and resolves `null`. */
export function within<T>(p: Promise<T>, ms: number): Promise<T | null> {
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(null), ms)
    p.then(
      (v) => {
        clearTimeout(timer)
        resolve(v)
      },
      () => {
        clearTimeout(timer)
        resolve(null)
      },
    )
  })
}

/**
 * A clip's window in this listener's stream: its transcript times through the map measured on the
 * stream. `null` when this stream cannot be measured (in time): the caller keeps the server's.
 */
export async function clipWindow(
  slug: string,
  url: string,
  transcriptStartMs: number,
  transcriptEndMs: number,
  waitMs = 2500,
): Promise<{ startMs: number; endMs: number } | null> {
  const sync = await within(syncFor(slug, url), waitMs)
  if (!sync) return null
  const map = await within(sync.ensure(transcriptStartMs / 1000), waitMs)
  if (!map) return null
  const start = toStream(map, transcriptStartMs / 1000)
  // Measured at the clip's start: an end beyond a break that changed would be off, so the window
  // keeps the clip's length.
  return { startMs: Math.round(start * 1000), endMs: Math.round(start * 1000) + (transcriptEndMs - transcriptStartMs) }
}

export function _resetForTests(): void {
  syncs.clear()
}
