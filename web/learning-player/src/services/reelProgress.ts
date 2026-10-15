/**
 * Remember where a short listen is (operator 2026-10-11: Home's "In progress" resumes the last one
 * left unfinished). Watched from the app shell, not the reel page, so a reel that plays on in the
 * mini-player after the listener left the page still moves its place.
 *
 * - Written only once something was heard: it played, or the listener moved past clip 1. Opening a
 *   reel page and leaving (or a start the browser blocked) leaves nothing behind.
 * - Each clip change writes the clip; the last clip played to its end clears it.
 * - ✕ (closing the reel) forgets it; "Listen to the full episode" (either choice) keeps it to come
 *   back to.
 * - Clearing names the reel, so finishing one on one device never erases another kept since.
 * - Signed out, nothing is written. Best-effort: a failed write loses a resume point, never the
 *   listening.
 */
import { watch, type WatchStopHandle } from 'vue'
import type { ReelState } from '../stores/player'
import { clearReelProgress, putReelProgress } from './api'

export function watchReelProgress(source: {
  reel: () => ReelState | null
  playing: () => boolean
  left: () => { id: string; how: 'close' | 'keep' } | null
  signedIn: () => boolean
}): WatchStopHandle {
  let last = ''
  const heard = new Set<string>()
  const stopReel = watch(
    () => {
      const r = source.reel()
      return r?.topic
        ? {
            id: r.topic.id,
            minutes: r.topic.minutes,
            index: r.index,
            clips: r.moments.length,
            done: r.done,
            playing: source.playing(),
          }
        : null
    },
    (now) => {
      if (!now || !source.signedIn()) return
      if (now.playing || now.index > 0) heard.add(now.id)
      if (!heard.has(now.id)) return
      const key = now.done ? `${now.id}:done` : `${now.id}:${now.index}`
      if (key === last) return
      last = key
      const write = now.done
        ? clearReelProgress(now.id)
        : putReelProgress(now.id, now.index, now.clips, now.minutes)
      write.catch(() => undefined)
    },
  )
  const stopLeft = watch(source.left, (left) => {
    if (!left || left.how !== 'close' || !source.signedIn()) return
    last = ''
    heard.delete(left.id)
    clearReelProgress(left.id).catch(() => undefined)
  })
  return () => {
    stopReel()
    stopLeft()
  }
}
