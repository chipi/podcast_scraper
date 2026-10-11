/**
 * Remember where a short listen is (operator 2026-10-11: Home's "In progress" resumes the last one
 * left unfinished). Watched from the app shell, not the reel page, so a reel that plays on in the
 * mini-player after the listener left the page still moves its place. Each clip change writes the
 * clip; the last clip played to its end clears it. Signed out, nothing is written. Best-effort: a
 * failed write loses a resume point, never the listening.
 */
import { watch, type WatchStopHandle } from 'vue'
import type { ReelState } from '../stores/player'
import { clearReelProgress, putReelProgress } from './api'

export function watchReelProgress(
  reel: () => ReelState | null,
  signedIn: () => boolean,
): WatchStopHandle {
  let last = ''
  return watch(
    () => {
      const r = reel()
      return r?.topic ? { id: r.topic.id, index: r.index, clips: r.moments.length, done: r.done } : null
    },
    (now) => {
      if (!now || !signedIn()) return
      const key = now.done ? `${now.id}:done` : `${now.id}:${now.index}`
      if (key === last) return
      last = key
      const write = now.done ? clearReelProgress() : putReelProgress(now.id, now.index, now.clips)
      write.catch(() => undefined)
    },
  )
}
