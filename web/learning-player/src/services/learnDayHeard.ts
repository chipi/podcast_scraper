/**
 * A learning day's clip heard to its end marks the day done (operator 2026-10-14).
 *
 * The player store notices the end (`learnEnd`) but may not call the API, so the shell does — here,
 * whichever page is showing, the same way the clip-heard stats are posted (`momentsHeard.ts`). A clip
 * skipped through to its end is reported not heard and leaves the day for the listener's own Done.
 */
import { watch, type WatchStopHandle } from 'vue'
import type { LearnEnd } from '../stores/player'
import { ApiError, learnOpenNext, learnStepDone } from './api'
import { track } from './analytics'

export function watchLearnDayHeard(source: {
  event: () => LearnEnd | null
  /** Record what became of the day, for the end card. */
  settle: (seq: number, status: LearnEnd['status']) => void
}): WatchStopHandle {
  return watch(
    () => source.event()?.seq,
    async () => {
      const e = source.event()
      if (!e || e.status !== 'pending') return
      if (!e.heard) {
        source.settle(e.seq, 'not_heard')
        return
      }
      try {
        await learnStepDone(e.subjectId, e.day - 1).catch(async (err: unknown) => {
          // "Play all" from a day not yet open (the one before it done, the next not opened):
          // the server answers 409 "that day has not come yet". Open it, then do it.
          if (!(err instanceof ApiError) || err.status !== 409) throw err
          await learnOpenNext(e.subjectId)
          await learnStepDone(e.subjectId, e.day - 1)
        })
        track('learn_path', { action: 'step_done', kind: e.kind, day: e.day, source: 'clip_end' })
      } catch {
        source.settle(e.seq, 'failed')
        return
      }
      // "Play all" goes on to a later day: open it now, so its clip can do it too. A day already
      // open (it came on its own) answers 409, which is fine.
      if (e.openNext) await learnOpenNext(e.subjectId).catch(() => undefined)
      source.settle(e.seq, 'done')
    },
  )
}
