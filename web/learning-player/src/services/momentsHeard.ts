/**
 * Report short-listen events for stats (operator 2026-10-12): a clip heard to its end, and "Keep
 * listening" from a clip into its full episode. A short listen saves no position and accrues no
 * listening time, so without this it left no trace and stats could say nothing about moments.
 *
 * Watched from the app shell, like `reelProgress`, so a reel that plays on in the mini-player
 * after its page was left still counts. Signed out, nothing is sent. Best-effort: a failed post
 * loses one statistic, never the listening.
 */
import { watch, type WatchStopHandle } from 'vue'
import type { MomentEvent } from '../stores/player'
import { postMomentHeard } from './api'

export function watchMomentsHeard(source: {
  event: () => MomentEvent | null
  signedIn: () => boolean
}): WatchStopHandle {
  return watch(source.event, (e) => {
    if (!e || !source.signedIn()) return
    postMomentHeard({
      kind: e.kind,
      slug: e.slug,
      ...(e.insightId ? { insight_id: e.insightId } : {}),
      ...(e.seconds != null ? { seconds: Math.round(e.seconds * 10) / 10 } : {}),
      ...(e.topic
        ? {
            subject_id: e.topic.id,
            subject_label: e.topic.label,
            ...(e.topic.kind ? { subject_kind: e.topic.kind } : {}),
          }
        : {}),
      client_ts: Math.floor(Date.now() / 1000),
    }).catch(() => undefined)
  })
}
