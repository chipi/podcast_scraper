/**
 * "Have I played this?" — one answer, for every surface that asks (operator 2026-09-23).
 *
 * The question used to have two answers and the app only ever heard one of them. `completed` held
 * the episodes marked played from the ⋯ menu; the player separately recorded `finished` when an
 * episode ran out. Nothing joined them, so an episode listened to the end was played NOWHERE: it
 * stayed in "Jump back in", carried no marker in the queue or recently-played, and the catalogue's
 * Played filter matched nothing at all.
 *
 * The join now happens server-side — `GET /api/app/completed` returns the union — so this is not
 * the place that defines "played". It exists for the one case the server cannot cover in time:
 *
 *   an episode finished OFFLINE is not in the server's answer until the position flush lands.
 *
 * That is exactly the case that prompted the report (a flight), so reading the device's own record
 * as well is the difference between the list being right when you land and being right ten minutes
 * later. Once the flush completes the server agrees and this leg becomes redundant — which is the
 * intended end state, not a fallback that quietly diverges.
 *
 * Reactivity: `completed.has` is a Pinia getter, so anything computed from `isPlayed` recomputes
 * when the set loads or a mark toggles. `localPosition` is a plain module map and is NOT reactive
 * — acceptable because the only writer that matters here is a finish, and the same finish PUTs
 * `/playback`, whose response refreshes `completed`. It is a second opinion on already-settled
 * history, not a live signal.
 */
import { localPosition, recordPosition } from '../services/playbackPositions'
import { useCompletedStore } from '../stores/completed'

export function usePlayed(): {
  isPlayed: (slug: string) => boolean
  togglePlayed: (slug: string) => Promise<boolean>
} {
  const completed = useCompletedStore()

  const isPlayed = (slug: string): boolean =>
    completed.has(slug) || localPosition(slug)?.finished === true

  /**
   * Flip it, both halves at once.
   *
   * Un-playing has to clear the DEVICE's finish flag as well as the server's, or the local leg
   * above keeps answering true and the toggle only goes one way — the listener taps "Mark
   * unplayed", the row stays marked, and the control looks broken. The server clears its own half
   * (`DELETE /completed/{slug}` retracts the finish record); this clears ours.
   */
  async function togglePlayed(slug: string): Promise<boolean> {
    if (!isPlayed(slug)) return completed.mark(slug)
    const local = localPosition(slug)
    if (local?.finished) {
      // Keep the resume point: un-playing says "I did not finish it", not "I was never here".
      recordPosition(slug, local.seconds, false, false)
    }
    return completed.unmark(slug)
  }

  return { isPlayed, togglePlayed }
}
