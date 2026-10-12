/**
 * Where a highlight is being saved — a full episode, a short listen, or a learning day — for the
 * stats that say where highlights come from (operator 2026-10-12).
 *
 * The capture store cannot ask the player itself (the player store already reads the capture
 * store), so the app shell registers how to tell: a short listen playing is `moment`; an episode
 * opened from a learning day (`?from=learn`) is `learn` while it stays open; anything else is
 * `episode`.
 */
export type CaptureSource = 'episode' | 'moment' | 'learn'

let inShortListen: () => boolean = () => false
let learnSlug: string | null = null

/** The app shell says how to tell a short listen is playing. */
export function setShortListenProbe(probe: () => boolean): void {
  inShortListen = probe
}

/** The player page says which episode, if any, was opened from a learning day. */
export function setLearnOrigin(slug: string | null): void {
  learnSlug = slug
}

export function captureSource(slug: string): CaptureSource {
  if (inShortListen()) return 'moment'
  return learnSlug && learnSlug === slug ? 'learn' : 'episode'
}
