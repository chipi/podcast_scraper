/**
 * Post-episode recap tunables (RFC-122 / #2038).
 *
 * Kept here — a player-domain config module, like `player/insights.ts` — rather than as a magic
 * number in the view, so the end-card's pacing is one obvious edit.
 */

/**
 * How long the end-card counts down before auto-continuing the queue, when a next episode is
 * queued. "Stay" cancels it; "Play next" skips the wait. Chosen to feel like a beat, not a stall —
 * long enough to read the recap, short enough not to interrupt a hands-off listen.
 */
export const RECAP_COUNTDOWN_SECONDS = 8
