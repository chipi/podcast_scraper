/**
 * Reduced-motion, in one place (S9).
 *
 * `prefers-reduced-motion` is an accessibility setting, not a preference about polish: for users
 * with vestibular disorders, motion sickness or migraine triggers, an unexpected smooth scroll can
 * cause actual nausea. It is also the setting people turn on precisely because software ignores it.
 *
 * This existed as an inline check in `TranscriptList` and nowhere else, so the player's transcript
 * scroll, the knowledge panel's insight scroll, and the mini-player's progress animation all
 * animated regardless. That is the branch's recurring shape — the right logic written once and
 * applied at one of four call sites — so it lives here now, where a call site is a visible thing
 * rather than a line someone remembered to copy.
 *
 * Read at call time, not module load: the OS setting can change mid-session, and a cached value
 * would keep animating for a user who just asked it to stop.
 */
export function prefersReducedMotion(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  )
}

/**
 * The `behavior` for a `scrollIntoView` / `scrollTo` call.
 *
 * Use this rather than a literal `'smooth'`; `spec-conformance.test.ts` fails on the literal, since
 * that is what a missed call site looks like.
 */
export function scrollBehavior(): ScrollBehavior {
  return prefersReducedMotion() ? 'auto' : 'smooth'
}
