/**
 * Tracks whether any LAYERED sheet is currently open, as a class on `<body>`.
 *
 * The stacked-sheet geometry needs the card at the BOTTOM of a stack pinned to a fixed height:
 * the peek above each card is `parent height − child height`, and a content-sized parent made that
 * gap unpredictable — a total overlap on a short parent, an over-tall band on a long one.
 *
 * The obvious selector for "a sheet with a stacked sheet above it" cannot work here. Every sheet is
 * `Teleport`ed to `<body>`, so a stacked child is a SIBLING of the card it covers, not a
 * descendant; and `body:has(.lp-sheet--stacked)` is rejected outright by the Tailwind PostCSS
 * selector parser at build time. A counter toggling one class is both portable and exactly as
 * precise, since the components already know their own depth.
 *
 * Counted, not boolean: closing the third card in a deck must not un-pin the first two.
 */
const BODY_CLASS = "lp-stack-open"

let openCount = 0

/** Register a layered sheet for as long as it is mounted. Returns the matching release. */
export function registerStackedSheet(): () => void {
  openCount += 1
  if (openCount === 1) document.body.classList.add(BODY_CLASS)
  let released = false
  return () => {
    // Guard against a double release (StrictMode-style remounts, or an unmount racing a close):
    // decrementing twice for one sheet would un-pin the stack while cards are still on screen.
    if (released) return
    released = true
    openCount = Math.max(0, openCount - 1)
    if (openCount === 0) document.body.classList.remove(BODY_CLASS)
  }
}

/** Test seam — the counter is module state and would otherwise leak between specs. */
export function __resetSheetStack(): void {
  openCount = 0
  document.body.classList.remove(BODY_CLASS)
}
