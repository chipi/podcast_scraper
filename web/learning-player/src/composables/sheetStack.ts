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

/**
 * Where a sheet should teleport to: the open modal `<dialog>` if there is one, else `body`.
 *
 * The Knowledge Panel opens with `showModal()`, which puts it in the browser's TOP LAYER — above
 * every z-index, no matter how large — and marks the rest of the document `inert`. A sheet
 * teleported to `<body>` from inside it therefore rendered UNDERNEATH the panel and could not be
 * seen or tapped, while every other signal said it had opened: no navigation, no new accessibility
 * elements, no visible change (2026-09-16). Rendering inside the dialog puts the sheet in the same
 * top-layer context as the panel that spawned it.
 *
 * Resolved once, when the sheet mounts — which is exactly when the question is being asked.
 */
export function sheetTeleportTarget(): HTMLElement | string {
  const openDialog = document.querySelector("dialog[open]")
  return openDialog instanceof HTMLElement ? openDialog : "body"
}

/** Test seam — the counter is module state and would otherwise leak between specs. */
export function __resetSheetStack(): void {
  openCount = 0
  document.body.classList.remove(BODY_CLASS)
}
