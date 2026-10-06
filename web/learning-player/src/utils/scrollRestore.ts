/**
 * Put the reader back where they were (operator 2026-10-04): Back from a person, topic, theme or
 * storyline returns to the exact place it was opened from, not the top of the page.
 *
 * The position alone is not enough. A detail page re-mounts and fetches on Back, so at the moment
 * the scroll is applied the page can still be a loading line a few hundred pixels tall, and the
 * browser clamps the offset to that — Back then lands at the top anyway. So the restore waits,
 * frame by frame, until the scroller is tall enough to hold the offset, and gives up after
 * `timeoutMs` (applying whatever it can reach) so a page that never grows cannot hang navigation.
 */

/** `null` means the page itself (the document scroller). */
type Scroller = HTMLElement | null

function scrollerOf(target: Scroller): Element | null {
  return target ?? (typeof document === 'undefined' ? null : document.scrollingElement)
}

function nextFrame(cb: () => void): void {
  if (typeof requestAnimationFrame === 'function') requestAnimationFrame(cb)
  else setTimeout(cb, 16)
}

/** Resolves once `target` can scroll to `top`, or once `timeoutMs` has passed. */
export function waitUntilScrollable(target: Scroller, top: number, timeoutMs = 2000): Promise<void> {
  const started = Date.now()
  return new Promise((resolve) => {
    const check = (): void => {
      const el = scrollerOf(target)
      const reachable = el ? el.scrollHeight - el.clientHeight >= top : true
      if (reachable || Date.now() - started >= timeoutMs) resolve()
      else nextFrame(check)
    }
    check()
  })
}

/**
 * Resolves with the first element matching `selector` under `root` once it exists, or null after
 * `timeoutMs`. An anchor such as a page's notes section renders only after the page's own fetch,
 * so looking for it at navigation time finds nothing and the link lands at the top.
 */
export function waitForElement(
  selector: string,
  timeoutMs = 3000,
  root: ParentNode = document
): Promise<HTMLElement | null> {
  const started = Date.now()
  return new Promise((resolve) => {
    const check = (): void => {
      const el = root.querySelector<HTMLElement>(selector)
      if (el) resolve(el)
      else if (Date.now() - started >= timeoutMs) resolve(null)
      else nextFrame(check)
    }
    check()
  })
}

/**
 * {@link waitForElement}, then wait until the element has stopped MOVING — its top unchanged for
 * `settleMs`. Existing is not enough for a section at the foot of a page: the notes render before
 * the rails above them, which then stream in and push them off screen after the scroll has landed
 * (measured 2026-10-04: the storyline page's notes ended at viewport ratio 0).
 */
/**
 * How long a link's target section may take to render. Generous on purpose: a note's Open under load
 * rendered its notes after the old 4s, and giving up meant no scroll at all — the reader left at the
 * top of a fully loaded page (measured 2026-10-04, e2e under 4 parallel workers).
 */
export const ANCHOR_WAIT_MS = 10_000

export async function waitForSettledElement(
  selector: string,
  timeoutMs = ANCHOR_WAIT_MS,
  settleMs = 300
): Promise<HTMLElement | null> {
  const started = Date.now()
  const el = await waitForElement(selector, timeoutMs)
  if (!el) return null
  return new Promise((resolve) => {
    let lastTop = el.getBoundingClientRect().top + window.scrollY
    let stableSince = Date.now()
    const check = (): void => {
      const top = el.getBoundingClientRect().top + window.scrollY
      if (top !== lastTop) {
        lastTop = top
        stableSince = Date.now()
      }
      if (Date.now() - stableSince >= settleMs || Date.now() - started >= timeoutMs) resolve(el)
      else nextFrame(check)
    }
    check()
  })
}

const USER_SCROLL_EVENTS = ["wheel", "touchstart", "pointerdown", "keydown"] as const

/**
 * Hold a restored position while the page is still filling in.
 *
 * One restore is not enough. Rails above the reader's place can arrive AFTER it — under load,
 * seconds after, with pauses between them — and every one moves the content: Back then lands above
 * or below the section the reader left, and a note's Open lands with the notes pushed off screen
 * (measured 2026-10-04: notes found 0.5s in, scrolled to, then shoved off by rails arriving later).
 *
 * So for up to `ms`, re-apply `desiredTop()` whenever the scroller has drifted from it. It must
 * never fight a scroll someone MEANT, and two signals say so:
 *
 * - the reader's own input (wheel, touch, click, key) ends the hold outright;
 * - a move AWAY from the target while the TARGET stayed put. Content arriving moves the target (the
 *   section the reader is held at shifts); a scroll the app makes on purpose (focus,
 *   scrollIntoView) moves the reader instead. A smooth scroll moving TOWARD the target is neither,
 *   so it is left to finish.
 *
 * Two end signals were tried first and both failed under load. Height stability: fetches arrive
 * more than half a second apart, so the hold ended in the pause and the next rail pushed the notes
 * away. Unchanged page height as "nobody inserted content": a rail filled in while a placeholder
 * below collapsed, the height held at 3098px and the notes still moved 1270px.
 * `startAfterMs` lets a smooth scroll get under way before the first correction.
 */
export function holdScroll(
  target: Scroller,
  desiredTop: () => number | null,
  ms = 8000,
  startAfterMs = 0
): void {
  if (typeof window === "undefined") return
  const started = Date.now()
  let released = false
  const release = (): void => {
    released = true
  }
  for (const e of USER_SCROLL_EVENTS) window.addEventListener(e, release, { passive: true, capture: true })
  const done = (): void => {
    for (const e of USER_SCROLL_EVENTS) window.removeEventListener(e, release, { capture: true })
  }
  const position = (): number => (target ? target.scrollTop : window.scrollY)
  let lastWant: number | null = null
  let lastDistance: number | null = null
  const tick = (): void => {
    if (released || Date.now() - started > ms) return done()
    const want = desiredTop()
    if (want != null) {
      const distance = Math.abs(position() - want)
      const targetStill = lastWant != null && Math.abs(want - lastWant) <= 4
      if (targetStill && lastDistance != null && distance > lastDistance + 4) return done()
      lastWant = want
      if (Date.now() - started >= startAfterMs && distance > 4) {
        if (target) target.scrollTop = want
        else window.scrollTo({ top: want })
      }
      lastDistance = Math.abs(position() - want)
    }
    nextFrame(tick)
  }
  nextFrame(tick)
}

/** The box that scrolls `el`: its nearest scrolling ancestor, or `null` for the page. */
export function scrollParentOf(el: HTMLElement): HTMLElement | null {
  let n = el.parentElement
  while (n) {
    const o = getComputedStyle(n).overflowY
    if ((o === "auto" || o === "scroll") && n.scrollHeight > n.clientHeight) return n
    n = n.parentElement
  }
  return null
}

/**
 * Keep `el` where it sat on screen (`viewportTop`) while the content around it settles.
 *
 * For a sheet closing back onto the control that opened it (operator 2026-10-04). The surface under
 * a sheet stays mounted, but it can still be LOADING: a voice tapped while the topic card was
 * half-rendered came back pushed 270px below the screen by sections that filled in above it while
 * the person sheet was open (measured: card 1016px tall at the tap, 1698px at close, scroll 0 — and
 * at scroll 0 the browser's own scroll anchoring does not engage).
 */
export function keepInPlace(el: HTMLElement, viewportTop: number): void {
  const box = scrollParentOf(el)
  const boxTop = box ? box.getBoundingClientRect().top : 0
  holdScroll(box, () => Math.max(0, offsetWithin(box, el) - (viewportTop - boxTop)))
}

/** Where `el` sits inside `target` (or the page), as a scroll offset. */
export function offsetWithin(target: Scroller, el: HTMLElement): number {
  const top = el.getBoundingClientRect().top
  return target ? top - target.getBoundingClientRect().top + target.scrollTop : top + window.scrollY
}

/**
 * Scroll `target` (or the page, for `null`) back to `top` once its content allows it, and hold it.
 *
 * ABANDONED if the reader moves first. The wait can be seconds on a slow fetch, and a restore that
 * lands after the reader has already scrolled somewhere and tapped something yanks them away from
 * it — measured 2026-10-04: Back from a similar topic, scroll to a voice, open the person on top,
 * and the late restore moved the topic underneath so the voice was gone when the person closed.
 */
export async function restoreScroll(target: Scroller, top: number, timeoutMs = 5000): Promise<void> {
  if (top <= 0 || typeof window === "undefined") return
  let moved = false
  const onInput = (): void => {
    moved = true
  }
  for (const e of USER_SCROLL_EVENTS) window.addEventListener(e, onInput, { passive: true, capture: true })
  const startedAt = target ? target.scrollTop : window.scrollY
  try {
    await waitUntilScrollable(target, top, timeoutMs)
  } finally {
    for (const e of USER_SCROLL_EVENTS) window.removeEventListener(e, onInput, { capture: true })
  }
  const now = target ? target.scrollTop : window.scrollY
  // Someone scrolled while we waited — the reader, or the app on their behalf. Theirs wins.
  if (moved || Math.abs(now - startedAt) > 4) return
  if (target) target.scrollTop = top
  else window.scrollTo({ top })
  holdScroll(target, () => top)
}
