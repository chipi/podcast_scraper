/**
 * Back returns the reader to the CONTROL they tapped, not to a pixel offset (operator 2026-10-04).
 *
 * A saved offset is only right if the page is laid out exactly as it was. It usually is not: the
 * reader taps a person while rails above are still loading, and by the time they come Back the page
 * has its full height. Measured on the storyline page: the voice was tapped at offset 949 on a page
 * 1709px tall; on Back the page was 2994px and offset 949 showed something else entirely.
 *
 * So when a link on the page is followed, remember WHICH element it was (its `data-testid` and its
 * index among same-testid elements) and where it sat in the viewport. On Back, wait for that element
 * to exist again and scroll so it sits in the same place — then keep it there while the rest of the
 * page fills in (`holdScroll` follows the element, not a number). Falls back to the offset when the
 * element cannot be identified.
 */
import { ANCHOR_WAIT_MS, holdScroll, offsetWithin, waitUntilScrollable } from './scrollRestore'

export interface ClickAnchor {
  testid: string
  index: number
  /** The element's top in the viewport when it was tapped. */
  viewportTop: number
}

let lastTarget: Element | null = null

/** Remember what the reader last activated. Capture phase, so a handler that navigates still counts. */
export function trackClicks(): void {
  if (typeof window === 'undefined') return
  window.addEventListener('click', (e) => (lastTarget = e.target as Element), { capture: true })
}

/**
 * Describe the element the reader just activated, for the page being left — or null when it is not
 * a PAGE element: a control inside a sheet or panel scrolls its own box, not the page.
 */
export function anchorFromLastClick(): ClickAnchor | null {
  const el = lastTarget?.closest?.('[data-testid]') as HTMLElement | null
  lastTarget = null
  if (!el || !el.isConnected || el.closest('dialog, [role="dialog"]')) return null
  const testid = el.dataset.testid as string
  const index = Array.from(document.querySelectorAll(`[data-testid="${CSS.escape(testid)}"]`)).indexOf(el)
  return { testid, index, viewportTop: el.getBoundingClientRect().top }
}

function find(a: ClickAnchor): HTMLElement | null {
  const all = document.querySelectorAll<HTMLElement>(`[data-testid="${CSS.escape(a.testid)}"]`)
  return all[a.index] ?? null
}

/** Resolves with the anchor's element once the re-mounted page renders it, or null after `timeoutMs`. */
export function waitForAnchor(a: ClickAnchor, timeoutMs = ANCHOR_WAIT_MS): Promise<HTMLElement | null> {
  const started = Date.now()
  return new Promise((resolve) => {
    const check = (): void => {
      const el = find(a)
      if (el) resolve(el)
      else if (Date.now() - started >= timeoutMs) resolve(null)
      else requestAnimationFrame(check)
    }
    check()
  })
}

/**
 * The page offset that puts the anchor back where it was, held while the page fills in. Null when
 * the element never came back (the caller falls back to the saved offset).
 */
export async function restoreToAnchor(a: ClickAnchor): Promise<number | null> {
  const el = await waitForAnchor(a)
  if (!el) return null
  const want = (): number => Math.max(0, offsetWithin(null, el) - a.viewportTop)
  await waitUntilScrollable(null, want(), 5000)
  holdScroll(null, want)
  return want()
}
