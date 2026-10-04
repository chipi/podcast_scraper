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
import {
  ANCHOR_WAIT_MS,
  holdScroll,
  keepInPlace,
  offsetWithin,
  waitUntilScrollable,
} from './scrollRestore'

export interface ClickAnchor {
  testid: string
  index: number
  /** The element's top in the viewport when it was tapped. */
  viewportTop: number
}

let lastTarget: Element | null = null

/**
 * The CONTROL a tap belongs to: the nearest button or link that carries a testid, else the nearest
 * testid. Not just `closest('[data-testid]')` — a control can contain testid'd parts (the related
 * person's role badge, `ec-related-person-role`), and anchoring on the badge put a DIFFERENT person
 * back on screen (measured 2026-10-04: off by 221px, every time the tap landed on a badge).
 */
function controlOf(target: Element | null): HTMLElement | null {
  return (target?.closest?.(
    'button[data-testid], a[data-testid], [role="button"][data-testid]'
  ) ?? target?.closest?.('[data-testid]')) as HTMLElement | null
}
/** Where the tapped control sat on screen AT THE TAP — before any handler or late render moves it. */
let lastTap: { el: Element; top: number; at: number } | null = null

/** Remember what the reader last activated. Capture phase, so a handler that navigates still counts. */
export function trackClicks(): void {
  if (typeof window === 'undefined') return
  window.addEventListener(
    'click',
    (e) => {
      lastTarget = e.target as Element
      const control = lastTarget.closest?.('button, a, [role="button"]') ?? controlOf(lastTarget)
      lastTap = control ? { el: control, top: control.getBoundingClientRect().top, at: Date.now() } : null
    },
    { capture: true }
  )
}

/**
 * Where `el` sat on screen when the reader tapped it, if it (or something inside it) was the last
 * thing tapped; null otherwise. A sheet opens a frame or more after the tap, and content still
 * loading above the opener can move it in between — measured 2026-10-04: tapped at 629, at 902 by
 * the time the person sheet mounted, so a position read at mount put it back off screen.
 */
export function tapTopOf(el: Element): number | null {
  if (!lastTap) return null
  return el === lastTap.el || el.contains(lastTap.el) || lastTap.el.contains(el) ? lastTap.top : null
}

/**
 * Describe the element the reader just activated, for the page being left — or null when it is not
 * a PAGE element: a control inside a sheet or panel scrolls its own box, not the page.
 */
export function anchorFromLastClick(): ClickAnchor | null {
  const el = controlOf(lastTarget)
  lastTarget = null
  if (!el || !el.isConnected || el.closest('dialog, [role="dialog"]')) return null
  const testid = el.dataset.testid as string
  const index = Array.from(document.querySelectorAll(`[data-testid="${CSS.escape(testid)}"]`)).indexOf(el)
  return { testid, index, viewportTop: tapTopOf(el) ?? el.getBoundingClientRect().top }
}

function find(a: ClickAnchor, root: ParentNode = document): HTMLElement | null {
  const all = root.querySelectorAll<HTMLElement>(`[data-testid="${CSS.escape(a.testid)}"]`)
  return all[a.index] ?? null
}

/** The element the reader tapped within the last `maxAgeMs`, if it is still in the document. */
export function lastTappedElement(maxAgeMs = Infinity): HTMLElement | null {
  if (!lastTap || Date.now() - lastTap.at > maxAgeMs) return null
  const el = lastTap.el as HTMLElement
  return el.isConnected ? el : null
}

/**
 * The same description as {@link anchorFromLastClick}, for a control INSIDE `root` (a card body
 * that re-renders its content in place), without consuming the click. Index is counted within
 * `root`, so the anchor survives the card being rebuilt.
 */
export function anchorWithin(root: Element): ClickAnchor | null {
  const el = controlOf(lastTarget)
  if (!el || !root.contains(el)) return null
  const testid = el.dataset.testid as string
  const index = Array.from(root.querySelectorAll(`[data-testid="${CSS.escape(testid)}"]`)).indexOf(el)
  return { testid, index, viewportTop: tapTopOf(el) ?? el.getBoundingClientRect().top }
}

/** Put `a` back where it sat on screen inside `root` once it renders again; false if it never did. */
export async function restoreAnchorWithin(root: HTMLElement, a: ClickAnchor): Promise<boolean> {
  const el = await waitForAnchor(a, ANCHOR_WAIT_MS, root)
  if (!el) return false
  keepInPlace(el, a.viewportTop)
  return true
}

/** Resolves with the anchor's element once the re-mounted page renders it, or null after `timeoutMs`. */
export function waitForAnchor(
  a: ClickAnchor,
  timeoutMs = ANCHOR_WAIT_MS,
  root: ParentNode = document
): Promise<HTMLElement | null> {
  const started = Date.now()
  return new Promise((resolve) => {
    const check = (): void => {
      const el = find(a, root)
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
