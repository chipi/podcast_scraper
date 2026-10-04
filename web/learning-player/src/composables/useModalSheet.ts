import { nextTick, onMounted, onUnmounted, watch, type Ref } from "vue"
import { useRoute, useRouter } from "vue-router"
import { lastTappedElement, tapTopOf } from "../utils/backAnchor"
import { keepInPlace } from "../utils/scrollRestore"

/**
 * The modal-sheet plumbing shared by every teleported sheet (EntityCard, StorylineCard, QueuePanel,
 * InterestsPicker): a focus trap (Tab cycles within the dialog, Escape closes, focus restores to the
 * opener on unmount) and — for sheets that want it — a history entry so hardware/browser Back closes
 * the sheet instead of navigating the page underneath.
 *
 * Four components had hand-written copies of this (two of them also the subtle three-close-path
 * history dance). This is the `.lp-sheet` geometry lesson one layer up: shared behaviour drifts when
 * it is copied. See EntityCard's original header for the full rationale of the three close paths and
 * why the history entry goes through the router, not raw pushState.
 *
 * `history` is opt-in: pass `{ key, value }` to record `?<key>=<value>()`. Back then drops the entry,
 * the query disappears, and the watcher fires `onClose`. Sheets with no URL (queue, interests) omit
 * it and get only the focus trap.
 */
/**
 * Every query key a sheet records. The router reads this list: a navigation that only ADDS one of
 * these is a sheet opening over the page, not a new page, so the page underneath keeps its scroll
 * (operator 2026-10-04 — a stacked topic → theme → person must close back onto the exact spot).
 * Typed, so a sheet cannot use a key the router does not know about.
 */
export const SHEET_HISTORY_KEYS = ["card", "card2", "storyline", "theme"] as const
export type SheetHistoryKey = (typeof SHEET_HISTORY_KEYS)[number]

export function useModalSheet(
  dialogEl: Ref<HTMLElement | null>,
  onClose: () => void,
  history?: { key: SheetHistoryKey; value: () => string }
): void {
  const route = useRoute()
  const router = useRouter()
  let restoreFocus: HTMLElement | null = null
  /** Where the opener sat on screen when the sheet opened — it goes back there (scrollRestore). */
  let openerTop: number | null = null
  /** The query went away on its own — a Back press, or a route change from inside the sheet. */
  let closedByNavigation = false

  function focusables(): HTMLElement[] {
    if (!dialogEl.value) return []
    const sel = 'a[href], button:not([disabled]), input, [tabindex]:not([tabindex="-1"])'
    return Array.from(dialogEl.value.querySelectorAll<HTMLElement>(sel))
  }

  function onKeydown(e: KeyboardEvent): void {
    if (e.key === "Escape") {
      onClose()
      return
    }
    if (e.key !== "Tab") return
    const items = focusables()
    if (items.length === 0) return
    const first = items[0]
    const last = items[items.length - 1]
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault()
      last.focus()
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault()
      first.focus()
    }
  }

  if (history) {
    watch(
      () => route.query[history.key],
      (v) => {
        if (!v) {
          closedByNavigation = true
          onClose()
        }
      }
    )
  }

  onMounted(() => {
    // The opener: what was TAPPED just now, else what has focus (a keyboard open). Focus alone is
    // wrong on touch — a tap does not move focus there, so inside a sheet it still sat on that
    // sheet's own ✕, and the stack put the ✕ back in place instead of the voice that was tapped
    // (measured 2026-10-04: theme → person, the voice came back 163px off).
    const focused = document.activeElement as HTMLElement | null
    restoreFocus =
      lastTappedElement(1000) ?? (focused && focused !== document.body ? focused : null)
    // At the TAP if this sheet was opened by one (backAnchor), else where the opener is now.
    openerTop =
      restoreFocus && restoreFocus !== document.body
        ? (tapTopOf(restoreFocus) ?? restoreFocus.getBoundingClientRect().top)
        : null
    window.addEventListener("keydown", onKeydown)
    void nextTick(() => (focusables()[0] ?? dialogEl.value)?.focus())
    if (history) void router.push({ query: { ...route.query, [history.key]: history.value() } })
  })

  onUnmounted(() => {
    window.removeEventListener("keydown", onKeydown)
    // preventScroll: the opener goes back to where the reader saw it, not to wherever focus() would
    // nudge it; keepInPlace then holds it there while the surface underneath finishes loading.
    restoreFocus?.focus?.({ preventScroll: true })
    if (restoreFocus?.isConnected && openerTop != null) keepInPlace(restoreFocus, openerTop)
    // Only when WE closed it (✕ / Escape / backdrop). A Back press or a navigation from inside the
    // sheet already consumed the entry — popping again would undo the user's actual navigation.
    if (history && !closedByNavigation && route.query[history.key]) void router.back()
  })
}
