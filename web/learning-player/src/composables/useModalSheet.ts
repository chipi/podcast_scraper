import { nextTick, onMounted, onUnmounted, watch, type Ref } from "vue"
import { useRoute, useRouter } from "vue-router"

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
export function useModalSheet(
  dialogEl: Ref<HTMLElement | null>,
  onClose: () => void,
  history?: { key: string; value: () => string }
): void {
  const route = useRoute()
  const router = useRouter()
  let restoreFocus: HTMLElement | null = null
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
    restoreFocus = document.activeElement as HTMLElement | null
    window.addEventListener("keydown", onKeydown)
    void nextTick(() => (focusables()[0] ?? dialogEl.value)?.focus())
    if (history) void router.push({ query: { ...route.query, [history.key]: history.value() } })
  })

  onUnmounted(() => {
    window.removeEventListener("keydown", onKeydown)
    restoreFocus?.focus?.()
    // Only when WE closed it (✕ / Escape / backdrop). A Back press or a navigation from inside the
    // sheet already consumed the entry — popping again would undo the user's actual navigation.
    if (history && !closedByNavigation && route.query[history.key]) void router.back()
  })
}
