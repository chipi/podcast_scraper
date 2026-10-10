/**
 * Pull a bottom sheet down by its grab handle to close it (operator 2026-10-10: the handle drew the
 * gesture and did nothing — "remove that part or add pull").
 *
 * Only the handle strip starts a drag, never the sheet's body: the body scrolls, and a pull that
 * began inside a scrolled list would fight it. The sheet follows the finger downward (never up),
 * closes past {@link SHEET_DISMISS_PX} or on a quick flick, and otherwise springs back.
 */
import type { Ref } from 'vue'

/** A pull this far down closes the sheet. */
export const SHEET_DISMISS_PX = 96
/** A flick this fast (px per ms) closes it from a shorter pull. */
export const SHEET_FLICK_SPEED = 0.6
const FLICK_MIN_PX = 24
const SETTLE_MS = 180

/** Whether a pull of `dy` px over `ms` milliseconds closes the sheet. */
export function shouldDismissSheet(dy: number, ms: number): boolean {
  if (dy >= SHEET_DISMISS_PX) return true
  return dy >= FLICK_MIN_PX && ms > 0 && dy / ms >= SHEET_FLICK_SPEED
}

function reducedMotion(): boolean {
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches
  } catch {
    return false
  }
}

/** Pointer handlers to bind on the handle (`v-bind`), moving `sheet` and calling `onDismiss`. */
export function useSheetDrag(sheet: Ref<HTMLElement | null>, onDismiss: () => void) {
  let startY = 0
  let startT = 0
  let dy = 0
  let dragging = false

  function place(px: number, animate: boolean): void {
    const el = sheet.value
    if (!el) return
    el.style.transition = animate && !reducedMotion() ? `transform ${SETTLE_MS}ms ease-out` : 'none'
    el.style.transform = px ? `translateY(${px}px)` : ''
  }

  function onPointerdown(e: PointerEvent): void {
    if (e.pointerType === 'mouse' && e.button !== 0) return
    dragging = true
    startY = e.clientY
    startT = performance.now()
    dy = 0
    ;(e.currentTarget as HTMLElement | null)?.setPointerCapture?.(e.pointerId)
    place(0, false)
  }
  function onPointermove(e: PointerEvent): void {
    if (!dragging) return
    dy = Math.max(0, e.clientY - startY)
    place(dy, false)
  }
  function onPointerup(): void {
    if (!dragging) return
    dragging = false
    if (!shouldDismissSheet(dy, performance.now() - startT)) {
      place(0, true)
      return
    }
    const height = sheet.value?.offsetHeight ?? window.innerHeight
    place(height, true)
    // Put the sheet back where it belongs once it is closed: some sheets stay mounted while hidden
    // (the Brief does), and would otherwise reopen still pulled down.
    window.setTimeout(
      () => {
        onDismiss()
        place(0, false)
      },
      reducedMotion() ? 0 : SETTLE_MS,
    )
  }
  function onPointercancel(): void {
    if (!dragging) return
    dragging = false
    place(0, true)
  }

  return { onPointerdown, onPointermove, onPointerup, onPointercancel }
}
