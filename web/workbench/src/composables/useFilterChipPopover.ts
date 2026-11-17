/**
 * Shared open/close machinery for chip-anchored filter popovers
 * (graph #658 / library #669 / digest #670 / search+explore #671).
 *
 * Usage:
 *
 *   const anchorRef = ref<HTMLButtonElement | null>(null)
 *   const panelRef = ref<HTMLDivElement | null>(null)
 *   const { open, toggle, close } = useFilterChipPopover(anchorRef, panelRef)
 *
 * Then bind ``ref="anchorRef"`` on the chip ``<button>``, ``ref="panelRef"``
 * on the popover ``<div role="dialog">``, ``v-show="open"`` on the panel,
 * and ``@click="toggle"`` plus ``:aria-expanded="open"`` /
 * ``aria-haspopup="dialog"`` on the chip.
 *
 * The consumer owns the refs so ``vue-tsc --noUnusedLocals`` recognises
 * the template ``ref="..."`` binding as a use of the local variable.
 *
 * Behaviour:
 *   - ``pointerdown`` outside both anchor and panel → close.
 *   - ``Escape`` → close + return focus to anchor (a11y).
 *   - Listeners attach on open and detach on close / unmount.
 */
import { nextTick, onUnmounted, ref, watch, type Ref } from 'vue'

export interface FilterChipPopoverHandle {
  open: Ref<boolean>
  toggle: () => void
  close: () => void
}

export function useFilterChipPopover(
  anchorRef: Ref<HTMLButtonElement | null>,
  panelRef: Ref<HTMLDivElement | null>,
): FilterChipPopoverHandle {
  const open = ref(false)

  function close(): void {
    open.value = false
  }

  function toggle(): void {
    open.value = !open.value
  }

  function onDocPointerDown(ev: MouseEvent | PointerEvent): void {
    if (!open.value) return
    const t = ev.target
    if (!(t instanceof Node)) return
    if (anchorRef.value?.contains(t)) return
    if (panelRef.value?.contains(t)) return
    close()
  }

  function onDocKeydown(ev: KeyboardEvent): void {
    if (!open.value || ev.key !== 'Escape') return
    ev.preventDefault()
    close()
    anchorRef.value?.focus()
  }

  watch(open, async (v) => {
    if (!v) {
      document.removeEventListener('pointerdown', onDocPointerDown, true)
      document.removeEventListener('keydown', onDocKeydown, true)
      return
    }
    await nextTick()
    document.addEventListener('pointerdown', onDocPointerDown, true)
    document.addEventListener('keydown', onDocKeydown, true)
  })

  onUnmounted(() => {
    document.removeEventListener('pointerdown', onDocPointerDown, true)
    document.removeEventListener('keydown', onDocKeydown, true)
  })

  return { open, toggle, close }
}
