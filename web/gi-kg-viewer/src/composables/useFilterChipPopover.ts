/**
 * Shared open/close machinery for chip-anchored filter popovers
 * (graph #658 / library #669 / digest #670 / search+explore #671).
 *
 * Usage:
 *
 *   const { open, anchorRef, panelRef, toggle, close } = useFilterChipPopover()
 *
 * Then bind ``ref="anchorRef"`` on the chip ``<button>``, ``ref="panelRef"``
 * on the popover ``<div role="dialog">``, ``v-show="open"`` on the panel,
 * and ``@click="toggle"`` plus ``:aria-expanded="open"`` /
 * ``aria-haspopup="dialog"`` on the chip.
 *
 * Behaviour:
 *   - ``pointerdown`` outside both anchor and panel → close.
 *   - ``Escape`` → close + return focus to anchor (a11y).
 *   - Listeners attach on open and detach on close / unmount.
 */
import { nextTick, onUnmounted, ref, watch } from 'vue'

export interface FilterChipPopover {
  open: ReturnType<typeof ref<boolean>>
  anchorRef: ReturnType<typeof ref<HTMLButtonElement | null>>
  panelRef: ReturnType<typeof ref<HTMLDivElement | null>>
  toggle: () => void
  close: () => void
}

export function useFilterChipPopover(): FilterChipPopover {
  const open = ref(false)
  const anchorRef = ref<HTMLButtonElement | null>(null)
  const panelRef = ref<HTMLDivElement | null>(null)

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

  return { open, anchorRef, panelRef, toggle, close }
}
