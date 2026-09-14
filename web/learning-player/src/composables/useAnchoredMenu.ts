/**
 * The ONE popover shell (operator 2026-09-13) — the shared open/position/dismiss behaviour behind
 * every menu that hangs off a trigger: the ⋯ overflow, the share menu, add-to-collection. Each used
 * to hand-roll this, and they drifted (different anchors, only one clamped to the viewport, one
 * needed a z-index hack because it was `absolute` inside a card). This centralises it:
 *
 *  - **Teleport-to-<body> + `position: fixed`** — escapes clipped/transformed ancestors and the
 *    sibling-card stacking fight a card-local `absolute` panel has, with no z-index hack.
 *  - **Positioned by {@link anchorPanel}** — measured AFTER render, so the panel is clamped to its
 *    real width and can never open off the edge of the screen.
 *  - **Dismissal:** outside pointer closes; scroll/resize reposition (a fixed panel would otherwise
 *    drift); Escape closes and restores focus to the trigger.
 *
 * The caller owns the panel's CONTENT and its trigger markup; this owns the shell. Bind `triggerEl`
 * to the trigger and `panelEl` to the teleported panel; the panel starts `invisible` (a class) and
 * this composable positions it and reveals it imperatively.
 *
 * Placement is IMPERATIVE (writes `panel.style` directly) rather than a reactive `:style` binding
 * on purpose: a reactive position would re-render the panel one tick after it mounts, and that
 * second patch blurs a just-focused menu item. Writing the style directly places it in the same
 * pass the panel first renders — one render, no blur, no off-position flash.
 */
import { nextTick, onBeforeUnmount, ref, watch, type Ref } from "vue"
import { anchorPanel, type AnchorOptions } from "../utils/anchorPanel"

/**
 * The component creates the trigger/panel template refs (so the template `ref=` bindings count as
 * usage) and passes them in; this composable drives them.
 */
export function useAnchoredMenu(
  triggerEl: Ref<HTMLElement | null>,
  panelEl: Ref<HTMLElement | null>,
  opts: AnchorOptions = {},
  hooks: { onOpened?: () => void } = {}
) {
  const open = ref(false)

  function place(): void {
    const trigger = triggerEl.value
    const panel = panelEl.value
    if (!trigger || !panel) return
    const { top, left } = anchorPanel(
      trigger.getBoundingClientRect(),
      { width: panel.offsetWidth, height: panel.offsetHeight },
      { width: window.innerWidth, height: window.innerHeight },
      opts
    )
    panel.style.top = `${top}px`
    panel.style.left = `${left}px`
    panel.style.visibility = "visible"
  }

  // Placement + the open hook are scheduled from the OPEN call itself (not a watcher), so their
  // `nextTick` continuation is queued within the click handler's call stack — the same ordering the
  // hand-rolled menus had, which is what lets focus land before a test's `await trigger('click')`
  // resolves. A watcher's callback runs a flush later and loses that race.
  async function afterOpen(): Promise<void> {
    await nextTick()
    place()
    hooks.onOpened?.()
  }
  function openMenu(): void {
    if (open.value) return
    open.value = true
    void afterOpen()
  }
  function close(restoreFocus = true): void {
    if (!open.value) return
    open.value = false
    if (restoreFocus) triggerEl.value?.focus()
  }
  function toggle(): void {
    if (open.value) close()
    else openMenu()
  }

  // A pointer outside both trigger and panel dismisses. The panel is teleported, so it is NOT inside
  // the trigger's subtree — both must be checked or a click on a menu item would close the menu
  // before the item fired. Deferred to a microtask so we never mutate state (re-patching the
  // teleport) mid-dispatch of a capture-phase event.
  function onDocPointer(e: PointerEvent): void {
    const target = e.target as Node
    if (triggerEl.value?.contains(target) || panelEl.value?.contains(target)) return
    queueMicrotask(() => close(false))
  }
  // A fixed panel drifts on scroll/resize — re-place it against the trigger's new rect rather than
  // chasing or stranding it.
  function onViewportChange(): void {
    if (open.value) place()
  }
  // Escape closes and restores focus (a trap otherwise). Global while open so it fires whether focus
  // is in the panel, on the trigger, or elsewhere.
  function onKeydown(e: KeyboardEvent): void {
    if (e.key === "Escape") close()
  }

  function bind(on: boolean): void {
    if (typeof document === "undefined") return
    if (on) {
      document.addEventListener("pointerdown", onDocPointer as EventListener, true)
      document.addEventListener("keydown", onKeydown)
      window.addEventListener("scroll", onViewportChange, true)
      window.addEventListener("resize", onViewportChange)
    } else {
      document.removeEventListener("pointerdown", onDocPointer as EventListener, true)
      document.removeEventListener("keydown", onKeydown)
      window.removeEventListener("scroll", onViewportChange, true)
      window.removeEventListener("resize", onViewportChange)
    }
  }

  // (De)register the global dismiss listeners whenever the menu opens or closes. Placement + the
  // open hook are handled by openMenu (above), not here.
  watch(open, (isOpen) => bind(isOpen))
  onBeforeUnmount(() => bind(false))

  return { open, toggle, close, place }
}
