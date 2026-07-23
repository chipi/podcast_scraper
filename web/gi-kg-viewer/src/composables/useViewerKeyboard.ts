import { onMounted, onUnmounted, type Ref } from 'vue'

function targetIsEditable(target: EventTarget | null): boolean {
  if (!target || !(target instanceof HTMLElement)) {
    return false
  }
  const tag = target.tagName
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
    return true
  }
  if (target.isContentEditable) {
    return true
  }
  return target.closest('[contenteditable="true"]') != null
}

export type MainTabId = 'digest' | 'library' | 'search' | 'graph' | 'dashboard'

/**
 * Global viewer keyboard shortcuts (documented in the shell UX spec):
 *
 * - Cmd-K / Ctrl-K (Search v3 §S3, RFC-107 §4) — opens the shell-wide command
 *   palette from ANY tab. Also fires from editable controls (so a user typing
 *   in a form can still summon the palette).
 * - Slash — Since Search v3 §S3, ALSO opens the palette (not the launcher's
 *   query field). When focus is in an editable control, `/` is a normal `/`.
 * - 1 / 2 / 3 / 4 / 5 — When focus is not in an editable control: switch the
 *   main tab to Digest / Library / Search / Graph / Dashboard respectively.
 *   Uses ``ev.key`` (not ``Digit*``) so the shortcut works on non-QWERTY
 *   layouts too. Search on 3 (RFC-107 §1 tab order); Graph shifts to 4;
 *   Dashboard shifts to 5.
 * - Escape — On the Graph main tab, when focus is not in an editable control:
 *   clear graph interaction and transient selection state. Palette handles
 *   its own Escape internally.
 */
export function useViewerKeyboard(opts: {
  /**
   * Callback for opening the Command Palette (Cmd-K / `/`). Since §S4-shell
   * the compact launcher is retired; the palette is the only `/` target.
   */
  openCommandPalette?: () => void
  /**
   * Legacy focus-launcher callback. Optional since §S4-shell (the launcher
   * no longer exists). Kept for future "focus a specific query field"
   * bindings; today it's only used as the `/` fallback when no palette
   * callback is provided (which is not a supported configuration in the
   * current shell).
   */
  focusSearch?: () => void
  clearGraphFocus: () => void
  isGraphTab: Ref<boolean>
  setMainTab?: (tab: MainTabId) => void
}): void {
  const tabKey: Record<string, MainTabId> = {
    1: 'digest',
    2: 'library',
    3: 'search',
    4: 'graph',
    5: 'dashboard',
  }

  function onKeydown(ev: KeyboardEvent): void {
    // Cmd-K / Ctrl-K opens the palette from anywhere (including editable
    // controls — that's the point of a summon shortcut).
    if ((ev.metaKey || ev.ctrlKey) && !ev.altKey && !ev.shiftKey && ev.key === 'k') {
      if (opts.openCommandPalette) {
        ev.preventDefault()
        opts.openCommandPalette()
        return
      }
    }
    if (ev.defaultPrevented || ev.ctrlKey || ev.metaKey || ev.altKey) {
      return
    }
    if (ev.key === '/' && !targetIsEditable(ev.target)) {
      ev.preventDefault()
      if (opts.openCommandPalette) {
        opts.openCommandPalette()
      } else if (opts.focusSearch) {
        opts.focusSearch()
      }
      return
    }
    if (
      opts.setMainTab
      && Object.prototype.hasOwnProperty.call(tabKey, ev.key)
      && !targetIsEditable(ev.target)
    ) {
      ev.preventDefault()
      opts.setMainTab(tabKey[ev.key])
      return
    }
    if (ev.key === 'Escape' && opts.isGraphTab.value) {
      if (targetIsEditable(ev.target)) {
        return
      }
      ev.preventDefault()
      opts.clearGraphFocus()
    }
  }

  onMounted(() => {
    window.addEventListener('keydown', onKeydown)
  })
  onUnmounted(() => {
    window.removeEventListener('keydown', onKeydown)
  })
}
