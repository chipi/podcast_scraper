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

export type MainTabId = 'digest' | 'library' | 'graph' | 'dashboard'

/**
 * Global viewer keyboard shortcuts (documented in the shell UX spec):
 *
 * - Slash — When focus is not in an editable control: expand the left query column if needed,
 *   switch the column to **Search** mode (vs Explore), then focus `#search-q`. Does not bind
 *   single-letter G / L globally; those remain click targets on each search hit row (avoids clashing
 *   with browser find and OS bindings).
 * - 1 / 2 / 3 / 4 (#674) — When focus is not in an editable control: switch the main tab to
 *   Digest / Library / Graph / Dashboard respectively. Uses ``ev.key`` (not ``Digit*``) so the
 *   shortcut works on non-QWERTY layouts too.
 * - Escape — On the Graph main tab, when focus is not in an editable control: clear graph
 *   interaction and transient selection state.
 */
export function useViewerKeyboard(opts: {
  focusSearch: () => void
  clearGraphFocus: () => void
  isGraphTab: Ref<boolean>
  setMainTab?: (tab: MainTabId) => void
}): void {
  const tabKey: Record<string, MainTabId> = {
    1: 'digest',
    2: 'library',
    3: 'graph',
    4: 'dashboard',
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.defaultPrevented || ev.ctrlKey || ev.metaKey || ev.altKey) {
      return
    }
    if (ev.key === '/' && !targetIsEditable(ev.target)) {
      ev.preventDefault()
      opts.focusSearch()
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
