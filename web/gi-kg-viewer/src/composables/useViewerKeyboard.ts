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

/**
 * RFC-062: / focuses search (when not typing in a field); Escape clears graph
 * interaction on the Graph tab (when focus is not in an editable control).
 */
export function useViewerKeyboard(opts: {
  focusSearch: () => void
  clearGraphFocus: () => void
  isGraphTab: Ref<boolean>
}): void {
  function onKeydown(ev: KeyboardEvent): void {
    if (ev.defaultPrevented || ev.ctrlKey || ev.metaKey || ev.altKey) {
      return
    }
    if (ev.key === '/' && !targetIsEditable(ev.target)) {
      ev.preventDefault()
      opts.focusSearch()
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
