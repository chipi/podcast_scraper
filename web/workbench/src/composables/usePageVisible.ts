import { onScopeDispose, ref, type Ref } from 'vue'

const pageVisibleShared = ref(typeof document === 'undefined' ? true : !document.hidden)
let pageVisibleListenerCount = 0

function syncPageVisibleShared(): void {
  if (typeof document === 'undefined') {
    pageVisibleShared.value = true
    return
  }
  pageVisibleShared.value = !document.hidden
}

/**
 * Tracks ``document.visibilityState`` so polling can pause while the tab is hidden
 * (GH-743 adaptive polling / backoff). One document listener is shared across callers.
 */
export function usePageVisible(): { pageVisible: Ref<boolean> } {
  if (typeof document !== 'undefined') {
    pageVisibleListenerCount += 1
    if (pageVisibleListenerCount === 1) {
      document.addEventListener('visibilitychange', syncPageVisibleShared)
      syncPageVisibleShared()
    }
    onScopeDispose(() => {
      pageVisibleListenerCount = Math.max(0, pageVisibleListenerCount - 1)
      if (pageVisibleListenerCount === 0) {
        document.removeEventListener('visibilitychange', syncPageVisibleShared)
      }
    })
  }

  return { pageVisible: pageVisibleShared }
}
