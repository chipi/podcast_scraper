import { readonly, ref, type Ref } from 'vue'

/**
 * Reactive online/offline awareness (F1.2).
 *
 * ONE shared listener (module singleton) wraps `navigator.onLine` + the window `online`/`offline`
 * events, so every caller reads the same flag and the DOM listeners are registered exactly once.
 *
 * `navigator.onLine` is a COARSE signal — "the OS has a usable network interface", not "the server
 * is reachable". So it is used to FAIL FAST (skip a call that cannot succeed) and to raise the
 * offline banner, NEVER to claim the network definitely works: an online-but-unreachable server is
 * still handled by the normal error path (and the fetch safety timeout). SSR / happy-dom safe —
 * defaults to online when `navigator`/`window` are absent, so nothing renders as "offline" in a
 * non-browser test env unless a test drives the events.
 */
const online = ref(typeof navigator === 'undefined' ? true : navigator.onLine)

let initialised = false
function ensureListeners(): void {
  if (initialised || typeof window === 'undefined') return
  initialised = true
  window.addEventListener('online', () => (online.value = true))
  window.addEventListener('offline', () => (online.value = false))
}

export function useOnline(): { isOnline: Readonly<Ref<boolean>> } {
  ensureListeners()
  return { isOnline: readonly(online) }
}

/**
 * Non-reactive read for the data layer (services), which must not hold a Vue ref. `true` when the
 * environment cannot tell (never blocks a call on a guess).
 */
export function isOffline(): boolean {
  return typeof navigator !== 'undefined' && navigator.onLine === false
}
