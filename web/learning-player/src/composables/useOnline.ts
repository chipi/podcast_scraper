import { computed, ref, type Ref } from 'vue'

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
 *
 * FORCED offline (Config): a persisted testing switch that makes the whole app behave offline even
 * on a live network — the banner shows, reads fail fast to cache/graceful-error, and WRITES fail
 * fast so the store routes them to the outbox for replay (api.ts :: apiFetch gates non-GET on
 * isOffline) — so the full offline UX can be exercised without pulling the cable. It ORs with the
 * real signal: forced OR navigator-offline.
 */
const FORCE_KEY = 'lp.forceOffline'
const navOnline = ref(typeof navigator === 'undefined' ? true : navigator.onLine)
const forced = ref(readForced())

function readForced(): boolean {
  try {
    return typeof localStorage !== 'undefined' && localStorage.getItem(FORCE_KEY) === '1'
  } catch {
    return false
  }
}

const isOnlineRef = computed(() => navOnline.value && !forced.value)

let initialised = false
function ensureListeners(): void {
  if (initialised || typeof window === 'undefined') return
  initialised = true
  window.addEventListener('online', () => (navOnline.value = true))
  window.addEventListener('offline', () => (navOnline.value = false))
}

export function useOnline(): {
  isOnline: Readonly<Ref<boolean>>
  forcedOffline: Readonly<Ref<boolean>>
  setForcedOffline: (on: boolean) => void
} {
  ensureListeners()
  return {
    isOnline: isOnlineRef,
    forcedOffline: computed(() => forced.value),
    setForcedOffline,
  }
}

/** Persist + apply the forced-offline testing switch. */
export function setForcedOffline(on: boolean): void {
  forced.value = on
  try {
    if (on) localStorage.setItem(FORCE_KEY, '1')
    else localStorage.removeItem(FORCE_KEY)
  } catch {
    /* storage blocked — the in-memory flag still applies for this session */
  }
}

/**
 * Non-reactive read for the data layer (services), which must not hold a Vue ref. `true` when
 * forced, or when the environment reports offline. Never blocks a call on a mere guess.
 */
export function isOffline(): boolean {
  return forced.value || (typeof navigator !== 'undefined' && navigator.onLine === false)
}
