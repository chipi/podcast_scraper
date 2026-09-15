import { computed, ref, type Ref } from 'vue'
import { Capacitor } from '@capacitor/core'
import { Network } from '@capacitor/network'

/**
 * Reactive online/offline awareness (F1.2).
 *
 * ONE shared listener (module singleton) so every caller reads the same flag and the DOM/native
 * listeners are registered exactly once.
 *
 * WEB: `navigator.onLine` + the window `online`/`offline` events. Reliable in a real browser.
 *
 * NATIVE (Capacitor / WKWebView): `navigator.onLine` and the window online/offline events are
 * NOT reliable — WKWebView routinely reports `navigator.onLine === false` while the device is
 * online (a false negative), especially right after cold start, and the `online` event may never
 * fire. Trusting it made `isOffline()` return true on a live network, and since the whole data
 * layer FAILS FAST on `isOffline()` (see services/api.ts), every read errored with "needs a
 * connection" — empty profile, empty collections, "couldn't reach the server" — until a reload
 * happened to re-read `navigator.onLine` as true. So on native we ignore that signal entirely and
 * use the OS-level `@capacitor/network` plugin, seeding OPTIMISTICALLY online: only a POSITIVE
 * "disconnected" report from the plugin flips us offline, never a boot-time guess.
 *
 * `navigator.onLine` is a COARSE signal even where it works — "the OS has a usable interface", not
 * "the server is reachable". It is used to FAIL FAST (skip a doomed call) and raise the banner,
 * NEVER to claim the network definitely works: an online-but-unreachable server is still handled by
 * the normal error path + the fetch safety timeout. SSR / happy-dom safe — defaults to online.
 *
 * FORCED offline (Config): a persisted testing switch that makes the whole app behave offline even
 * on a live network. It ORs with the real signal: forced OR reported-offline.
 */
const FORCE_KEY = 'lp.forceOffline'
const isNativePlatform = Capacitor.isNativePlatform()
// Seed: on the web, trust navigator.onLine. On native, seed online and let the Network plugin
// correct it — never let a WKWebView boot-time false-negative fail the first requests.
const navOnline = ref(isNativePlatform ? true : typeof navigator === 'undefined' ? true : navigator.onLine)
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
  if (initialised) return
  initialised = true
  if (isNativePlatform) {
    // The accurate OS-level signal. Seed from the current status, then track changes. The listener
    // handle is never removed on purpose — this is a process-lifetime singleton.
    void Network.getStatus()
      .then((s) => (navOnline.value = s.connected))
      .catch(() => {
        /* status unavailable — stay optimistically online; the error path still catches real failures */
      })
    void Network.addListener('networkStatusChange', (s) => (navOnline.value = s.connected))
  } else if (typeof window !== 'undefined') {
    window.addEventListener('online', () => (navOnline.value = true))
    window.addEventListener('offline', () => (navOnline.value = false))
  }
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
 * forced, or when the shared flag reports offline. On native that flag is driven by the Capacitor
 * Network plugin (never `navigator.onLine`); on the web by the window online/offline events. Never
 * blocks a call on a mere boot-time guess. Use for WRITES (route to the outbox) and the banner.
 */
export function isOffline(): boolean {
  return forced.value || navOnline.value === false
}

/**
 * ONLY the explicit Config testing switch — NOT the auto-detected signal.
 *
 * READS gate on THIS, not {@link isOffline}, so an unreliable connectivity guess can never BLOCK a
 * read on a device that is actually online (2026-09-15). The auto signal false-negatives in the
 * field — WKWebView's `navigator.onLine`, a cellular/VPN handoff, a slow `@capacitor/network`
 * report at cold start — and when reads fast-failed on it the whole app looked broken ("profile
 * stopped", "half of it doesn't work") while genuinely connected. A read now always attempts and
 * relies on the real error path (timeout + retry) to fail gracefully; the banner still reflects the
 * auto signal, it just no longer gates the request.
 */
export function isForcedOffline(): boolean {
  return forced.value
}
