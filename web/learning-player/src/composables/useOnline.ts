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

/**
 * The server is reachable-but-not-working, or not reachable at all, while the DEVICE's network is
 * fine (incident 2026-09-16: a reboot lost the signing secret; the process stayed up and every
 * authed route failed). `@capacitor/network` reports the device's link state and is blind to this,
 * so before this existed the app stayed in its "online" state and rendered a hybrid of cached
 * content and error cards while insisting nothing was wrong.
 *
 * Set by the data layer (services/api.ts), which is the only thing that can observe it. Debounced
 * by a small consecutive-failure count so one slow or aborted request cannot flip the whole UI.
 */
const serverDown = ref(false)
let consecutiveFailures = 0
const FAILURES_BEFORE_DEGRADED = 2

function readForced(): boolean {
  try {
    return typeof localStorage !== 'undefined' && localStorage.getItem(FORCE_KEY) === '1'
  } catch {
    return false
  }
}

/**
 * WHY the app is offline — `null` when it is not.
 *
 * Replaces a boolean, because the three causes need different behaviour and very different words.
 * Saying "we couldn't reach the server" while the Config switch is on is false (the app chose not
 * to ask), and offering "Try again" for a request the read gate will refuse is a button that
 * cannot work.
 *
 * Precedence is deliberate: an explicit operator choice outranks a device fact, which outranks an
 * inference about the server.
 */
export type OfflineReason = 'forced' | 'network' | 'server' | null

const offlineReasonRef = computed<OfflineReason>(() => {
  if (forced.value) return 'forced'
  if (navOnline.value === false) return 'network'
  if (serverDown.value) return 'server'
  return null
})

const isOnlineRef = computed(() => offlineReasonRef.value === null)

/**
 * Report the outcome of a real request so connectivity reflects the SERVER, not just the radio.
 *
 * `ok === true` clears the degraded state immediately — that is the whole recovery path, and it is
 * why reads must never be gated on `server` (a gated read could never succeed, so the app could
 * never discover the server had come back).
 */
export function reportServerReachable(ok: boolean): void {
  if (ok) {
    consecutiveFailures = 0
    if (serverDown.value) serverDown.value = false
    return
  }
  consecutiveFailures += 1
  if (consecutiveFailures >= FAILURES_BEFORE_DEGRADED) serverDown.value = true
}

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
  offlineReason: Readonly<Ref<OfflineReason>>
  setForcedOffline: (on: boolean) => void
} {
  ensureListeners()
  return {
    isOnline: isOnlineRef,
    forcedOffline: computed(() => forced.value),
    offlineReason: offlineReasonRef,
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
  return offlineReasonRef.value !== null
}

/** Non-reactive read of WHY we are offline, for services that cannot hold a Vue ref. */
export function offlineReason(): OfflineReason {
  return offlineReasonRef.value
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
