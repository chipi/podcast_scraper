/**
 * PWA update composable.
 *
 * Wraps `virtual:pwa-register/vue` (from vite-plugin-pwa) so the app
 * can surface a "New version available — Reload" toast instead of the
 * silent-update-stall trap that comes with `registerType: 'autoUpdate'`.
 *
 * The stall (per PWA shipping guide, §3): a user with a long-lived tab
 * (very common for a listening app) keeps running the old service
 * worker until they close every tab and reopen. Prompt-style hand-off
 * gives them a clear "Reload to update" action while keeping the option
 * to dismiss and stay on the current version until convenient.
 *
 * Also proactively calls `registration.update()` on visibility restore
 * (tab refocus) so a returning user quickly sees fresh SW state without
 * needing a hard reload.
 */

import { onMounted, onUnmounted, ref, watch } from 'vue'
import { useRegisterSW } from 'virtual:pwa-register/vue'
import { isNative } from '../services/native'

export interface PwaUpdateApi {
  /** True when a new service worker has installed and is waiting to take over. */
  needRefresh: ReturnType<typeof ref<boolean>>
  /** True when the app is ready to work offline (first cache filled). */
  offlineReady: ReturnType<typeof ref<boolean>>
  /** Activate the waiting SW and reload the page. */
  applyUpdate: () => Promise<void>
  /** Dismiss the "new version" prompt without reloading. */
  dismissUpdate: () => void
}

export function usePwaUpdate(): PwaUpdateApi {
  // Give the browser a friendly period to auto-check for updates without
  // waiting for navigation. 15 min is arbitrary but matches typical
  // "long-listening-session" cadence.
  const AUTO_UPDATE_CHECK_MS = 15 * 60 * 1000

  const { needRefresh, offlineReady, updateServiceWorker } = useRegisterSW({
    immediate: true,
    onRegisteredSW(_swUrl, registration) {
      if (!registration) return
      // Periodic background update check while the tab is open.
      const interval = setInterval(() => {
        void registration.update().catch(() => {
          /* network hiccup — try again next tick */
        })
      }, AUTO_UPDATE_CHECK_MS)
      // Cleanup on tab close (best-effort — browsers will GC the tab anyway).
      window.addEventListener(
        'beforeunload',
        () => clearInterval(interval),
        { once: true },
      )
    },
    onRegisterError(err) {
      // Log but do not throw — a missing SW should not brick the app.

      console.warn('[pwa] service-worker registration failed:', err)
    },
  })

  // NATIVE: no prompt — take the new worker at once. The app's code ships in the store build, but
  // the worker kept serving the PREVIOUS build's cached shell until a new one took over, so the
  // first launch after every update ran the old version (measured 2026-10-05: installed 21:53:52Z,
  // the first launch ran 21:25:15Z). One quick reload on that launch is the price.
  if (isNative()) {
    watch(
      needRefresh,
      (waiting) => {
        if (waiting) void updateServiceWorker(true)
      },
      { immediate: true },
    )
  }

  function onVisibilityChange(): void {
    if (document.hidden) return
    void navigator.serviceWorker?.getRegistration().then((reg) => {
      if (reg) void reg.update()
    })
  }

  onMounted(() => {
    document.addEventListener('visibilitychange', onVisibilityChange)
  })
  onUnmounted(() => {
    document.removeEventListener('visibilitychange', onVisibilityChange)
  })

  async function applyUpdate(): Promise<void> {
    await updateServiceWorker(true)
  }

  function dismissUpdate(): void {
    needRefresh.value = false
  }

  return { needRefresh, offlineReady, applyUpdate, dismissUpdate }
}
