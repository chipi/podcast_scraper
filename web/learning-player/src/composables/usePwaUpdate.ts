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
 *
 * WEB ONLY. The native app runs the code it was installed with, and nothing else (operator
 * 2026-10-06): its pages and logic change only through a store release; the server supplies the API
 * and media. A service worker there could only ever be BEHIND the install — its scope is the app's
 * own origin, so it cached the previous install's code and served it on the first launch after every
 * update (measured: installed 21:53:52Z, the first launch ran 21:25:15Z). It never brought anything
 * from the web deploy. So native registers none, and removes any worker and caches an earlier
 * version left behind. Offline data on native lives in `contentCache` (Filesystem) and downloaded
 * artwork in the downloads store, neither of which is the worker's.
 */

import { onMounted, onUnmounted, ref } from 'vue'
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

/** Unregister every service worker and delete every Cache Storage cache this origin holds. */
async function removeServiceWorkers(): Promise<void> {
  try {
    const regs = (await navigator.serviceWorker?.getRegistrations()) ?? []
    await Promise.all(regs.map((r) => r.unregister()))
    if (typeof caches !== 'undefined') {
      const keys = await caches.keys()
      await Promise.all(keys.map((k) => caches.delete(k)))
    }
  } catch (err) {
    console.warn('[pwa] removing the native service worker failed:', err)
  }
}

export function usePwaUpdate(): PwaUpdateApi {
  if (isNative()) {
    void removeServiceWorkers()
    return {
      needRefresh: ref(false),
      offlineReady: ref(false),
      applyUpdate: async () => {},
      dismissUpdate: () => {},
    }
  }

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
