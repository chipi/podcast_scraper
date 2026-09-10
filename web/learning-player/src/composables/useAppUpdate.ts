/**
 * Native app-update check (wave-I.6).
 *
 * The player app and the backend are versioned **independently** (`__APP_VERSION__` is the
 * learning-player package; the backend has its own `code_version`), so the update signal is the
 * server's published `player_version` — the current RELEASED player-app version, on the same scale
 * as the baked `__APP_VERSION__`. When the running native build is older, an update is available.
 *
 * Native only: on the web the service worker owns updates (`PwaUpdateToast` reloads to the new
 * bundle). Native builds are baked and update through the store, so this is the native concern.
 * When the deploy hasn't set `player_version` (null), the check is skipped rather than compare
 * mismatched scales — a false "update available" is worse than none.
 */
import { ref } from 'vue'

import { getHealth } from '../services/api'
import { isNative } from '../services/native'

/** Leading numeric parts of a dot-version ("1.2.0" → [1,2,0]); non-numeric suffixes → 0. */
function parts(v: string): number[] {
  return v.split('.').map((p) => {
    const n = parseInt(p, 10)
    return Number.isFinite(n) ? n : 0
  })
}

/** True when `candidate` is strictly newer than `current`, compared numerically part-by-part. */
export function isVersionNewer(candidate: string, current: string): boolean {
  const a = parts(candidate)
  const b = parts(current)
  const len = Math.max(a.length, b.length)
  for (let i = 0; i < len; i++) {
    const x = a[i] ?? 0
    const y = b[i] ?? 0
    if (x !== y) return x > y
  }
  return false
}

/**
 * The store / update-channel URL for native. EMPTY pre-launch (the app is TestFlight-only, no
 * published App Store URL yet) — when unset the banner stays informational: a truthful "update
 * available" with no dead link. Fill this at launch.
 */
export const APP_STORE_URL = ''

export function useAppUpdate() {
  const updateAvailable = ref(false)
  const latestVersion = ref<string | null>(null)
  const dismissed = ref(false)

  /** Best-effort: never throws. No-op on web and when the server hasn't published a version. */
  async function check(): Promise<void> {
    if (!isNative()) return
    const health = await getHealth()
    const server = health?.player_version
    if (!server) return
    if (isVersionNewer(server, __APP_VERSION__)) {
      latestVersion.value = server
      updateAvailable.value = true
    }
  }

  function dismiss(): void {
    dismissed.value = true
  }

  return { updateAvailable, latestVersion, dismissed, dismiss, storeUrl: APP_STORE_URL, check }
}
