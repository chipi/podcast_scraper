import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

/**
 * USERPREFS-1 — cross-device user preferences for learning-player.
 *
 * Mirrors the gi-kg-viewer store's shape (see
 * `web/gi-kg-viewer/src/stores/userPreferences.ts`) but stripped down
 * to what learning-player needs:
 *
 * - GET/PATCH `/api/app/preferences` — free-form JSON payload owned by
 *   the client; the server just round-trips it.
 * - Silent degrade when the endpoint is unavailable (unauthenticated,
 *   offline, dev server down) — the `available` flag flips false and
 *   consumers fall back to their own localStorage mirror.
 * - No BroadcastChannel yet — learning-player is single-tab-focused; a
 *   future PR can add cross-tab sync following the gi-kg-viewer pattern
 *   if a use case emerges.
 *
 * Related: gh #1213 (whole-surface USERPREFS-1 adoption).
 */

const PREFS_URL = '/api/app/preferences'

interface PrefsResponse {
  preferences: Record<string, unknown>
}

async function safeParseJson(res: Response): Promise<PrefsResponse | null> {
  let doc: unknown
  try {
    doc = await res.json()
  } catch {
    return null
  }
  if (!doc || typeof doc !== 'object') return { preferences: {} }
  const prefs = (doc as { preferences?: unknown }).preferences
  if (!prefs || typeof prefs !== 'object') return { preferences: {} }
  return { preferences: prefs as Record<string, unknown> }
}

export const useUserPreferencesStore = defineStore('userPreferences', () => {
  const preferences = ref<Record<string, unknown>>({})
  const hydrated = ref(false)
  const hydrating = ref(false)
  const available = ref(true)

  function get<T = unknown>(key: string): T | undefined {
    const v = preferences.value[key]
    return v === undefined ? undefined : (v as T)
  }

  async function hydrate(): Promise<void> {
    if (hydrated.value || hydrating.value) return
    hydrating.value = true
    try {
      const res = await fetch(PREFS_URL, {
        method: 'GET',
        credentials: 'include',
      })
      if (!res.ok) {
        available.value = false
        hydrated.value = true
        return
      }
      const parsed = await safeParseJson(res)
      preferences.value = parsed?.preferences ?? {}
      hydrated.value = true
    } catch {
      available.value = false
      hydrated.value = true
    } finally {
      hydrating.value = false
    }
  }

  async function set(key: string, value: unknown): Promise<void> {
    // Optimistic local update — feature stores watch this ref and hydrate
    // their state from it if changed. Silent-degrade the network call.
    preferences.value = { ...preferences.value, [key]: value }
    if (!available.value) return
    try {
      const res = await fetch(PREFS_URL, {
        method: 'PATCH',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [key]: value }),
      })
      if (!res.ok) available.value = false
    } catch {
      available.value = false
    }
  }

  return {
    preferences: computed(() => preferences.value),
    hydrated: computed(() => hydrated.value),
    available: computed(() => available.value),
    hydrate,
    get,
    set,
  }
})
