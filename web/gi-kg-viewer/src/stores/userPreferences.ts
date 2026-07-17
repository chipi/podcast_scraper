import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import {
  fetchUserPreferences,
  patchUserPreferences,
  type UserPreferencesResponse,
} from '../api/userPreferencesApi'

/**
 * USERPREFS-1 — cross-device sync for UI opinion-state.
 *
 * Free-form key/value store on top of `/api/app/preferences`. Individual
 * feature stores (graphLenses, theme, panel state, corpus path, …) call
 * `set(key, value)` to write-through to the server + get sync across
 * devices; `get(key)` returns the last-known server value (falls back
 * to `undefined` when unset, and every consumer treats undefined as
 * "use my local default").
 *
 * Failure modes are silent by design: no server, 401, offline — the
 * store keeps its local mirror and consumers stay on their localStorage
 * fallback. Cross-device sync is a nice-to-have, not a hard requirement
 * for the viewer to function.
 *
 * Ownership: the SERVER is not the source of truth for the local
 * default — each consuming store owns its default. The server is
 * the source of truth for "the user changed it explicitly". First
 * hydrate() call applies any server values to consumers.
 */

const HYDRATION_TIMEOUT_MS = 5000

export const useUserPreferencesStore = defineStore('userPreferences', () => {
  const local = ref<Record<string, unknown>>({})
  const hydrated = ref(false)
  const hydrating = ref(false)
  /** Available on the server — turns to false permanently on 401 / 404 /
   *  network so subsequent patch calls skip the roundtrip entirely. */
  const available = ref(true)

  const state = computed(() => ({ ...local.value }))

  function apply(payload: UserPreferencesResponse | null): void {
    if (payload == null) {
      available.value = false
      return
    }
    local.value = { ...payload.preferences }
  }

  async function hydrate(): Promise<void> {
    if (hydrated.value || hydrating.value) return
    hydrating.value = true
    try {
      const controller = new AbortController()
      const t = setTimeout(() => controller.abort(), HYDRATION_TIMEOUT_MS)
      try {
        const payload = await fetchUserPreferences()
        apply(payload)
      } finally {
        clearTimeout(t)
      }
    } finally {
      hydrated.value = true
      hydrating.value = false
    }
  }

  function get<T = unknown>(key: string): T | undefined {
    return local.value[key] as T | undefined
  }

  /** Write-through: updates the local mirror synchronously (so subsequent
   *  reads see the change immediately) AND fires a PATCH to the server.
   *  Pass `null` to delete the key server-side (server contract). Never
   *  rejects — logs on failure and moves on. */
  async function set(key: string, value: unknown): Promise<void> {
    if (value === null || value === undefined) {
      delete local.value[key]
    } else {
      local.value = { ...local.value, [key]: value }
    }
    if (!available.value) return
    const patched = await patchUserPreferences({ [key]: value ?? null })
    if (patched == null) {
      available.value = false
    }
  }

  /** Write-through many keys in one server round-trip. Use for the initial
   *  localStorage → server migration when a feature store first hydrates. */
  async function setMany(updates: Record<string, unknown>): Promise<void> {
    for (const [k, v] of Object.entries(updates)) {
      if (v === null || v === undefined) delete local.value[k]
      else local.value = { ...local.value, [k]: v }
    }
    if (!available.value) return
    const normalised: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(updates)) normalised[k] = v ?? null
    const patched = await patchUserPreferences(normalised)
    if (patched == null) available.value = false
  }

  return {
    local,
    hydrated,
    hydrating,
    available,
    state,
    hydrate,
    get,
    set,
    setMany,
  }
})
