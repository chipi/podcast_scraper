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
/** Same-browser cross-tab broadcast channel name (BroadcastChannel API).
 *  Every tab writes on `set()` and listens for other tabs' writes so
 *  toggling a lens in tab A propagates to tab B without waiting for the
 *  next hydrate. Independent of the server round-trip — same-browser
 *  sync works even when the user is offline / unauthenticated. */
const CROSS_TAB_CHANNEL = 'ps_user_preferences_sync'

interface CrossTabMessage {
  /** Random per-tab id so the sender ignores its own echo. */
  senderId: string
  key: string
  /** `null` = delete key (matches server + set() semantics). */
  value: unknown
}

/** Per-tab id survives one tab lifetime — no need to persist. */
function makeTabId(): string {
  const bytes = new Uint8Array(6)
  crypto.getRandomValues(bytes)
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
}

export const useUserPreferencesStore = defineStore('userPreferences', () => {
  const local = ref<Record<string, unknown>>({})
  const hydrated = ref(false)
  const hydrating = ref(false)
  /** Available on the server — turns to false permanently on 401 / 404 /
   *  network so subsequent patch calls skip the roundtrip entirely. */
  const available = ref(true)

  const state = computed(() => ({ ...local.value }))

  /* BroadcastChannel setup — happy-dom and older browsers may lack the
     API; skip silently. */
  const tabId = makeTabId()
  let channel: BroadcastChannel | null = null
  try {
    if (typeof BroadcastChannel !== 'undefined') {
      channel = new BroadcastChannel(CROSS_TAB_CHANNEL)
      channel.addEventListener('message', (ev: MessageEvent<CrossTabMessage>) => {
        const msg = ev.data
        if (!msg || typeof msg !== 'object') return
        if (msg.senderId === tabId) return
        if (typeof msg.key !== 'string') return
        if (msg.value === null || msg.value === undefined) {
          const next = { ...local.value }
          delete next[msg.key]
          local.value = next
        } else {
          local.value = { ...local.value, [msg.key]: msg.value }
        }
      })
    }
  } catch {
    /* not supported / permission denied — cross-tab is opportunistic */
  }

  function broadcast(key: string, value: unknown): void {
    if (!channel) return
    try {
      channel.postMessage({ senderId: tabId, key, value } as CrossTabMessage)
    } catch {
      /* postMessage can throw for un-cloneable payloads; drop silently */
    }
  }

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
   *  reads see the change immediately), broadcasts to other tabs via
   *  BroadcastChannel (same-browser cross-tab), AND fires a PATCH to the
   *  server. Pass `null` to delete the key server-side (server contract).
   *  Never rejects — logs on failure and moves on. */
  async function set(key: string, value: unknown): Promise<void> {
    if (value === null || value === undefined) {
      delete local.value[key]
    } else {
      local.value = { ...local.value, [key]: value }
    }
    broadcast(key, value ?? null)
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
      broadcast(k, v ?? null)
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
