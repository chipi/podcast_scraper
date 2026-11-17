import { defineStore } from 'pinia'
import { computed, onScopeDispose, ref } from 'vue'

/* Test-only registry of live BroadcastChannel instances opened by this
 * store. Real browser tabs close the channel implicitly on unload; in
 * tests, however, the pinia store is retained across test files in the
 * same worker, so channels leak → worker event loop stays busy →
 * happy-dom's AsyncTaskManager teardown fires against live handles →
 * vitest can't finalize → v8 coverage aggregation stalls.
 *
 * ``__closeAllUserPreferencesChannels`` (invoked by ``vitest`` via
 * ``vi.hooks`` or a per-file afterEach only when needed) drains the
 * registry so no live channel escapes a test scope. Not exported from
 * the main index; consumed only by tests that mount components which
 * transitively construct this store. */
const openChannels = new Set<BroadcastChannel>()
if (typeof globalThis !== 'undefined') {
  ;(
    globalThis as unknown as { __closeAllUserPreferencesChannels?: () => void }
  ).__closeAllUserPreferencesChannels = () => {
    for (const ch of openChannels) {
      try {
        ch.close()
      } catch {
        /* channel may already be closed by GC; ignore */
      }
    }
    openChannels.clear()
  }
}

import {
  fetchUserPreferences,
  patchUserPreferences,
  replaceUserPreferences,
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
     API; skip silently.
     Teardown: register onScopeDispose to close the channel when the
     enclosing Pinia scope is disposed. Without this, tests that mount
     any component transitively importing this store leave the channel
     open across worker teardown → happy-dom's AsyncTaskManager.abortAll
     fires against a still-live handle and vitest can't finalize the
     worker, stalling coverage aggregation. Real browser tabs closing
     the tab GCs the channel anyway; this only fires in test / SSR
     dispose paths. */
  const tabId = makeTabId()
  let channel: BroadcastChannel | null = null
  try {
    if (typeof BroadcastChannel !== 'undefined') {
      channel = new BroadcastChannel(CROSS_TAB_CHANNEL)
      openChannels.add(channel)
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

  onScopeDispose(() => {
    if (channel) {
      try {
        channel.close()
      } catch {
        /* channel may already be closed by GC; ignore */
      }
      openChannels.delete(channel)
      channel = null
    }
  })

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
        const payload = await fetchUserPreferences(controller.signal)
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

  /**
   * USERPREFS-1 (#1215) — reset all preferences to their defaults.
   *
   * Behaviour:
   * - Clears the in-memory ``local`` map so consumer watchers on
   *   ``userPrefs.get(key)`` see ``undefined`` and fall back to their
   *   defaults (each store's own default value / localStorage's own
   *   default read).
   * - PUTs `{}` to `/api/app/preferences` so the server-side blob is
   *   empty too. Silent-degrade on failure — the local reset stands.
   * - Broadcasts each cleared key to other tabs so their consumers
   *   also reset.
   * - Consumer stores DO NOT auto-clear their own localStorage mirrors;
   *   that's per-store because the mirror is device-local and each
   *   store's own "reset" contract may differ (e.g. a lens store may
   *   want to reset to default-on flags, not just "unset").
   *
   * For "reset just section X", callers can iterate the section's known
   * keys and pass them to ``setMany`` with ``null`` values (existing
   * PATCH-with-null-deletes contract).
   */
  async function resetToDefaults(): Promise<void> {
    const clearedKeys = Object.keys(local.value)
    local.value = {}
    for (const k of clearedKeys) broadcast(k, null)
    if (!available.value) return
    const replaced = await replaceUserPreferences({})
    if (replaced == null) available.value = false
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
    resetToDefaults,
  }
})
