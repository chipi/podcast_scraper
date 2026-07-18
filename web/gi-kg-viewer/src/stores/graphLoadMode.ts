import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'

import { useUserPreferencesStore } from './userPreferences'

/**
 * graph-v3 tier 8-5 — load-mode opt-in flag.
 *
 * Controls whether the graph mounts the FULL merged artifact
 * (`'everything'`, today's behaviour) or a top-down synthetic slice
 * of super-theme nodes + inter-community bridges (`'topDown'`,
 * default target for tier 8).
 *
 * Ships in tier 8-5 as a **noop toggle** — both modes render the
 * same graph today. Subsequent tiers (8-1..8-4, 8-6) wire behaviour
 * to the mode without changing this file. Landing the plumbing
 * first means each behaviour tier is a small, reviewable diff.
 *
 * Persists via USERPREFS-1 (`graphLoadMode`) so it survives reload +
 * follows the user across devices. Same-browser cross-tab sync via
 * the shared BroadcastChannel in useUserPreferencesStore.
 */

const STORAGE_KEY = 'ps_graph_load_mode'
const PREF_KEY = 'graphLoadMode'

export type GraphLoadMode = 'topDown' | 'everything'

/* Default is `'everything'` while tier 8 is being built out — flipping
 * to `'topDown'` before mount / expand / search-reveals-hidden all
 * land would render an unusable graph. When tier 8-2 is stable we'll
 * flip the default here in one line. */
const DEFAULT_MODE: GraphLoadMode = 'everything'

function readInitial(): GraphLoadMode {
  try {
    if (typeof localStorage === 'undefined') return DEFAULT_MODE
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw === 'topDown' || raw === 'everything') return raw
    return DEFAULT_MODE
  } catch {
    return DEFAULT_MODE
  }
}

export const useGraphLoadModeStore = defineStore('graphLoadMode', () => {
  const mode = ref<GraphLoadMode>(readInitial())
  const userPrefs = useUserPreferencesStore()
  let applyingRemote = false

  const isTopDown = computed(() => mode.value === 'topDown')

  function setMode(next: GraphLoadMode): void {
    if (next === mode.value) return
    mode.value = next
  }

  function toggleMode(): void {
    setMode(mode.value === 'topDown' ? 'everything' : 'topDown')
  }

  /* Write-through to localStorage + USERPREFS-1. Silent on failure —
     the localStorage mirror is authoritative until the server
     preferences endpoint responds. */
  watch(
    mode,
    (v) => {
      try {
        if (typeof localStorage !== 'undefined') {
          localStorage.setItem(STORAGE_KEY, v)
        }
      } catch {
        /* ignore quota / private mode */
      }
      if (applyingRemote) return
      void userPrefs.set(PREF_KEY, v)
    },
  )

  /* Apply server-hydrated value once the preferences store lands one.
     Same guard pattern as the graphLenses store: `applyingRemote`
     suppresses the write-through echo. Defer the clear to the next
     microtask so it lands AFTER the mode-watch (default 'pre' flush =
     async) — clearing it synchronously in a finally-block would let
     the remote-applied write loop right back into ``userPrefs.set``. */
  watch(
    () => userPrefs.get<GraphLoadMode>(PREF_KEY),
    (v) => {
      if (v !== 'topDown' && v !== 'everything') return
      if (v === mode.value) return
      applyingRemote = true
      mode.value = v
      void Promise.resolve().then(() => {
        applyingRemote = false
      })
    },
    { immediate: true },
  )

  return {
    mode,
    isTopDown,
    setMode,
    toggleMode,
  }
})
