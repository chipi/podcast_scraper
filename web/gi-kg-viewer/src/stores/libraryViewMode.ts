import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

import { useUserPreferencesStore } from './userPreferences'

/**
 * USERPREFS-1 adoption for Library tab view mode.
 *
 * Was previously a component-local ref in ``LibraryTab.vue`` with just
 * ``localStorage.setItem/getItem``. Promoted to a store + USERPREFS-1
 * write-through so the operator's choice of Shows vs Episodes syncs
 * across devices, matching the other USERPREFS-1-adopted stores
 * (``graphLoadMode``, ``graphLenses``, ``theme``).
 *
 * Follows the same shape as ``graphLoadMode.ts``:
 * - localStorage mirror is authoritative until the server responds,
 * - write-through PATCH on every mutation,
 * - server-hydrated value applied via a guarded watcher to avoid
 *   echoing back to the server.
 */

export type LibraryViewMode = 'shows' | 'episodes'

const STORAGE_KEY = 'gikg.library.mode'
const PREF_KEY = 'libraryViewMode'

const DEFAULT_MODE: LibraryViewMode = 'episodes'

function readInitial(): LibraryViewMode {
  try {
    if (typeof localStorage === 'undefined') return DEFAULT_MODE
    return localStorage.getItem(STORAGE_KEY) === 'shows' ? 'shows' : 'episodes'
  } catch {
    return DEFAULT_MODE
  }
}

export const useLibraryViewModeStore = defineStore('libraryViewMode', () => {
  const mode = ref<LibraryViewMode>(readInitial())
  const userPrefs = useUserPreferencesStore()
  let applyingRemote = false

  function setMode(next: LibraryViewMode): void {
    if (next === mode.value) return
    mode.value = next
  }

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

  watch(
    () => userPrefs.get<LibraryViewMode>(PREF_KEY),
    (v) => {
      if (v !== 'shows' && v !== 'episodes') return
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
    setMode,
  }
})
