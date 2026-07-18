import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

import { useUserPreferencesStore } from './userPreferences'

export type ThemeChoice = 'light' | 'dark' | 'auto'

const STORAGE_KEY = 'gi-kg-viewer-theme'
/** USERPREFS-1 cross-device sync key (server-side namespace inside
 *  /api/app/preferences). Keep in sync with the localStorage key here. */
const PREF_KEY = 'theme'

function applyTheme(choice: ThemeChoice): void {
  const root = document.documentElement
  if (choice === 'auto') {
    root.removeAttribute('data-theme')
  } else {
    root.setAttribute('data-theme', choice)
  }
}

function coerceThemeChoice(v: unknown, fallback: ThemeChoice): ThemeChoice {
  return v === 'light' || v === 'dark' || v === 'auto' ? v : fallback
}

function loadSaved(): ThemeChoice {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v === 'light' || v === 'dark' || v === 'auto') return v
  } catch {
    /* ignore */
  }
  return 'dark'
}

export const useThemeStore = defineStore('theme', () => {
  const choice = ref<ThemeChoice>(loadSaved())

  applyTheme(choice.value)

  const userPrefs = useUserPreferencesStore()
  let applyingRemote = false

  watch(choice, (v) => {
    applyTheme(v)
    try {
      localStorage.setItem(STORAGE_KEY, v)
    } catch {
      /* ignore */
    }
    if (applyingRemote) return
    void userPrefs.set(PREF_KEY, v)
  })

  /* USERPREFS-1 — apply server-hydrated value once the preferences store
     resolves. Server wins over localStorage; missing → keep the local
     fallback. Cross-tab BroadcastChannel updates flow through the same
     path because userPrefs.state reactively updates.

     Guard note: the choice-watch above is default-async (Vue 'pre'
     flush), so clearing ``applyingRemote`` synchronously in a
     finally-block would clear it BEFORE the choice-watch fires, letting
     the remote-applied write loop right back into ``userPrefs.set``.
     Defer the clear to the next microtask so it lands AFTER the
     choice-watch callback has run. */
  watch(
    () => userPrefs.get<ThemeChoice>(PREF_KEY),
    (remote) => {
      if (remote == null) return
      const next = coerceThemeChoice(remote, choice.value)
      if (next === choice.value) return
      applyingRemote = true
      choice.value = next
      void Promise.resolve().then(() => {
        applyingRemote = false
      })
    },
    { immediate: true },
  )

  function cycle(): void {
    const order: ThemeChoice[] = ['light', 'dark', 'auto']
    const idx = order.indexOf(choice.value)
    choice.value = order[(idx + 1) % order.length]
  }

  return { choice, cycle }
})
