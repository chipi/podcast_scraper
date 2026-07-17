import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'

import { useUserPreferencesStore } from './userPreferences'

/**
 * graph-v3 tier 8-2 — expanded-super-theme state for top-down mode.
 *
 * When the user taps a `SuperTheme` node on the canvas we add its
 * `super_theme_id` here; the top-down slice builder reads this set
 * and injects the super-theme's child TopicClusters + their tagged
 * Topics + a one-hop neighbourhood of Insights/Persons.
 *
 * Only relevant when `useGraphLoadModeStore().isTopDown`. Persisted
 * via USERPREFS-1 so the user's "I was reading about X" state
 * survives reload + cross-device sync.
 *
 * The set is kept as an array in preferences (Set doesn't survive
 * JSON round-tripping); the store hydrates back into a Set on read.
 */

const STORAGE_KEY = 'ps_graph_topdown_expanded_supers'
const PREF_KEY = 'graphTopDownExpandedSupers'

function readInitial(): Set<string> {
  try {
    if (typeof localStorage === 'undefined') return new Set()
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return new Set()
    const parsed = JSON.parse(raw) as unknown
    if (!Array.isArray(parsed)) return new Set()
    return new Set(parsed.filter((v): v is string => typeof v === 'string'))
  } catch {
    return new Set()
  }
}

export const useGraphTopDownStore = defineStore('graphTopDown', () => {
  const expandedSuperThemeIds = ref<Set<string>>(readInitial())
  const userPrefs = useUserPreferencesStore()
  let applyingRemote = false

  const hasExpansions = computed(() => expandedSuperThemeIds.value.size > 0)

  function isExpanded(sid: string): boolean {
    return expandedSuperThemeIds.value.has(sid)
  }

  function expandSuperTheme(sid: string): void {
    if (expandedSuperThemeIds.value.has(sid)) return
    const next = new Set(expandedSuperThemeIds.value)
    next.add(sid)
    expandedSuperThemeIds.value = next
  }

  function collapseSuperTheme(sid: string): void {
    if (!expandedSuperThemeIds.value.has(sid)) return
    const next = new Set(expandedSuperThemeIds.value)
    next.delete(sid)
    expandedSuperThemeIds.value = next
  }

  function toggleSuperTheme(sid: string): void {
    if (expandedSuperThemeIds.value.has(sid)) collapseSuperTheme(sid)
    else expandSuperTheme(sid)
  }

  function clearExpanded(): void {
    if (expandedSuperThemeIds.value.size === 0) return
    expandedSuperThemeIds.value = new Set()
  }

  /* Write-through: localStorage + USERPREFS-1. Same
     `applyingRemote` echo-suppression as other stores. */
  watch(
    expandedSuperThemeIds,
    (v) => {
      const arr = Array.from(v)
      try {
        if (typeof localStorage !== 'undefined') {
          if (arr.length > 0) {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(arr))
          } else {
            localStorage.removeItem(STORAGE_KEY)
          }
        }
      } catch {
        /* ignore */
      }
      if (applyingRemote) return
      void userPrefs.set(PREF_KEY, arr)
    },
  )

  watch(
    () => userPrefs.get<string[]>(PREF_KEY),
    (v) => {
      if (!Array.isArray(v)) return
      const remote = new Set(v.filter((x): x is string => typeof x === 'string'))
      /* Set-equality no-op check so a redundant hydrate doesn't cascade. */
      if (
        remote.size === expandedSuperThemeIds.value.size &&
        Array.from(remote).every((x) => expandedSuperThemeIds.value.has(x))
      ) {
        return
      }
      applyingRemote = true
      try {
        expandedSuperThemeIds.value = remote
      } finally {
        applyingRemote = false
      }
    },
    { immediate: true },
  )

  return {
    expandedSuperThemeIds,
    hasExpansions,
    isExpanded,
    expandSuperTheme,
    collapseSuperTheme,
    toggleSuperTheme,
    clearExpanded,
  }
})
