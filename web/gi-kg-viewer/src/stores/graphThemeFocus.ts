import { defineStore } from 'pinia'
import { ref } from 'vue'

/**
 * graph-v3 tier 7-3 — legend → canvas focus bus.
 *
 * The theme legend calls `setFocus(themeIds)` when a super-theme or cluster
 * row is clicked; GraphCanvas watches this store and dims every node whose
 * `themeClusterId` is NOT in the focus set. Empty set = no focus (canvas is
 * in its default state).
 *
 * Ephemeral, not synced across tabs / devices — focus is an in-session read
 * pattern, not a persistent preference.
 *
 * Kept as a tiny standalone store (not folded into `graphLenses` or
 * `subject`) so the responsibility is single-purpose: one write path, one
 * read path, easy to test.
 */
export const useGraphThemeFocusStore = defineStore('graphThemeFocus', () => {
  const focusedThemeIds = ref<Set<string>>(new Set())

  function setFocus(ids: Iterable<string>): void {
    focusedThemeIds.value = new Set(ids)
  }

  function clearFocus(): void {
    if (focusedThemeIds.value.size === 0) return
    focusedThemeIds.value = new Set()
  }

  function hasFocus(): boolean {
    return focusedThemeIds.value.size > 0
  }

  function isFocused(themeId: string): boolean {
    return focusedThemeIds.value.has(themeId)
  }

  return {
    focusedThemeIds,
    setFocus,
    clearFocus,
    hasFocus,
    isFocused,
  }
})
