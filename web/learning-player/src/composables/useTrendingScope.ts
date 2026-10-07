import { computed, type ComputedRef } from "vue"

import { useAuthStore } from "../stores/auth"
import { useUserPreferencesStore } from "../stores/userPreferences"

export type TrendingScope = "corpus" | "mine"

/** The app-level trending lens preference key (persisted per-user, cross-device). */
export const TRENDING_SCOPE_PREF = "lp.trendingScope"

/**
 * The personal-trending lens (#2030): `mine` (only the listener's own world — what they heard,
 * saved or follow) vs `corpus` (rising across everyone). Stored in `useUserPreferencesStore`
 * (`PATCH /api/app/preferences`), so the choice follows the user across devices, and forced to
 * `corpus` when signed out — `scope=mine` is auth-gated and meaningless without a listening history.
 *
 * `mine` is the DEFAULT for a signed-in listener (operator 2026-10-07); they can switch it off, and
 * the choice sticks. Only the Trends section reads this. Card momentum badges and the Browse topic /
 * people lists stay corpus-wide: since "mine" became strictly the listener's own world, following
 * it there would strip the badge from every topic outside it and empty Browse for a new account.
 */
export function useTrendingScope(): {
  scope: ComputedRef<TrendingScope>
  setScope: (v: TrendingScope) => void
} {
  const auth = useAuthStore()
  const prefs = useUserPreferencesStore()
  const scope = computed<TrendingScope>(() => {
    if (!auth.isAuthenticated) return "corpus"
    return prefs.get<TrendingScope>(TRENDING_SCOPE_PREF) === "corpus" ? "corpus" : "mine"
  })
  function setScope(v: TrendingScope): void {
    void prefs.set(TRENDING_SCOPE_PREF, v)
  }
  return { scope, setScope }
}
