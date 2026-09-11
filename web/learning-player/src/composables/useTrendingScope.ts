import { computed, type ComputedRef } from "vue"

import { useAuthStore } from "../stores/auth"
import { useUserPreferencesStore } from "../stores/userPreferences"

export type TrendingScope = "corpus" | "mine"

/** The app-level trending lens preference key (persisted per-user, cross-device). */
export const TRENDING_SCOPE_PREF = "lp.trendingScope"

/**
 * The personal-trending lens (#2030) as ONE source of truth: `corpus` (rising across everyone) vs
 * `mine` (ranked by the signed-in user's own engagement). Stored in `useUserPreferencesStore`
 * (`PATCH /api/app/preferences`), so the choice follows the user across devices, and forced to
 * `corpus` when signed out — `scope=mine` is auth-gated and meaningless without a listening history.
 *
 * Every trending surface (Home rails, the topic/storyline card momentum badges via
 * `useTrendingIndex`, and the Browse topic/people lists) reads this, so a single Home toggle
 * governs all of them; only Home writes it.
 */
export function useTrendingScope(): {
  scope: ComputedRef<TrendingScope>
  setScope: (v: TrendingScope) => void
} {
  const auth = useAuthStore()
  const prefs = useUserPreferencesStore()
  const scope = computed<TrendingScope>(() => {
    if (!auth.isAuthenticated) return "corpus"
    return prefs.get<TrendingScope>(TRENDING_SCOPE_PREF) === "mine" ? "mine" : "corpus"
  })
  function setScope(v: TrendingScope): void {
    void prefs.set(TRENDING_SCOPE_PREF, v)
  }
  return { scope, setScope }
}
