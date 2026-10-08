import { computed, ref } from "vue"

import { useUserPreferencesStore } from "../stores/userPreferences"

/** Synced: unset → never run; `active` → running; `restart` → asked for again from Settings; `done`. */
export const GUIDED_START_PREF = "lp.guidedStart"
/** Synced: when "Not now" was tapped (ms). A `true` from before the snooze counts as long ago. */
export const GUIDED_SNOOZED_PREF = "lp.interests.dismissed"
const SNOOZED_LOCAL_KEY = "lp.interests.dismissed"
/** "Not now" is not "never" (operator 2026-10-08): the guide comes back after this long. */
export const GUIDED_SNOOZE_MS = 3 * 24 * 60 * 60 * 1000

function readLocalSnooze(): number | null {
  try {
    const v = Number(localStorage.getItem(SNOOZED_LOCAL_KEY))
    return Number.isFinite(v) && v > 1 ? v : null
  } catch {
    return null
  }
}

/**
 * The guided start's state, shared by Home (which shows it) and Settings (which can bring it back).
 *
 * "Not now" snoozes it for three days rather than hiding it for good: the listener said "not now",
 * not "never". Settings can restart it from step 1 at any time, even for a listener who already
 * has interests and shows (operator 2026-10-08).
 */
export function useGuidedStart() {
  const prefs = useUserPreferencesStore()
  const localSnooze = ref<number | null>(readLocalSnooze())
  const state = computed(() => prefs.get<string>(GUIDED_START_PREF))
  const snoozedAt = computed<number | null>(() => {
    const remote = prefs.get<unknown>(GUIDED_SNOOZED_PREF)
    return typeof remote === "number" ? remote : localSnooze.value
  })

  function isSnoozed(now = Date.now()): boolean {
    return snoozedAt.value !== null && now - snoozedAt.value < GUIDED_SNOOZE_MS
  }

  function snooze(now = Date.now()): void {
    localSnooze.value = now
    try {
      localStorage.setItem(SNOOZED_LOCAL_KEY, String(now))
    } catch {
      /* private mode / storage disabled — the synced preference still carries it */
    }
    void prefs.set(GUIDED_SNOOZED_PREF, now)
  }

  async function restart(): Promise<void> {
    localSnooze.value = null
    try {
      localStorage.removeItem(SNOOZED_LOCAL_KEY)
    } catch {
      /* nothing stored */
    }
    await Promise.all([prefs.set(GUIDED_SNOOZED_PREF, null), prefs.set(GUIDED_START_PREF, "restart")])
  }

  return { state, isSnoozed, snooze, restart }
}
