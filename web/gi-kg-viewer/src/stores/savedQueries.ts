import { defineStore } from 'pinia'
import { computed } from 'vue'
import { useUserPreferencesStore } from './userPreferences'

/**
 * Saved queries + Recent — Search v3 §S7 (RFC-107, ADR-119).
 *
 * Thin Pinia wrapper over ``useUserPreferencesStore`` that owns the
 * writer side of the Saved + Recent surfaces. Readers (LeftPanel,
 * CommandPalette) already ship — they render honest empty states today
 * and auto-populate the moment this store writes.
 *
 * USERPREFS-1 namespaces (server persists via ``/api/app/preferences``):
 *   * ``search.savedQueries`` — ``SavedQueryEntry[]`` (small — dozens at
 *     most). Order = user's save order (newest first when the list is
 *     rendered).
 *   * ``search.recentQueries`` — ``RecentQueryEntry[]`` ring buffer,
 *     capped at ``RECENT_MAX`` = 20 (RFC-107 §S7). Newest first.
 *
 * ADR-119 conformance: NO per-corpus persistence — the whole point of
 * moving state to USERPREFS-1 is that user prefs follow the user, not
 * the corpus. Corpus telemetry stays in ``search/query_log``.
 *
 * The wrapper is stateless — every read goes through
 * ``userPrefs.get(...)`` so the shared BroadcastChannel keeps other
 * tabs in sync without a per-tab in-memory mirror. Writes go through
 * ``userPrefs.set(...)`` which is write-through: local mirror updates
 * synchronously, PATCH fires to the server, other-tab broadcast fires.
 */

export interface SavedQueryEntry {
  /** Client-generated id — stable across renames, unique per user. */
  id: string
  /** Raw query text — this is what runs on the wire. */
  q: string
  /** Optional user-supplied display name; falls back to ``q`` in readers. */
  label?: string
  /** UNIX milliseconds. Renderers may show; also used for stable sort. */
  ts?: number
}

export interface RecentQueryEntry {
  /** Raw query text. */
  q: string
  /** UNIX milliseconds. Newest first in the buffer. */
  ts?: number
}

const SAVED_KEY = 'search.savedQueries'
const RECENT_KEY = 'search.recentQueries'
const RECENT_MAX = 20

/**
 * Stable-ish id from ``q`` + fresh nonce. Not cryptographic — collisions
 * are recovered by the "id already exists → skip" branch in ``save``.
 */
function newSavedId(q: string): string {
  const stamp = Date.now().toString(36)
  const rand = Math.random().toString(36).slice(2, 8)
  const seed = q.slice(0, 8).replace(/[^a-z0-9]/gi, '') || 'q'
  return `sq-${seed}-${stamp}-${rand}`.toLowerCase()
}

export const useSavedQueriesStore = defineStore('savedQueries', () => {
  const userPrefs = useUserPreferencesStore()

  function listSaved(): SavedQueryEntry[] {
    const raw = userPrefs.get<SavedQueryEntry[]>(SAVED_KEY)
    return Array.isArray(raw) ? raw : []
  }

  function listRecent(limit?: number): RecentQueryEntry[] {
    const raw = userPrefs.get<RecentQueryEntry[]>(RECENT_KEY)
    const list = Array.isArray(raw) ? raw : []
    if (limit == null) return list
    return list.slice(0, Math.max(0, limit))
  }

  /**
   * Push a query onto the Recent ring buffer. De-dupes on ``q`` — a
   * repeat of the same query moves to the front instead of appearing
   * twice. Timestamps use ``Date.now()`` at push time so the buffer
   * order matches the recency the user actually sees.
   *
   * Silent no-op on empty / whitespace-only ``q``.
   */
  async function pushRecent(q: string): Promise<void> {
    const term = q.trim()
    if (!term) return
    const existing = listRecent()
    const filtered = existing.filter((e) => e.q !== term)
    const next: RecentQueryEntry[] = [
      { q: term, ts: Date.now() },
      ...filtered,
    ].slice(0, RECENT_MAX)
    await userPrefs.set(RECENT_KEY, next)
  }

  /**
   * Save a query with an optional display name. Returns the created
   * ``SavedQueryEntry``. When ``q`` is already saved, updates the
   * existing entry's label + timestamp instead of duplicating.
   *
   * Silent no-op on empty / whitespace-only ``q`` (returns ``null``).
   */
  async function saveQuery(
    q: string,
    label?: string,
  ): Promise<SavedQueryEntry | null> {
    const term = q.trim()
    if (!term) return null
    const trimmedLabel = label?.trim()
    const list = listSaved()
    const existingIdx = list.findIndex((e) => e.q === term)
    let entry: SavedQueryEntry
    if (existingIdx >= 0) {
      const prior = list[existingIdx]
      entry = {
        ...prior,
        label: trimmedLabel || prior.label,
        ts: Date.now(),
      }
      const next = [entry, ...list.filter((_, i) => i !== existingIdx)]
      await userPrefs.set(SAVED_KEY, next)
    } else {
      entry = {
        id: newSavedId(term),
        q: term,
        ...(trimmedLabel ? { label: trimmedLabel } : {}),
        ts: Date.now(),
      }
      const next: SavedQueryEntry[] = [entry, ...list]
      await userPrefs.set(SAVED_KEY, next)
    }
    return entry
  }

  async function removeSaved(id: string): Promise<void> {
    const trimmed = id.trim()
    if (!trimmed) return
    const list = listSaved()
    const next = list.filter((e) => e.id !== trimmed)
    if (next.length === list.length) return // no-op
    await userPrefs.set(SAVED_KEY, next)
  }

  async function clearRecent(): Promise<void> {
    await userPrefs.set(RECENT_KEY, [])
  }

  async function resetAll(): Promise<void> {
    // Delete both keys — server contract: null = delete.
    await userPrefs.set(SAVED_KEY, null)
    await userPrefs.set(RECENT_KEY, null)
  }

  const isSaved = computed(() => (q: string): boolean => {
    const term = q.trim()
    if (!term) return false
    return listSaved().some((e) => e.q === term)
  })

  return {
    listSaved,
    listRecent,
    pushRecent,
    saveQuery,
    removeSaved,
    clearRecent,
    resetAll,
    isSaved,
    RECENT_MAX,
    SAVED_KEY,
    RECENT_KEY,
  }
})
