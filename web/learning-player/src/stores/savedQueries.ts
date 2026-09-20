import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import { useUserPreferencesStore } from './userPreferences'

/**
 * Saved queries (#1261-8) — the listener's ring-buffered list of searches
 * they want to check back on ("AI regulation", "sleep science"). Persisted
 * cross-device via USERPREFS-1 under a single dedicated key so the whole
 * list round-trips in one PATCH.
 *
 * Additions push to the front and dedupe by normalized query string
 * (trimmed + lowercased for comparison; original casing preserved for
 * display). Bound at ``MAX_SAVED_QUERIES`` — power-listener territory, not
 * a queue.
 */

const PREF_KEY = 'lp.savedQueries'
export const MAX_SAVED_QUERIES = 20

export interface SavedQuery {
  q: string
  scope: 'all' | 'mine'
  saved_at: number
}

function normalize(q: string): string {
  return q.trim().toLowerCase()
}

function readList(raw: unknown): SavedQuery[] {
  if (!Array.isArray(raw)) return []
  const out: SavedQuery[] = []
  for (const item of raw) {
    if (!item || typeof item !== 'object') continue
    const q = (item as Record<string, unknown>).q
    if (typeof q !== 'string' || !q.trim()) continue
    const rawScope = (item as Record<string, unknown>).scope
    const scope: 'all' | 'mine' = rawScope === 'mine' ? 'mine' : 'all'
    const savedAtRaw = (item as Record<string, unknown>).saved_at
    const savedAt =
      typeof savedAtRaw === 'number' && Number.isFinite(savedAtRaw) ? savedAtRaw : 0
    out.push({ q: q.trim(), scope, saved_at: savedAt })
  }
  return out.slice(0, MAX_SAVED_QUERIES)
}

export const useSavedQueriesStore = defineStore('savedQueries', () => {
  const prefs = useUserPreferencesStore()
  const items = ref<SavedQuery[]>([])

  /* Our own writes are already applied locally; this guard stops the mirror below from
     re-applying them, and — the part that mattered — from reverting them. Same
     echo-suppression `graphTopDown` uses; this store was the one that never got it. */
  let writingLocally = false

  // Mirror the preferences store's payload → refresh on any prefs mutation
  // (whether from initial hydrate, another feature's write, or an external
  // patch). Immediate so hydrated state is honored on first read.
  //
  // Skipped while a local write is in flight. `prefs.set` updates its ref optimistically and THEN
  // PATCHes; any refresh resolving in between carries a server snapshot that predates our write,
  // and applying it silently undid the write we had just shown the user. The symptom was Save →
  // tap again to undo → still "Saved ✓": the revert made `isSaved()` false, so the second tap took
  // the save branch and re-saved. Intermittent by nature — it needs a refresh to land between two
  // taps — which is exactly why it surfaced as a flaky e2e rather than a reported bug.
  // `flush: 'sync'` is load-bearing, not a style choice. With the default ('pre') the callback is
  // queued and can run AFTER the write has settled and the flag is back to false — so the guard
  // would read as protecting the window while letting the very interleaving it exists for through.
  watch(
    () => prefs.get<unknown>(PREF_KEY),
    (raw) => {
      if (writingLocally) return
      items.value = readList(raw)
    },
    { immediate: true, flush: 'sync' },
  )

  /** Apply optimistically, persist, and keep the mirror off our back until the write settles. */
  async function commit(next: SavedQuery[]): Promise<void> {
    items.value = next
    writingLocally = true
    try {
      await prefs.set(PREF_KEY, next)
    } finally {
      writingLocally = false
    }
  }

  const list = computed<SavedQuery[]>(() => items.value)
  const count = computed(() => items.value.length)

  function isSaved(q: string, scope: 'all' | 'mine' = 'all'): boolean {
    const key = normalize(q)
    if (!key) return false
    return items.value.some((it) => normalize(it.q) === key && it.scope === scope)
  }

  async function save(
    q: string,
    scope: 'all' | 'mine' = 'all',
    now: number = Date.now(),
  ): Promise<void> {
    const key = normalize(q)
    if (!key) return
    const filtered = items.value.filter(
      (it) => !(normalize(it.q) === key && it.scope === scope),
    )
    const next = [{ q: q.trim(), scope, saved_at: now }, ...filtered].slice(0, MAX_SAVED_QUERIES)
    await commit(next)
  }

  async function remove(q: string, scope: 'all' | 'mine' = 'all'): Promise<void> {
    const key = normalize(q)
    if (!key) return
    const next = items.value.filter(
      (it) => !(normalize(it.q) === key && it.scope === scope),
    )
    if (next.length === items.value.length) return
    await commit(next)
  }

  async function clear(): Promise<void> {
    if (!items.value.length) return
    await commit([])
  }

  return { list, count, isSaved, save, remove, clear }
})
