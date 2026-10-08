import { computed, ref, watch, type Ref } from 'vue'
import { getFavoritesPage } from '../services/api'
import { useFavoritesStore } from '../stores/favorites'
import type { EpisodeSummary, FavoriteEntity, FavoriteKind } from '../services/types'

/** The Saved tab's filter bar, shared by every section. */
export interface SavedFilters {
  search: Ref<string>
  color: Ref<string | null>
  sort: Ref<string>
}

const PAGE = 5
/** The server's page cap (`limit` ≤ 100). */
const MAX_LIMIT = 100
const SEARCH_DEBOUNCE_MS = 250

/**
 * One Saved section — one kind of favourite — read from the server a page at a time.
 *
 * Saved used to load every favourite with its card (a catalog read per saved episode) and filter,
 * sort and cap them on the phone. Now each section asks the server for its first five under the
 * current filters, and "Show more" asks for the next five. A filter change starts again at five; a
 * change to the favourites themselves (`store.version`: a heart, a colour) reloads the rows already
 * shown, so the list does not collapse under the user's thumb.
 *
 * `counts` is the server's per-kind tally under the same search and colour — what the type chips and
 * the "nothing matches" note need without loading every section.
 */
export function useFavoritesPage(kind: FavoriteKind, filters: SavedFilters) {
  const store = useFavoritesStore()
  const episodes = ref<EpisodeSummary[]>([])
  const entities = ref<FavoriteEntity[]>([])
  const total = ref(0)
  const counts = ref<Partial<Record<FavoriteKind, number>>>({})
  const loading = ref(false)
  const error = ref(false)
  let wanted = PAGE
  let seq = 0

  const shown = computed(() => (kind === 'episode' ? episodes.value.length : entities.value.length))

  async function fetchRange(offset: number, limit: number, append: boolean): Promise<void> {
    const mine = ++seq
    loading.value = true
    try {
      const page = await getFavoritesPage({
        kind,
        q: filters.search.value,
        color: filters.color.value,
        sort: filters.sort.value === 'title' ? 'title' : 'recent',
        offset,
        limit: Math.min(limit, MAX_LIMIT),
      })
      if (mine !== seq) return // a newer request owns the section now
      const eps = page.episodes ?? []
      const ents = page.entities ?? []
      episodes.value = append ? [...episodes.value, ...eps] : eps
      entities.value = append ? [...entities.value, ...ents] : ents
      total.value = page.total ?? eps.length + ents.length
      counts.value = page.counts ?? {}
      error.value = false
    } catch {
      if (mine === seq) error.value = true
    } finally {
      if (mine === seq) loading.value = false
    }
  }

  /** From the top, keeping as many rows as are wanted (five after a filter change). */
  function reload(): Promise<void> {
    return fetchRange(0, wanted, false)
  }

  let moreInFlight = false
  async function showMore(): Promise<void> {
    // A tap while the NEXT PAGE is on its way would supersede it (and a run of taps, every one).
    // Only a "more" blocks a "more": a background revalidation must not swallow the tap.
    if (moreInFlight) return
    wanted = shown.value + PAGE
    moreInFlight = true
    try {
      await fetchRange(shown.value, PAGE, true)
    } finally {
      moreInFlight = false
    }
  }

  /** Back to the first five ("Show less"), without a request. */
  function showLess(): void {
    wanted = PAGE
    episodes.value = episodes.value.slice(0, PAGE)
    entities.value = entities.value.slice(0, PAGE)
  }

  /** The "Show more" / "Show less" toggle: more while rows remain, else back to five. */
  function toggle(): void {
    if (total.value > shown.value) void showMore()
    else showLess()
  }

  let debounce: ReturnType<typeof setTimeout> | null = null
  watch(filters.search, () => {
    if (debounce) clearTimeout(debounce)
    debounce = setTimeout(() => {
      wanted = PAGE
      void reload()
    }, SEARCH_DEBOUNCE_MS)
  })
  watch([filters.color, filters.sort], () => {
    wanted = PAGE
    void reload()
  })
  // The favourites changed (saved, unsaved, recoloured): the same rows, refreshed.
  watch(
    () => store.version,
    () => void reload(),
  )

  return {
    episodes,
    entities,
    total,
    counts,
    loading,
    error,
    shown,
    remaining: computed(() => Math.max(0, total.value - shown.value)),
    reload,
    showMore,
    showLess,
    toggle,
  }
}
