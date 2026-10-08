import { computed, ref, watch, type Ref } from 'vue'
import { getNotesPage } from '../services/api'
import { useCaptureStore } from '../stores/capture'
import type { Note } from '../services/types'

const PAGE = 5
const MAX = 100
const SEARCH_DEBOUNCE_MS = 250

/**
 * The user's notes, newest first, a page at a time from the server (`getNotesPage`).
 *
 * `counts` are the kind chips' numbers and are counted over ALL notes, never the search: a chip
 * that changed its number as you typed would be answering a different question from the one it
 * asks (operator 2026-09-19). So a search costs one extra row-less request for them.
 *
 * The highlights a page's notes are on go into the capture store, which is where a note's link
 * and label look them up. A note added, edited or removed anywhere moves the store's `version`,
 * and the rows already shown are fetched again.
 */
export function useNotesPage(
  search: Ref<string>,
  kinds: Ref<string[]>,
  {
    pageSize = PAGE,
    words = false,
    enabled = () => true,
    debounceMs = SEARCH_DEBOUNCE_MS,
  }: { pageSize?: number; words?: boolean; enabled?: () => boolean; debounceMs?: number } = {},
) {
  const capture = useCaptureStore()
  const items = ref<Note[]>([])
  const total = ref(0)
  const counts = ref<Record<string, number>>({})
  const loading = ref(false)
  const error = ref(false)
  let wanted = pageSize
  let seq = 0

  async function fetch(offset: number, limit: number, append: boolean): Promise<void> {
    const mine = ++seq
    if (!enabled()) {
      // Nothing asked for (Search before it has run): nothing shown, nothing fetched.
      items.value = []
      total.value = 0
      return
    }
    loading.value = true
    try {
      const q = search.value.trim()
      const [page, all] = await Promise.all([
        getNotesPage({ q, kinds: kinds.value, words, offset, limit: Math.min(MAX, limit) }),
        q ? getNotesPage({ limit: 1 }) : null,
      ])
      if (mine !== seq) return
      capture.merge(page.highlights ?? [])
      items.value = append ? [...items.value, ...page.items] : page.items
      total.value = page.total
      counts.value = (all ?? page).counts ?? {}
      error.value = false
    } catch {
      if (mine === seq) error.value = true
    } finally {
      if (mine === seq) loading.value = false
    }
  }

  const reload = (): Promise<void> => fetch(0, wanted, false)

  function toggle(): void {
    if (total.value > items.value.length) {
      wanted = items.value.length + pageSize
      void fetch(items.value.length, pageSize, true)
    } else {
      wanted = pageSize
      items.value = items.value.slice(0, pageSize)
    }
  }

  let debounce: ReturnType<typeof setTimeout> | null = null
  watch(search, () => {
    if (debounce) clearTimeout(debounce)
    const run = () => {
      wanted = pageSize
      void reload()
    }
    // A search box waits for typing to pause; a search that has already RUN does not.
    if (debounceMs > 0) debounce = setTimeout(run, debounceMs)
    else run()
  })
  watch(kinds, () => {
    wanted = pageSize
    void reload()
  })
  watch(
    () => capture.version,
    () => void reload(),
  )

  return {
    items,
    total,
    counts,
    loading,
    error,
    /** Whether the user has any note at all (the chips' counts are unfiltered). */
    anyNotes: computed(() => Object.values(counts.value).some((n) => n > 0)),
    remaining: computed(() => Math.max(0, total.value - items.value.length)),
    reload,
    toggle,
  }
}
