import { computed, ref, watch, type Ref } from 'vue'
import { getHighlightsPage } from '../services/api'
import { useCaptureStore } from '../stores/capture'
import type { Highlight } from '../services/types'

export interface HighlightFilters {
  search: Ref<string>
  color: Ref<string | null>
  sort: Ref<string>
  mutedOnly: Ref<boolean>
}

const PAGE = 5
const MAX = 100
const SEARCH_DEBOUNCE_MS = 250

interface GroupState {
  slug: string
  /** Highlight ids in display order (newest first) — resolved through the store when rendered. */
  ids: string[]
  /** How many highlights in this episode match the filters (the server's count). */
  total: number
}

/**
 * Saved's Highlights, a page of EPISODES at a time from the server (`getHighlightsPage`).
 *
 * Five episodes, then five more; within an episode five highlights, then five more — the same
 * rhythm as before, but each "more" is a request for exactly those rows instead of a slice of a
 * library loaded whole. Rows go into the capture store (`merge`) and are rendered THROUGH it, so a
 * recolour, an unsave or a new note shows at once; the store's `version` then refetches the rows
 * already shown, so the counts follow.
 */
export function useHighlightsPage(filters: HighlightFilters) {
  const capture = useCaptureStore()
  const groupState = ref<GroupState[]>([])
  const total = ref(0)
  const episodeTotal = ref(0)
  const loading = ref(false)
  const error = ref(false)
  /** Episodes wanted on screen, and highlights wanted per episode (beyond the first five). */
  let wantedGroups = PAGE
  const wantedIn = new Map<string, number>()
  let seq = 0

  const query = () => ({
    q: filters.search.value,
    color: filters.color.value,
    muted: filters.mutedOnly.value,
    sort: (filters.sort.value === 'title' ? 'title' : 'recent') as 'title' | 'recent',
  })

  function groupsFrom(items: Highlight[], counts: Record<string, number>): GroupState[] {
    const out: GroupState[] = []
    const at = new Map<string, GroupState>()
    for (const h of items) {
      let g = at.get(h.episode_slug)
      if (!g) {
        g = { slug: h.episode_slug, ids: [], total: counts[h.episode_slug] ?? 0 }
        at.set(h.episode_slug, g)
        out.push(g)
      }
      g.ids.push(h.id)
    }
    return out
  }

  async function reload(): Promise<void> {
    const mine = ++seq
    loading.value = true
    try {
      const perEpisode = Math.min(MAX, Math.max(PAGE, ...wantedIn.values()))
      const page = await getHighlightsPage({ ...query(), offset: 0, limit: Math.min(MAX, wantedGroups), perEpisode })
      if (mine !== seq) return
      capture.merge(page.items, page.notes)
      // Each episode shows only what was asked of IT, even if a sibling asked for more.
      groupState.value = groupsFrom(page.items, page.episode_counts).map((g) => ({
        ...g,
        ids: g.ids.slice(0, wantedIn.get(g.slug) ?? PAGE),
      }))
      total.value = page.total
      episodeTotal.value = page.episode_total
      error.value = false
    } catch {
      if (mine === seq) error.value = true
    } finally {
      if (mine === seq) loading.value = false
    }
  }

  let moreInFlight = false
  async function showMoreGroups(): Promise<void> {
    // A tap while the NEXT PAGE is on its way would supersede it (and a run of taps, every one).
    // Only a "more" blocks a "more": a background revalidation must not swallow the tap.
    if (moreInFlight) return
    moreInFlight = true
    const mine = ++seq
    const offset = groupState.value.length
    wantedGroups = offset + PAGE
    try {
      const page = await getHighlightsPage({ ...query(), offset, limit: PAGE, perEpisode: PAGE })
      if (mine !== seq) return
      capture.merge(page.items, page.notes)
      groupState.value = [...groupState.value, ...groupsFrom(page.items, page.episode_counts)]
    } catch {
      if (mine === seq) error.value = true
    } finally {
      moreInFlight = false
    }
  }

  function toggleGroups(): void {
    if (episodeTotal.value > groupState.value.length) void showMoreGroups()
    else {
      wantedGroups = PAGE
      groupState.value = groupState.value.slice(0, PAGE)
    }
  }

  /** The next five highlights of one episode (or back to five once all are out). */
  const busyIn = new Set<string>()
  async function toggleIn(slug: string): Promise<void> {
    const g = groupState.value.find((x) => x.slug === slug)
    if (!g || busyIn.has(slug)) return
    if (g.ids.length >= g.total) {
      wantedIn.delete(slug)
      g.ids = g.ids.slice(0, PAGE)
      return
    }
    const want = Math.min(MAX, g.ids.length + PAGE)
    wantedIn.set(slug, want)
    busyIn.add(slug)
    try {
      const page = await getHighlightsPage({ ...query(), episode: slug, offset: 0, limit: 1, perEpisode: want })
      capture.merge(page.items, page.notes)
      g.ids = page.items.map((h) => h.id)
      g.total = page.episode_counts[slug] ?? g.total
    } catch {
      error.value = true
    } finally {
      busyIn.delete(slug)
    }
  }

  /** The groups as rendered: each id resolved through the store (its edits show at once). */
  const groups = computed(() => {
    const byId = new Map(capture.highlights.map((h) => [h.id, h]))
    return groupState.value
      .map((g) => ({
        slug: g.slug,
        total: g.total,
        highlights: g.ids.map((id) => byId.get(id)).filter((h): h is Highlight => !!h),
      }))
      .filter((g) => g.highlights.length)
  })

  let debounce: ReturnType<typeof setTimeout> | null = null
  watch(filters.search, () => {
    if (debounce) clearTimeout(debounce)
    debounce = setTimeout(() => {
      wantedGroups = PAGE
      wantedIn.clear()
      void reload()
    }, SEARCH_DEBOUNCE_MS)
  })
  watch([filters.color, filters.sort, filters.mutedOnly], () => {
    wantedGroups = PAGE
    wantedIn.clear()
    void reload()
  })
  watch(
    () => capture.version,
    () => void reload(),
  )

  return {
    groups,
    total,
    episodeTotal,
    loading,
    error,
    remainingGroups: computed(() => Math.max(0, episodeTotal.value - groupState.value.length)),
    reload,
    toggleGroups,
    toggleIn,
  }
}

export type HighlightsPageState = ReturnType<typeof useHighlightsPage>
