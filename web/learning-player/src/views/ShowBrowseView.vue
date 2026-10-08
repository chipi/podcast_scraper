<script setup lang="ts">
/**
 * Show browse index — every show in the corpus as a square-artwork grid (ShowTile), so you can page
 * through the catalogue by show, not just by episode. A Browse-hub tab (embedded) and a standalone
 * route reached from Home/Library. Tiles are followable so you can follow while browsing.
 */
import { computed, onMounted, ref, watch } from "vue"
import { useI18n } from "vue-i18n"
import { RouterLink } from "vue-router"
import BackIcon from "../components/BackIcon.vue"
import ShowTile from "../components/ShowTile.vue"
import ShowRow from "../components/ShowRow.vue"
import SectionStatus from "../components/SectionStatus.vue"
import ListToolbar from "../components/ListToolbar.vue"
import Sparkline from "../components/Sparkline.vue"
import FollowButton from "../components/FollowButton.vue"
import FavoriteButton from "../components/FavoriteButton.vue"
import { trendColor } from "../components/trending"
import { listSortOptions, type ListSortValue } from "../utils/listSort"
import { getPodcastsPage, getTrending } from "../services/api"
import { isArrayCache, readCached, writeCached } from "../services/contentCache"
import { useLibraryStore } from "../stores/library"
import { useSignInGate } from "../composables/useSignInGate"
import type { Podcast, TrendingEntity } from "../services/types"

// Shows adds one sort the shared list util does not carry (operator 2026-09-14): "Trending" orders
// by the show's momentum velocity (RFC-103, GET /api/app/trending?kind=show) and blends a sparkline
// into each row so you can see HOW it's trending. Episodes keeps the four shared sorts only.
type ShowSort = ListSortValue | "trending"

// `embedded` — rendered as a tab panel inside the Browse hub (drops heading/back-Home/padding).
// `initialSort` — Discover's "See all →" on the trending-shows rail lands here pre-sorted by Trending
// (operator 2026-09-14); a later prop change re-applies it, since this view is kept-alive.
const props = withDefaults(
  defineProps<{ embedded?: boolean; initialSort?: ShowSort; initialView?: "grid" | "list" }>(),
  { embedded: false, initialSort: undefined, initialView: undefined }
)

// Embedded in the Discover band, the grid is a dispatch surface, not the full index: reveal in
// chunks of 10 with a "Show more" (operator 2026-09-14). The standalone page shows everything (a
// page of 200, the server's cap). Since 2026-10-08 each chunk is a SERVER page under the bar's
// search, sort and category, instead of the whole catalogue filtered here.
const CHUNK = 10
const PAGE = computed(() => (props.embedded ? CHUNK : 200))

const { t } = useI18n()

/** The shows loaded so far under the current search / sort / category. */
const shows = ref<Podcast[]>([])
/** How many match, across all pages (the server's count). */
const total = ref(0)
/** Whether the catalogue has any show at all — the bar stays even when a search matches nothing. */
const catalogueSize = ref(0)
/** Every category in the catalogue (the filter's options). */
const allCategories = ref<string[]>([])
const trendById = ref<Map<string, TrendingEntity>>(new Map())
const loading = ref(true)
const error = ref(false)
const stale = ref(false)

// Follow from a LIST row (operator 2026-09-17). The grid had Follow via ShowTile and the list had
// nothing, so the same catalogue exposed different capabilities depending on which view you were in
// — the rule the episode list/grid pair already holds to. The heart needs no wiring here; the
// favourites store backs it directly.
const library = useLibraryStore()
const { isGated, gated } = useSignInGate()
// Which row is mid-toggle, so only that row's pill disables rather than all of them.
const busyFollow = ref<string | null>(null)

// Gated for the same reason ShowTile gates: the store reverts optimistically on failure, so an
// ungated signed-out click flips the pill, fires a 401 and flips back.
//
// `gated()` wraps a ZERO-argument action, so the row is closed over per call rather than passed
// through it.
function toggleFollow(p: Podcast): void {
  gated(async () => {
    busyFollow.value = p.feed_id
    try {
      await library.toggle(p.feed_id, { title: p.title ?? p.feed_id })
    } finally {
      busyFollow.value = null
    }
  })()
}

// Filter + sort so the grid stays browsable as the catalogue grows.
const search = ref("")
const sort = ref<ShowSort>(props.initialSort ?? "newest")
// Kept-alive: a later "See all →" click changes the prop after setup ran, so re-apply it.
watch(() => props.initialSort, (v) => { if (v) sort.value = v })
// Grid ⇄ list view toggle (BS.2) — the same affordance Browse › Episodes carries. Grid is
// artwork-first; list is title-first + denser for scanning a long catalogue.
const view = ref<"grid" | "list">(props.initialView ?? "grid")
// See-all from the trending rail lands in list view so the per-row sparklines show (operator).
watch(() => props.initialView, (v) => { if (v) view.value = v })
const titleOf = (s: Podcast) => s.title ?? s.feed_id

// Category facet (BS.1) — the distinct categories present in the catalogue, alphabetized. The
// picker only renders when at least one show carries a category, so a corpus without any is unchanged.
const categoryFilter = ref<string>("")
const categories = computed(() => allCategories.value)
// The shared four-way sort (Newest / Oldest / A–Z / Z–A) plus a Shows-only "Trending" (velocity).
const sortOptions = computed(() => [
  ...listSortOptions(t),
  { value: "trending" as const, label: t("list.sortTrending") },
])
const categoryOptions = computed(() => [
  { value: "", label: t("browse.allCategories") },
  ...categories.value.map((c) => ({ value: c, label: c })),
])

/** The rows on screen — exactly what the server returned, in its order. */
const visible = computed(() => shows.value)
const capped = visible
const hasMore = computed(() => shows.value.length < total.value)

function pageQuery(offset: number, limit: number) {
  return {
    q: search.value,
    category: categoryFilter.value || undefined,
    sort: sort.value,
    offset,
    limit,
  }
}

/** The next chunk from the server. */
async function showMore(): Promise<void> {
  if (moreLoading.value) return
  moreLoading.value = true
  try {
    const page = await getPodcastsPage(pageQuery(shows.value.length, PAGE.value))
    shows.value = [...shows.value, ...page.items]
    total.value = page.total
  } finally {
    moreLoading.value = false
  }
}
const moreLoading = ref(false)

let debounce: ReturnType<typeof setTimeout> | null = null
watch(search, () => {
  if (debounce) clearTimeout(debounce)
  debounce = setTimeout(() => void load(), 250)
})
watch([sort, categoryFilter], () => void load())

let seq = 0
async function load(): Promise<void> {
  const mine = ++seq
  // A re-query keeps the rows on screen until the answer lands (no skeleton flash per keystroke).
  if (!shows.value.length) loading.value = true
  error.value = false
  try {
    const [page, trend] = await Promise.all([
      getPodcastsPage(pageQuery(0, PAGE.value)),
      // Best-effort: the committed corpus ships no velocity, so this is often empty — the Trending
      // sort then just reads alphabetically and no sparklines render, which is honest.
      getTrending("show", "corpus", 50).catch(() => [] as TrendingEntity[]),
    ])
    if (mine !== seq) return
    shows.value = page.items
    total.value = page.total
    allCategories.value = page.categories
    if (!search.value.trim() && !categoryFilter.value) catalogueSize.value = page.total
    else catalogueSize.value = Math.max(catalogueSize.value, page.total)
    stale.value = false
    trendById.value = new Map(trend.map((e) => [e.entity_id, e]))
    void writeCached("browse.shows", page.items)
  } catch {
    // The show list we last saw beats "couldn't load" — this one at least reported the failure
    // rather than pretending the corpus was empty, but it still had nothing to show (#1909).
    const cached = await readCached<typeof shows.value>("browse.shows", isArrayCache)
    if (mine !== seq) return
    if (cached?.length) {
      shows.value = cached
      total.value = cached.length
      catalogueSize.value = Math.max(catalogueSize.value, cached.length)
      stale.value = true
    } else {
      error.value = true
    }
  } finally {
    if (mine === seq) loading.value = false
  }
}
onMounted(load)
</script>

<template>
  <section
    :class="embedded ? '' : 'lp-page pb-8 pt-4'"
    data-testid="show-browse-view"
  >
    <RouterLink
      v-if="!embedded"
      :to="{ name: 'home' }"
      class="lp-nav mb-4"
      data-testid="browse-back-home"
    >
      <BackIcon /> {{ t("browse.backHome") }}
    </RouterLink>
    <h1 v-if="!embedded" class="mb-4 font-display text-3xl font-extrabold tracking-tight">
      {{ t("browse.shows") }}
    </h1>
    <!-- Standalone, never chained into a neighbouring v-if/v-else: slotting a notice into
         such a chain once made the final v-else (the content) unreachable. -->
    <p v-if="stale" class="mb-3 text-sm text-muted" data-testid="browse-stale-shows">
      {{ t("browse.stale") }}
    </p>

    <!-- F1.3/F1.4: reserve the grid shape while loading; retry on a no-cache failure. -->
    <SectionStatus
      v-if="loading || error"
      :phase="loading ? 'loading' : 'error'"
      :rows="6"
      @retry="load"
    />
    <template v-else-if="catalogueSize">
      <!-- The SAME shared bar as Browse › Episodes (operator 2026-09-14): search + sort + category
           facet + view toggle, one compact row. The category facet (BS.1) is passed as the optional
           filter only when the catalogue carries categories. -->
      <ListToolbar
        v-model:search="search"
        v-model:sort="sort"
        v-model:filter="categoryFilter"
        v-model:view="view"
        :sort-options="sortOptions"
        :filter-options="categories.length ? categoryOptions : undefined"
        :search-placeholder="t('browse.filterShows')"
        search-testid="show-browse-search"
        sort-testid="show-browse-sort"
        filter-testid="show-browse-category"
        view-testid="show-view"
      />
      <template v-if="visible.length">
        <ul
          v-if="view === 'grid'"
          class="grid grid-cols-3 gap-3 sm:grid-cols-4"
          data-testid="show-browse-grid"
        >
          <li v-for="p in capped" :key="p.feed_id"><ShowTile :show="p" followable /></li>
        </ul>
        <!-- List view: the SHARED ShowRow — 128px artwork and the description beside it, built to
             the same proportions as the episode list (operator 2026-09-17). It was a 44px thumbnail
             with a title and a count, which read as a different kind of thing from the episode rows
             one tab across. Library's Saved tab renders the identical component. -->
        <ul v-else class="flex flex-col" data-testid="show-browse-list">
          <li v-for="p in capped" :key="p.feed_id">
            <!-- Controls UNDER the artwork, where the Episodes tab's cards put theirs (operator
                 2026-10-05): one page, one place for controls. Follow is the plain inline pill
                 there, not the plated overlay one — it sits on the page, not on a picture. -->
            <ShowRow :show="p" actions-below>
              <template #actions>
                <FollowButton
                  variant="inline"
                  :following="library.has(p.feed_id)"
                  :busy="busyFollow === p.feed_id"
                  :gated="isGated"
                  @toggle="toggleFollow(p)"
                />
                <FavoriteButton :item="{ kind: 'show', ref: p.feed_id, label: titleOf(p) }" />
              </template>
              <!-- Trend sparkline under the row's text, only when sorting by Trending (operator
                   2026-09-14): how the show's momentum has moved, hued by velocity. It does NOT
                   belong in the overlay column — it is information, not a control. Absent when the
                   corpus carries no series. -->
              <template v-if="sort === 'trending' && trendById.get(p.feed_id)" #meta>
                <Sparkline
                  :values="trendById.get(p.feed_id)!.series"
                  :width="56"
                  :height="16"
                  :stroke-width="1.4"
                  :style="{ color: trendColor(trendById.get(p.feed_id)!.velocity) }"
                />
              </template>
            </ShowRow>
          </li>
        </ul>
        <button
          v-if="hasMore"
          type="button"
          class="mt-4 w-full rounded-xl border border-border py-2.5 text-sm font-bold text-accent transition hover:bg-overlay"
          data-testid="show-browse-more"
          @click="showMore"
        >
          {{ t("catalog.loadMore") }}
        </button>
      </template>
      <p v-else class="text-muted">{{ t("browse.noShowMatches") }}</p>
    </template>
    <p v-else class="text-muted">{{ t("browse.empty") }}</p>
  </section>
</template>
