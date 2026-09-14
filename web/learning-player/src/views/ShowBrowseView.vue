<script setup lang="ts">
/**
 * Show browse index — every show in the corpus as a square-artwork grid (ShowTile), so you can page
 * through the catalogue by show, not just by episode. A Browse-hub tab (embedded) and a standalone
 * route reached from Home/Library. Tiles are followable so you can follow while browsing.
 */
import { computed, onMounted, ref, watch } from "vue"
import { useI18n } from "vue-i18n"
import { RouterLink } from "vue-router"
import ShowTile from "../components/ShowTile.vue"
import SectionStatus from "../components/SectionStatus.vue"
import ListToolbar from "../components/ListToolbar.vue"
import Sparkline from "../components/Sparkline.vue"
import { trendColor } from "../components/trending"
import { listSortOptions, type ListSortValue } from "../utils/listSort"
import { getPodcasts, getTrending } from "../services/api"
import { isArrayCache, readCached, writeCached } from "../services/contentCache"
import { showArtwork } from "../utils/episode"
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
// chunks of 10 with a "Show more" (operator 2026-09-14). The standalone page shows everything.
const CHUNK = 10

const { t } = useI18n()

const shows = ref<Podcast[]>([])
const trendById = ref<Map<string, TrendingEntity>>(new Map())
const loading = ref(true)
const error = ref(false)
const stale = ref(false)

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
const categories = computed(() =>
  [...new Set(shows.value.map((s) => s.category).filter((c): c is string => !!c))].sort((a, b) =>
    a.localeCompare(b)
  )
)
// The shared four-way sort (Newest / Oldest / A–Z / Z–A) plus a Shows-only "Trending" (velocity).
const sortOptions = computed(() => [
  ...listSortOptions(t),
  { value: "trending" as const, label: t("list.sortTrending") },
])
const categoryOptions = computed(() => [
  { value: "", label: t("browse.allCategories") },
  ...categories.value.map((c) => ({ value: c, label: c })),
])

const visible = computed(() => {
  let list = shows.value.filter((s) => s.feed_id)
  const q = search.value.trim().toLowerCase()
  if (q) list = list.filter((s) => titleOf(s).toLowerCase().includes(q))
  if (categoryFilter.value) list = list.filter((s) => s.category === categoryFilter.value)
  const byTitle = (a: Podcast, b: Podcast) => titleOf(a).localeCompare(titleOf(b))
  // Newest/oldest sort by the feed's last_updated; a feed with none sorts last, then by title so
  // the order is stable (a corpus with no dates reads alphabetically rather than at random).
  const byDate = (s: Podcast) => s.last_updated ?? ""
  const velocityOf = (s: Podcast) => trendById.value.get(s.feed_id)?.velocity ?? -Infinity
  return [...list].sort((a, b) => {
    // Trending: highest momentum velocity first; shows with no trend series fall to the end by title.
    if (sort.value === "trending") return velocityOf(b) - velocityOf(a) || byTitle(a, b)
    if (sort.value === "az") return byTitle(a, b)
    if (sort.value === "za") return byTitle(b, a)
    const cmp =
      sort.value === "newest"
        ? byDate(b).localeCompare(byDate(a))
        : byDate(a).localeCompare(byDate(b))
    return cmp || byTitle(a, b)
  })
})

// Reveal `visible` in chunks when embedded; standalone shows all. Reset to one chunk whenever the
// filtered set changes so a narrowed search never hides behind a "Show more" from a prior query.
const shownCount = ref(CHUNK)
const capped = computed(() =>
  props.embedded ? visible.value.slice(0, shownCount.value) : visible.value
)
const hasMore = computed(() => capped.value.length < visible.value.length)
watch([search, sort, categoryFilter], () => {
  shownCount.value = CHUNK
})

async function load(): Promise<void> {
  loading.value = true
  error.value = false
  try {
    const [rows, trend] = await Promise.all([
      getPodcasts(),
      // Best-effort: the committed corpus ships no velocity, so this is often empty — the Trending
      // sort then just reads alphabetically and no sparklines render, which is honest.
      getTrending("show", "corpus", 50).catch(() => [] as TrendingEntity[]),
    ])
    shows.value = rows
    trendById.value = new Map(trend.map((e) => [e.entity_id, e]))
    void writeCached("browse.shows", rows)
  } catch {
    // The show list we last saw beats "couldn't load" — this one at least reported the failure
    // rather than pretending the corpus was empty, but it still had nothing to show (#1909).
    const cached = await readCached<typeof shows.value>("browse.shows", isArrayCache)
    if (cached?.length) {
      shows.value = cached
      stale.value = true
    } else {
      error.value = true
    }
  } finally {
    loading.value = false
  }
}
onMounted(load)
</script>

<template>
  <section
    :class="embedded ? '' : 'mx-auto max-w-3xl px-4 pb-8 pt-4'"
    data-testid="show-browse-view"
  >
    <RouterLink
      v-if="!embedded"
      :to="{ name: 'home' }"
      class="lp-nav mb-4"
      data-testid="browse-back-home"
    >
      ‹ {{ t("browse.backHome") }}
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
    <template v-else-if="shows.length">
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
        <!-- List view: title-first rows, denser than the tile grid; tap opens the show. -->
        <ul v-else class="flex flex-col" data-testid="show-browse-list">
          <li v-for="p in capped" :key="p.feed_id">
            <RouterLink
              :to="{ name: 'podcast', params: { feedId: p.feed_id } }"
              class="flex items-center gap-3 border-b border-border py-2 no-underline text-canvas-foreground hover:bg-overlay"
            >
              <img
                v-if="showArtwork(p)"
                :src="showArtwork(p)!"
                alt=""
                loading="lazy"
                class="h-11 w-11 shrink-0 rounded-lg bg-elevated object-cover"
              />
              <div v-else class="h-11 w-11 shrink-0 rounded-lg bg-elevated" />
              <span class="min-w-0 flex-1">
                <span class="block truncate text-sm font-semibold">{{ titleOf(p) }}</span>
                <span class="lp-kicker block">{{
                  t("podcast.episodeCount", { count: p.episode_count }, p.episode_count)
                }}</span>
              </span>
              <!-- Trend sparkline blended in when sorting by Trending (operator 2026-09-14): how the
                   show's momentum has moved, hued by velocity. Absent when the corpus has no series. -->
              <Sparkline
                v-if="sort === 'trending' && trendById.get(p.feed_id)"
                :values="trendById.get(p.feed_id)!.series"
                :width="56"
                :height="16"
                :stroke-width="1.4"
                class="shrink-0"
                :style="{ color: trendColor(trendById.get(p.feed_id)!.velocity) }"
              />
              <span class="shrink-0 text-muted" aria-hidden="true">›</span>
            </RouterLink>
          </li>
        </ul>
        <button
          v-if="hasMore"
          type="button"
          class="mt-4 w-full rounded-xl border border-border py-2.5 text-sm font-bold text-accent transition hover:bg-overlay"
          data-testid="show-browse-more"
          @click="shownCount += CHUNK"
        >
          {{ t("catalog.loadMore") }}
        </button>
      </template>
      <p v-else class="text-muted">{{ t("browse.noShowMatches") }}</p>
    </template>
    <p v-else class="text-muted">{{ t("browse.empty") }}</p>
  </section>
</template>
