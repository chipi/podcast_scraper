<script setup lang="ts">
/**
 * Show browse index — every show in the corpus as a square-artwork grid (ShowTile), so you can page
 * through the catalogue by show, not just by episode. A Browse-hub tab (embedded) and a standalone
 * route reached from Home/Library. Tiles are followable so you can follow while browsing.
 */
import { computed, onMounted, ref } from "vue"
import { useI18n } from "vue-i18n"
import { RouterLink } from "vue-router"
import ShowTile from "../components/ShowTile.vue"
import SectionStatus from "../components/SectionStatus.vue"
import ViewToggle from "../components/ViewToggle.vue"
import { getPodcasts } from "../services/api"
import { isArrayCache, readCached, writeCached } from "../services/contentCache"
import { showArtwork } from "../utils/episode"
import type { Podcast } from "../services/types"

// `embedded` — rendered as a tab panel inside the Browse hub (drops heading/back-Home/padding).
withDefaults(defineProps<{ embedded?: boolean }>(), { embedded: false })

const { t } = useI18n()

const shows = ref<Podcast[]>([])
const loading = ref(true)
const error = ref(false)
const stale = ref(false)

// Filter + sort so the grid stays browsable as the catalogue grows.
const search = ref("")
const sort = ref<"az" | "episodes">("az")
// Grid ⇄ list view toggle (BS.2) — the same affordance Browse › Episodes carries. Grid is
// artwork-first; list is title-first + denser for scanning a long catalogue.
const view = ref<"grid" | "list">("grid")
const titleOf = (s: Podcast) => s.title ?? s.feed_id

// Category facet (BS.1) — the distinct categories present in the catalogue, alphabetized. The
// picker only renders when at least one show carries a category, so a corpus without any is unchanged.
const categoryFilter = ref<string>("")
const categories = computed(() =>
  [...new Set(shows.value.map((s) => s.category).filter((c): c is string => !!c))].sort((a, b) =>
    a.localeCompare(b)
  )
)

const visible = computed(() => {
  let list = shows.value.filter((s) => s.feed_id)
  const q = search.value.trim().toLowerCase()
  if (q) list = list.filter((s) => titleOf(s).toLowerCase().includes(q))
  if (categoryFilter.value) list = list.filter((s) => s.category === categoryFilter.value)
  return [...list].sort((a, b) =>
    sort.value === "episodes"
      ? b.episode_count - a.episode_count || titleOf(a).localeCompare(titleOf(b))
      : titleOf(a).localeCompare(titleOf(b))
  )
})

async function load(): Promise<void> {
  loading.value = true
  error.value = false
  try {
    const rows = await getPodcasts()
    shows.value = rows
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
      <!-- Filter + sort — shows grow, so keep the grid searchable + orderable. -->
      <div class="mb-4 flex flex-wrap items-center gap-2">
        <input
          v-model="search"
          type="search"
          :placeholder="t('browse.filterShows')"
          class="lp-search min-w-0 flex-1 rounded-full border border-border bg-surface px-4 py-2 text-sm text-canvas-foreground outline-none focus:border-accent"
          data-testid="show-browse-search"
        />
        <select
          v-model="sort"
          class="shrink-0 rounded-full border border-border bg-surface px-3 py-2 text-sm font-semibold text-canvas-foreground outline-none focus:border-accent"
          data-testid="show-browse-sort"
        >
          <option value="az">{{ t("browse.sortShowsAZ") }}</option>
          <option value="episodes">{{ t("browse.sortShowsEpisodes") }}</option>
        </select>
        <!-- Category facet (BS.1) — only when the catalogue carries any categories. -->
        <select
          v-if="categories.length"
          v-model="categoryFilter"
          class="shrink-0 rounded-full border border-border bg-surface px-3 py-2 text-sm font-semibold text-canvas-foreground outline-none focus:border-accent"
          data-testid="show-browse-category"
          :aria-label="t('browse.categoryFilter')"
        >
          <option value="">{{ t("browse.allCategories") }}</option>
          <option v-for="c in categories" :key="c" :value="c">{{ c }}</option>
        </select>
        <!-- Grid ⇄ list view toggle (BS.2), same control as Browse › Episodes. -->
        <ViewToggle v-model="view" testid-grid="show-view-grid" testid-list="show-view-list" />
      </div>
      <template v-if="visible.length">
        <ul
          v-if="view === 'grid'"
          class="grid grid-cols-3 gap-3 sm:grid-cols-4"
          data-testid="show-browse-grid"
        >
          <li v-for="p in visible" :key="p.feed_id"><ShowTile :show="p" followable /></li>
        </ul>
        <!-- List view: title-first rows, denser than the tile grid; tap opens the show. -->
        <ul v-else class="flex flex-col" data-testid="show-browse-list">
          <li v-for="p in visible" :key="p.feed_id">
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
              <span class="shrink-0 text-muted" aria-hidden="true">›</span>
            </RouterLink>
          </li>
        </ul>
      </template>
      <p v-else class="text-muted">{{ t("browse.noShowMatches") }}</p>
    </template>
    <p v-else class="text-muted">{{ t("browse.empty") }}</p>
  </section>
</template>
