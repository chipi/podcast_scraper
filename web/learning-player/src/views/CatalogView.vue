<script setup lang="ts">
/**
 * Catalog — global all-episodes view (PRD-038 FR1). Newest-first, paginated with a "load more"
 * control (20/page). A shared ListToolbar (UXS-014) filters/sorts/searches the list; engaging any
 * control auto-loads the remaining pages so the controls cover the whole catalog, not just what's
 * been paged in.
 */
import { computed, onActivated, onDeactivated, onMounted, ref, watch } from "vue"
import { useI18n } from "vue-i18n"
defineOptions({ name: "CatalogView" }) // stable name for <keep-alive :include> (App.vue)
import EpisodeCard from "../components/EpisodeCard.vue"
import EpisodeTile from "../components/EpisodeTile.vue"
import ListToolbar from "../components/ListToolbar.vue"
import SectionStatus from "../components/SectionStatus.vue"
import { getInProgressSlugs, getPodcastsPage, getWorldEpisodeSlugs, listEpisodes } from "../services/api"
import { useRoute } from "vue-router"
import Tabs from "../components/Tabs.vue"
import { useLibraryStore } from "../stores/library"
import { isArrayCache, readCached, writeCached } from "../services/contentCache"
import { useCompletedStore } from "../stores/completed"
import { usePlayed } from "../composables/usePlayed"
import { useDownloadsStore } from "../stores/downloads"
import { useAuthStore } from "../stores/auth"
import { isNative } from "../services/native"
import { listSortOptions } from "../utils/listSort"
import type { EpisodeSummary } from "../services/types"

// `embedded` — rendered as the Episodes tab panel inside the Browse hub, which supplies the page
// heading; drop our own so it isn't shown twice.
const props = withDefaults(defineProps<{ embedded?: boolean }>(), { embedded: false })

// Embedded in the Discover band the list is a dispatch surface, so it reveals in a small chunk with
// a "Load more" (operator 2026-09-14); the standalone catalog keeps the larger page (20/page).
const PAGE_SIZE = props.embedded ? 10 : 20
const { t } = useI18n()
const episodes = ref<EpisodeSummary[]>([])
const page = ref(0)
const hasMore = ref(false)
const loading = ref(false)
const error = ref(false)

const completed = useCompletedStore()
const { isPlayed } = usePlayed()
const downloads = useDownloadsStore()
const auth = useAuthStore()

const route = useRoute()
const library = useLibraryStore()

const search = ref("")
const sort = ref("newest")
const filter = ref("all")
/**
 * WHICH episodes, independent of their state (operator 2026-10-09): All, Shows I follow, or Mine
 * — the listener's world, ADR-162. A second row rather than more entries in the state filter, so
 * "Shows I follow" combines with "Unplayed". Deep-linkable: `?from=` and `?state=` (What's new and
 * Discover 2 open the list already filtered).
 */
type From = "all" | "following" | "mine"
const from = ref<From>("all")
const fromTabs = computed(() => [
  { key: "all" as const, label: t("browse.fromAll"), testid: "catalog-from-all" },
  { key: "following" as const, label: t("browse.fromFollowing"), testid: "catalog-from-following" },
  { key: "mine" as const, label: t("browse.fromMine"), testid: "catalog-from-mine" },
])
// Fetched only when a filter needs them.
const worldSlugs = ref<Set<string> | null>(null)
const inProgressSlugs = ref<Set<string> | null>(null)
watch(
  from,
  async (f) => {
    if (f === "following") void library.ensureLoaded().catch(() => {})
    if (f === "mine" && !worldSlugs.value) {
      worldSlugs.value = new Set(await getWorldEpisodeSlugs().catch(() => [] as string[]))
    }
  },
  { immediate: true },
)
watch(
  filter,
  async (f) => {
    if (f === "inprogress" && !inProgressSlugs.value) {
      inProgressSlugs.value = new Set(await getInProgressSlugs().catch(() => [] as string[]))
    }
  },
  { immediate: true },
)
// Kept alive: a follow, save or listen elsewhere changes both sets, so a return re-reads the ones
// in use and drops the rest (re-read on next use). Skips the mount's own activation.
let leftOnce = false
onDeactivated(() => {
  leftOnce = true
})
onActivated(async () => {
  if (!leftOnce || !auth.isAuthenticated) return
  worldSlugs.value = null
  inProgressSlugs.value = null
  if (from.value === "mine") {
    worldSlugs.value = new Set(await getWorldEpisodeSlugs().catch(() => [] as string[]))
  }
  if (filter.value === "inprogress") {
    inProgressSlugs.value = new Set(await getInProgressSlugs().catch(() => [] as string[]))
  }
})
const FROM_VALUES: From[] = ["all", "following", "mine"]
function applyQuery(): void {
  const f = String(route.query.from ?? "")
  if ((FROM_VALUES as string[]).includes(f)) from.value = f as From
  const st = String(route.query.state ?? "")
  if (st) filter.value = st
}
applyQuery()
watch(() => [route.query.from, route.query.state], applyQuery)
const show = ref("")
// List (banded rows) vs grid (tiles) — BE.5. Grid is flat (time bands are a list-only device).
const view = ref<"list" | "grid">("list")

// Filter options for the toolbar (BE.6/BE.7). Downloaded is native-only (nothing downloads on web).
const filterOptions = computed(() => {
  const opts = [
    { value: "all", label: t("list.filterAll") },
    { value: "unplayed", label: t("list.filterUnplayed") },
    ...(auth.isAuthenticated ? [{ value: "inprogress", label: t("list.filterInProgress") }] : []),
    { value: "played", label: t("list.filterPlayed") },
    { value: "insights", label: t("list.filterInsights") },
  ]
  if (isNative()) opts.splice(3, 0, { value: "downloaded", label: t("list.filterDownloaded") })
  return opts
})
// The shared four-way sort (Newest / Oldest / A–Z / Z–A) — identical to Browse › Shows.
const sortOptions = computed(() => listSortOptions(t))
const shows = ref<{ id: string; label: string }[]>([])
const controlsActive = computed(
  () =>
    search.value.trim() !== "" ||
    sort.value !== "newest" ||
    filter.value !== "all" ||
    from.value !== "all" ||
    show.value !== ""
)

/**
 * Browse offline showed a bare red "Couldn't load episodes." on an empty page — no retry, no
 * content, on a device that had rendered this exact list minutes earlier (#1909).
 *
 * Only the FIRST page is snapshotted. Browsing deep into a corpus is not something to promise
 * offline, and the operator did not ask for it — but the page you land on should be the page you
 * last saw, not a red sentence.
 */
const BROWSE_CACHE_KEY = "browse.episodes"
const stale = ref(false)

async function loadMore(): Promise<void> {
  loading.value = true
  error.value = false
  try {
    const next = page.value + 1
    const res = await listEpisodes({ page: next, pageSize: PAGE_SIZE })
    episodes.value.push(...res.items)
    page.value = next
    hasMore.value = res.has_more
    // No `stale = false` here: it starts false, and the fallback below sets `hasMore = false`, so
    // there is no path from a stale list back through a successful load within one mount. Leaving
    // the assignment in would be unreachable code that reads as though a recovery path exists.
    if (next === 1) void writeCached(BROWSE_CACHE_KEY, res.items)
  } catch {
    // A failed FIRST page falls back to the last one we saw. A failed later page is just the end
    // of what we can show — the list above it is still correct, so it is not an error state.
    if (page.value === 0 && !episodes.value.length) {
      const cached = await readCached<EpisodeSummary[]>(BROWSE_CACHE_KEY, isArrayCache)
      if (cached?.length) {
        episodes.value = cached
        stale.value = true
        hasMore.value = false
        return
      }
    }
    error.value = true
  } finally {
    loading.value = false
  }
}

// Filtering/sorting only makes sense over the whole list — pull the rest in when a control is used.
async function loadAll(): Promise<void> {
  while (hasMore.value && !error.value) await loadMore()
}
watch(controlsActive, (active) => {
  if (active && hasMore.value) void loadAll()
})

const visible = computed<EpisodeSummary[]>(() => {
  let list = episodes.value
  const q = search.value.trim().toLowerCase()
  if (q) {
    list = list.filter(
      (e) => e.title.toLowerCase().includes(q) || (e.podcast_title ?? "").toLowerCase().includes(q)
    )
  }
  if (filter.value === "insights") list = list.filter((e) => e.has_gi)
  // Played means played, however you got there: marked from the menu OR listened to the end.
  // These read `completed.has` directly, which is the hand-marked half alone -- so filtering
  // Played returned "no episodes match" for a listener who had finished plenty (2026-09-23).
  else if (filter.value === "unplayed") list = list.filter((e) => !isPlayed(e.slug))
  else if (filter.value === "played") list = list.filter((e) => isPlayed(e.slug))
  else if (filter.value === "downloaded") list = list.filter((e) => downloads.isDownloaded(e.slug))
  else if (filter.value === "inprogress") {
    const ip = inProgressSlugs.value
    list = ip ? list.filter((e) => ip.has(e.slug) && !isPlayed(e.slug)) : []
  }
  if (from.value === "following") {
    const followed = new Set(library.feedIds)
    list = list.filter((e) => followed.has(e.feed_id))
  } else if (from.value === "mine") {
    const w = worldSlugs.value
    list = w ? list.filter((e) => w.has(e.slug)) : []
  }
  if (show.value) list = list.filter((e) => e.feed_id === show.value)
  const byDate = (e: EpisodeSummary) => e.publish_date ?? ""
  const sorted = [...list]
  if (sort.value === "newest") sorted.sort((a, b) => byDate(b).localeCompare(byDate(a)))
  else if (sort.value === "oldest") sorted.sort((a, b) => byDate(a).localeCompare(byDate(b)))
  else if (sort.value === "az") sorted.sort((a, b) => a.title.localeCompare(b.title))
  else if (sort.value === "za") sorted.sort((a, b) => b.title.localeCompare(a.title))
  return sorted
})

/**
 * The visible list, cut into time bands — or one unlabelled band when time is not the order.
 *
 * Bands are relative to now rather than to calendar boundaries, so "this week" means the last
 * seven days rather than "since Monday": a Monday-morning reader should not watch the section
 * they were reading yesterday empty itself out.
 *
 * `oldest` sort still groups, but the bands arrive in the order the sort dictates — the labels
 * describe the content either way, which is the whole point of preferring them to an every-Nth
 * divider.
 */
/**
 * What is RENDERED. Unfiltered, the server pages the list (Load more fetches the next page). Once
 * a control is active every page is fetched, because a search, sort or filter is only right over
 * the whole list — but showing all of it at once dumped the whole catalogue on the page with no
 * Load more (operator 2026-10-05). So the matches are revealed a page at a time too.
 */
const displayCount = ref(PAGE_SIZE)
watch([search, sort, filter, from, show], () => {
  displayCount.value = PAGE_SIZE
})
const shown = computed<EpisodeSummary[]>(() =>
  controlsActive.value ? visible.value.slice(0, displayCount.value) : visible.value
)
const moreMatches = computed(() => controlsActive.value && visible.value.length > displayCount.value)
function onLoadMore(): void {
  if (controlsActive.value) displayCount.value += PAGE_SIZE
  else void loadMore()
}

const grouped = computed<Array<{ key: string; label: string; items: EpisodeSummary[] }>>(() => {
  const timeOrdered = sort.value === "newest" || sort.value === "oldest"
  if (!timeOrdered || search.value.trim()) {
    return [{ key: "all", label: "", items: shown.value }]
  }
  const now = Date.now()
  const DAY = 86_400_000
  const band = (e: EpisodeSummary): string => {
    const t = e.publish_date ? Date.parse(e.publish_date) : NaN
    if (Number.isNaN(t)) return "undated"
    const age = (now - t) / DAY
    if (age < 7) return "week"
    if (age < 31) return "month"
    if (age < 366) return "year"
    return "older"
  }
  const labels: Record<string, string> = {
    week: t("catalog.groupWeek"),
    month: t("catalog.groupMonth"),
    year: t("catalog.groupYear"),
    older: t("catalog.groupOlder"),
    undated: t("catalog.groupUndated"),
  }
  const out: Array<{ key: string; label: string; items: EpisodeSummary[] }> = []
  for (const ep of shown.value) {
    const k = band(ep)
    const last = out[out.length - 1]
    if (last && last.key === k) last.items.push(ep)
    else out.push({ key: k, label: labels[k] ?? "", items: [ep] })
  }
  return out
})

const countLabel = computed(() =>
  controlsActive.value
    ? t("list.count", { shown: visible.value.length, total: episodes.value.length })
    : ""
)

onMounted(async () => {
  // Sets the played/downloaded filters read from; fire-and-forget so they don't gate first paint.
  if (auth.isAuthenticated) void completed.ensureLoaded().catch(() => {})
  if (isNative()) void downloads.ensureLoaded().catch(() => {})
  await loadMore()
  // Opened already filtered (`?from=` / `?state=`, What's new's "all ›"): `controlsActive` never
  // CHANGES, so its watcher never pulls the rest in — the filter ran over page one alone and a
  // followed show's episodes on page two were missing (browse-filters.spec, 2026-10-10).
  if (controlsActive.value && hasMore.value) void loadAll()
  // The filter's show names only — every show, without descriptions (2026-10-08).
  shows.value = (
    await getPodcastsPage({ compact: true, limit: 200, sort: "az" })
      .then((p) => p.items)
      .catch(() => [])
  )
    .filter((p) => p.feed_id)
    .map((p) => ({ id: p.feed_id, label: p.title ?? p.feed_id }))
})
</script>

<template>
  <section>
    <h1 v-if="!embedded" class="mb-5 font-display text-3xl font-extrabold tracking-tight">
      {{ t("catalog.heading") }}
    </h1>

    <!-- OUTSIDE the loading/error/empty/list chain below, deliberately. Slotting it in the middle
         made `v-else-if` chain off THIS element instead of the loading one, so whenever the list
         was stale the final `v-else` — the list itself — was skipped: the notice rendered and the
         episodes did not, which is worse than the red sentence it replaced. -->
    <p v-if="stale" class="mb-3 text-sm text-muted" data-testid="catalog-stale">
      {{ t("catalog.stale") }}
    </p>

    <!-- F1.3/F1.4: reserve the list shape while the first page loads; retry on failure. -->
    <SectionStatus
      v-if="episodes.length === 0 && (loading || error)"
      :phase="loading ? 'loading' : 'error'"
      :rows="5"
      @retry="loadMore"
    />
    <p v-else-if="episodes.length === 0" class="text-muted">{{ t("catalog.empty") }}</p>

    <div v-else>
      <!-- Which episodes — a row of its own so it combines with the state filter below. -->
      <div v-if="auth.isAuthenticated" class="mb-3" data-testid="catalog-from">
        <Tabs
          v-model="from"
          :tabs="fromTabs"
          :label="t('browse.fromLabel')"
          id-prefix="catalog-from"
          variant="pill"
          pattern="radio"
          dense
        />
      </div>
      <ListToolbar
        v-model:search="search"
        v-model:sort="sort"
        v-model:filter="filter"
        v-model:view="view"
        :filter-options="filterOptions"
        :sort-options="sortOptions"
        :count="countLabel"
        :search-placeholder="t('browse.filterEpisodes')"
      />

      <p v-if="visible.length === 0" class="text-muted">{{ t("list.noMatches") }}</p>

      <!-- Grouped by WHEN, not chunked by count (#1978).
           The catalogue's compositional problem was measured, not assumed: 29 structurally
           identical rows down a 4,929px page with nothing to break them — "a spreadsheet with
           pictures". The critic's own alternative was "a divider every six rows", which breaks
           monotony while meaning nothing; these headings carry information instead, and they are
           the eyebrow system already used everywhere else rather than a new device.
           Only when the list is actually IN time order. Under `title` sort, or with a search
           term active, a "This week" heading over an alphabetical list would be a lie, so the
           grouping disappears and the flat list returns. -->
      <!-- List: banded rows. Grid: a flat tile grid (no time bands). -->
      <template v-if="view === 'list'">
        <template v-for="group in grouped" :key="group.key">
          <h2 v-if="group.label" class="lp-kicker mb-2 mt-6 first:mt-0">{{ group.label }}</h2>
          <EpisodeCard v-for="ep in group.items" :key="ep.slug" :episode="ep" moments-link />
        </template>
      </template>
      <!-- 3 columns on a phone, 4 from `sm` — the same shape the Shows grid and Library already use
           (operator 2026-09-17). Episodes were 2/3, so switching tabs changed the column count. -->
      <ul v-else class="grid grid-cols-3 gap-3 sm:grid-cols-4" data-testid="episode-grid">
        <li v-for="ep in shown" :key="ep.slug"><EpisodeTile :episode="ep" /></li>
      </ul>

      <!-- Same treatment as the Shows list's load-more (operator 2026-09-19): the two sit on the
           same page, one above the other, and were a pill and a full-width bar respectively. The
           full-width bar wins — it reads as the end of the list rather than as a stray control. -->
      <div class="mt-2">
        <button
          v-if="(hasMore && !controlsActive) || (moreMatches && !loading)"
          type="button"
          :disabled="loading"
          :aria-busy="loading"
          class="mt-4 w-full rounded-xl border border-border py-2.5 text-sm font-bold text-accent transition hover:bg-overlay disabled:opacity-50"
          data-testid="catalog-load-more"
          @click="onLoadMore"
        >
          {{ loading ? t("catalog.loading") : t("catalog.loadMore") }}
        </button>
        <p v-else-if="loading && controlsActive" class="mt-6 text-center text-sm text-muted">
          {{ t("catalog.loading") }}
        </p>
      </div>
    </div>
  </section>
</template>
