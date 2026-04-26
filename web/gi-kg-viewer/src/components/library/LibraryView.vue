<script setup lang="ts">
import {
  computed,
  nextTick,
  onActivated,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from 'vue'
import { storeToRefs } from 'pinia'
import { fetchCorpusEpisodes, fetchCorpusFeeds, type CorpusEpisodeListItem, type CorpusFeedItem } from '../../api/corpusLibraryApi'
import LibraryFilterBar from './LibraryFilterBar.vue'
import PodcastCover from '../shared/PodcastCover.vue'
import { useCorpusLensStore } from '../../stores/corpusLens'
import { useDashboardNavStore } from '../../stores/dashboardNav'
import { useSubjectStore } from '../../stores/subject'
import { useShellStore } from '../../stores/shell'
import {
  digestRowFeedLabelWithCatalog,
  libraryEpisodeSummaryLine,
} from '../../utils/digestRowDisplay'
import { isPublishDateWithin24hRolling, recencyDotHoverTitle } from '../../utils/digestRecency'
import { feedNameHoverWithCatalogLookup } from '../../utils/feedHoverTitle'
import { formatDurationSeconds } from '../../utils/formatDuration'
import { normalizeFeedIdForViewer } from '../../utils/feedId'
import { handleVerticalListArrowKeydown } from '../../utils/listRowArrowNav'
import { inferCorpusLensPreset } from '../../utils/localCalendarDate'
import { StaleGeneration } from '../../utils/staleGeneration'

defineOptions({ name: 'LibraryView' })

/** ``GET /api/corpus/episodes`` page size (cursor pagination; scroll + Load more). */
const LIBRARY_EPISODES_PAGE_SIZE = 20

const emit = defineEmits<{
  'focus-search': [
    payload: { feed: string; query: string; since?: string; feedDisplayTitle?: string },
  ]
}>()

const shell = useShellStore()
const subject = useSubjectStore()
const corpusLens = useCorpusLensStore()
const dashboardNav = useDashboardNavStore()

/** Optional upper bound (``YYYY-MM-DD``) from Dashboard handoff; cleared after load. */
const libraryUntilYmd = ref('')
/** When ``false``, request episodes missing GI from the API. */
const libraryHasGiFilter = ref<boolean | undefined>(undefined)
const { sinceYmd, activePreset } = storeToRefs(corpusLens)

/** Debounce episode reload when the shared date field changes (keystrokes). */
const CORPUS_LENS_DEBOUNCE_MS = 400
let corpusLensDebounceTimer: ReturnType<typeof setTimeout> | null = null

function scheduleReloadEpisodesFromCorpusLens(): void {
  if (corpusLensDebounceTimer) {
    clearTimeout(corpusLensDebounceTimer)
  }
  corpusLensDebounceTimer = setTimeout(() => {
    corpusLensDebounceTimer = null
    subject.clearSubject()
    void loadEpisodes(false)
  }, CORPUS_LENS_DEBOUNCE_MS)
}

function applySinceDateReloadEpisodesNow(): void {
  if (corpusLensDebounceTimer) {
    clearTimeout(corpusLensDebounceTimer)
    corpusLensDebounceTimer = null
  }
  subject.clearSubject()
  void loadEpisodes(false)
}

function truncateSummaryPart(s: string, max: number): string {
  const t = s.trim()
  if (t.length <= max) {
    return t
  }
  return `${t.slice(0, max - 1)}…`
}

const feeds = ref<CorpusFeedItem[]>([])
const feedsError = ref<string | null>(null)
const feedsLoading = ref(false)

const libraryFeedsGate = new StaleGeneration()
const libraryEpisodesGate = new StaleGeneration()

/** Non-empty ``display_title`` from ``GET /api/corpus/feeds`` keyed by ``feed_id``. */
const feedDisplayTitleById = computed(() => {
  const m: Record<string, string> = {}
  for (const f of feeds.value) {
    const id = normalizeFeedIdForViewer(f.feed_id)
    const t = f.display_title?.trim()
    if (id && t) {
      m[id] = t
    }
  }
  return m
})

/** `null` = all feeds; string (including empty) = filter by ``feed_id`` */
const feedFilterId = ref<string | null>(null)

const episodes = ref<CorpusEpisodeListItem[]>([])
const episodesError = ref<string | null>(null)
const episodesLoading = ref(false)
const nextCursor = ref<string | null>(null)
const titleQ = ref('')
/** Substring filter on summary title or bullets (typed or cleared manually). */
const topicQ = ref('')
/** Only episodes whose bridge topics include at least one corpus multi-member cluster topic. */
const topicClusterOnly = ref(false)

/** Visible line in the feed list: title when present, else id (no redundant long id in parens). */
// Feed-row display helpers (`feedRowVisibleLabel`, `feedRowTitleAttr`,
// `feedRowAccessibleName`) moved to ``utils/corpusFeedRowDisplay`` in
// #658 so the chip + library code share one definition. Library still
// uses ``feedRowVisibleLabel`` for the feed-display lookup map below.

/** Feed line for list / detail — matches Library sidebar titles when available. */
function episodeRowFeedLabel(row: {
  feed_id: string
  feed_display_title?: string | null
}): string {
  return digestRowFeedLabelWithCatalog(row, feedDisplayTitleById.value)
}

type EpisodeRowForFeedHover = CorpusEpisodeListItem

function episodeListFeedHoverTitle(row: EpisodeRowForFeedHover): string {
  return feedNameHoverWithCatalogLookup(row, feeds.value, normalizeFeedIdForViewer)
}

function episodeFeedInlineVisibleList(row: EpisodeRowForFeedHover): boolean {
  const fid = row.feed_id?.trim()
  if (fid) {
    return true
  }
  const lab = episodeRowFeedLabel(row)
  return lab !== 'Unknown feed' && Boolean(lab.trim())
}

/** Preset lower bound for ``since`` (publish date on or after), local calendar day. */
function setSincePreset(kind: 'all' | 7 | 30 | 90): void {
  corpusLens.setPreset(kind)
}

function clearAllLibraryFilters(): void {
  feedFilterId.value = null
  titleQ.value = ''
  topicQ.value = ''
  topicClusterOnly.value = false
  corpusLens.setPreset('all')
  applySinceDateReloadEpisodesNow()
}

/** Shown next to **Episodes** heading: loaded row count; **+** when more pages exist. */
const libraryEpisodesCountDisplay = computed(() => {
  if (episodesLoading.value && episodes.value.length === 0) {
    return ''
  }
  const n = episodes.value.length
  if (n === 0 && !episodesLoading.value) {
    return '(0)'
  }
  if (n === 0) {
    return ''
  }
  return nextCursor.value ? `(${n}+)` : `(${n})`
})

const libraryEpisodesRegionAriaLabel = computed(() => {
  if (episodesLoading.value && episodes.value.length === 0) {
    return 'Episodes'
  }
  const n = episodes.value.length
  if (episodesError.value) {
    return 'Episodes'
  }
  if (n === 0) {
    return 'Episodes, no matches'
  }
  if (nextCursor.value) {
    return `Episodes, ${n} loaded, more available when you scroll or use Load more`
  }
  return `Episodes, ${n} items`
})

/** Same recap line as Digest episode cards (preview + title/bullets; ``topics`` fallback). */
function episodeListSummaryLine(e: CorpusEpisodeListItem): string {
  return libraryEpisodeSummaryLine(e)
}

function episodeKey(e: CorpusEpisodeListItem): string {
  return e.metadata_relative_path
}

function isEpisodeSelected(e: CorpusEpisodeListItem): boolean {
  const p = e.metadata_relative_path?.trim()
  const cur = subject.episodeMetadataPath?.trim()
  return Boolean(p && cur && p === cur)
}

/**
 * Load feed catalog from ``GET /api/corpus/feeds``.
 * @returns false if a newer ``loadFeeds`` started before this one applied results.
 */
async function loadFeeds(): Promise<boolean> {
  const seq = libraryFeedsGate.bump()
  feedsError.value = null
  const root = shell.corpusPath.trim()
  if (!root || !shell.healthStatus) {
    feeds.value = []
    return libraryFeedsGate.isCurrent(seq)
  }
  feedsLoading.value = true
  try {
    const body = await fetchCorpusFeeds(root)
    if (libraryFeedsGate.isStale(seq)) {
      return false
    }
    feeds.value = body.feeds
  } catch (e) {
    if (libraryFeedsGate.isStale(seq)) {
      return false
    }
    feeds.value = []
    feedsError.value = e instanceof Error ? e.message : String(e)
    return libraryFeedsGate.isCurrent(seq)
  } finally {
    if (libraryFeedsGate.isCurrent(seq)) {
      feedsLoading.value = false
    }
  }
  return libraryFeedsGate.isCurrent(seq)
}

const episodesListScrollRootRef = ref<HTMLElement | null>(null)
const episodesInfiniteSentinelRef = ref<HTMLElement | null>(null)
let episodesInfiniteObserver: IntersectionObserver | null = null

function teardownEpisodesInfiniteObserver(): void {
  episodesInfiniteObserver?.disconnect()
  episodesInfiniteObserver = null
}

function maybeAppendEpisodesFromScroll(): void {
  if (!nextCursor.value || episodesLoading.value) return
  void loadEpisodes(true)
}

function setupEpisodesInfiniteObserver(): void {
  teardownEpisodesInfiniteObserver()
  const root = episodesListScrollRootRef.value
  const sentinel = episodesInfiniteSentinelRef.value
  if (!root || !sentinel || !nextCursor.value || episodesLoading.value) {
    return
  }
  episodesInfiniteObserver = new IntersectionObserver(
    (entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        maybeAppendEpisodesFromScroll()
      }
    },
    { root, rootMargin: '100px 0px 0px 0px', threshold: 0 },
  )
  episodesInfiniteObserver.observe(sentinel)
}

async function loadEpisodes(append: boolean): Promise<void> {
  const seq = libraryEpisodesGate.bump()
  episodesError.value = null
  const root = shell.corpusPath.trim()
  if (!root || !shell.healthStatus) {
    episodes.value = []
    nextCursor.value = null
    return
  }
  episodesLoading.value = true
  try {
    const cursor = append ? nextCursor.value : null
    const opts: Parameters<typeof fetchCorpusEpisodes>[1] = {
      q: titleQ.value.trim() || undefined,
      topicQ: topicQ.value.trim() || undefined,
      topicClusterOnly: topicClusterOnly.value || undefined,
      since: sinceYmd.value.trim() || undefined,
      until: libraryUntilYmd.value.trim() || undefined,
      hasGi: libraryHasGiFilter.value,
      limit: LIBRARY_EPISODES_PAGE_SIZE,
      cursor,
    }
    if (feedFilterId.value !== null) {
      opts.feedId = feedFilterId.value
    }
    const body = await fetchCorpusEpisodes(root, opts)
    if (libraryEpisodesGate.isStale(seq)) {
      return
    }
    if (append) {
      episodes.value = [...episodes.value, ...body.items]
    } else {
      episodes.value = body.items
    }
    nextCursor.value = body.next_cursor
  } catch (e) {
    if (libraryEpisodesGate.isStale(seq)) {
      return
    }
    if (!append) {
      episodes.value = []
    }
    nextCursor.value = null
    episodesError.value = e instanceof Error ? e.message : String(e)
  } finally {
    if (libraryEpisodesGate.isCurrent(seq)) {
      episodesLoading.value = false
    }
    if (libraryEpisodesGate.isCurrent(seq) && !episodesError.value) {
      const path = subject.episodeMetadataPath?.trim()
      if (path && nextCursor.value === null) {
        const inList = episodes.value.some((e) => e.metadata_relative_path === path)
        if (!inList) {
          subject.clearSubject()
        }
      }
      if (
        !append &&
        episodes.value.length > 0 &&
        subject.episodeMetadataPath == null &&
        subject.kind == null
      ) {
        void nextTick(() => {
          selectEpisode(episodes.value[0]!)
        })
      }
    }
  }
}

function selectEpisode(row: CorpusEpisodeListItem): void {
  subject.focusEpisode(row.metadata_relative_path, {
    uiTitle: row.episode_title?.trim() || null,
  })
}

const LIBRARY_EPISODE_ROW_SELECTOR = '[data-library-episode-row]'

function onLibraryEpisodeRowKeydown(e: KeyboardEvent, index: number): void {
  handleVerticalListArrowKeydown(e, index, {
    itemCount: episodes.value.length,
    scrollRoot: episodesListScrollRootRef.value,
    rowSelector: LIBRARY_EPISODE_ROW_SELECTOR,
    activateIndex: (i) => selectEpisode(episodes.value[i]!),
  })
}

function applyEpisodeFilters(): void {
  subject.clearSubject()
  void loadEpisodes(false)
}

/**
 * Reload catalog when corpus path or API health changes.
 *
 * Reset Library **filter form** only when the **corpus path** changes. A follow-up
 * ``fetchHealth`` that tweaks ``healthStatus`` (refetch, casing, or flag-only deltas)
 * must **not** clear ``topic_cluster_only`` and other filters — that made the cluster
 * checkbox look broken (list jumped back to the unfiltered page).
 */
watch(
  () => [shell.corpusPath.trim(), shell.healthStatus] as const,
  async (newV, oldV) => {
    const pathChanged = oldV !== undefined && oldV[0] !== newV[0]
    if (pathChanged) {
      subject.clearSubject()
      feedFilterId.value = null
      titleQ.value = ''
      topicQ.value = ''
      topicClusterOnly.value = false
      corpusLens.reset()
    }
    void (async () => {
      const stillCurrent = await loadFeeds()
      if (!stillCurrent) {
        return
      }
      await loadEpisodes(false)
    })()
  },
  { immediate: true },
)

watch(feedFilterId, () => {
  subject.clearSubject()
  void loadEpisodes(false)
})

/**
 * Checkbox ``v-model`` can commit after a raw ``@change`` handler runs, so reloading in
 * ``@change`` sometimes called ``GET /api/corpus/episodes`` with the **previous**
 * ``topic_cluster_only`` flag (list looked unchanged). Flush after Vue updates the ref.
 */
watch(
  topicClusterOnly,
  () => {
    if (!shell.corpusPath.trim() || !shell.healthStatus) {
      return
    }
    subject.clearSubject()
    void loadEpisodes(false)
  },
  { flush: 'post' },
)

watch(sinceYmd, () => {
  scheduleReloadEpisodesFromCorpusLens()
})

// #669: Title and Summary inputs are persistent (no Apply button), so
// keystrokes debounce the same way ``sinceYmd`` does. Enter on either
// input still calls ``applyEpisodeFilters`` for an immediate reload.
watch(titleQ, () => {
  scheduleReloadEpisodesFromCorpusLens()
})

watch(topicQ, () => {
  scheduleReloadEpisodesFromCorpusLens()
})

/** Cursor pages also load when the list sentinel scrolls into view (same API as **Load more**). */
watch(
  () =>
    [nextCursor.value, episodes.value.length, episodesLoading.value] as const,
  () => {
    void nextTick(setupEpisodesInfiniteObserver)
  },
  { flush: 'post' },
)

function applyDashboardLibraryHandoff(): void {
  const nav = dashboardNav.consumeLibraryHandoffIfPending()
  if (!nav) {
    return
  }
  if (nav.feedId !== undefined) {
    feedFilterId.value = nav.feedId.trim() ? nav.feedId : null
  }
  if (nav.since !== undefined) {
    corpusLens.sinceYmd = nav.since ?? ''
  }
  libraryUntilYmd.value = nav.until?.trim() ?? ''
  libraryHasGiFilter.value = nav.missingGiOnly === true ? false : undefined
  subject.clearSubject()
  void loadEpisodes(false)
}

/** First paint from Dashboard can run before ``onActivated`` in some keep-alive timings. */
onMounted(() => {
  applyDashboardLibraryHandoff()
  void nextTick(setupEpisodesInfiniteObserver)
})

onActivated(() => {
  applyDashboardLibraryHandoff()
  void nextTick(setupEpisodesInfiniteObserver)
})

onBeforeUnmount(() => {
  if (corpusLensDebounceTimer) {
    clearTimeout(corpusLensDebounceTimer)
    corpusLensDebounceTimer = null
  }
  teardownEpisodesInfiniteObserver()
})
</script>

<template>
  <div
    class="flex h-full min-h-[280px] flex-col gap-2 p-3 text-surface-foreground"
    data-testid="library-root"
  >
    <p v-if="!shell.healthStatus" class="text-sm text-muted">
      Corpus Library needs a healthy API. Set corpus path, fix API connection, or use the Graph tab with local files.
    </p>
    <p
      v-else-if="!shell.corpusLibraryApiAvailable"
      class="text-sm text-danger"
    >
      This viewer API does not expose Corpus Library (<code class="text-xs">/api/corpus/*</code>).
      Stop the process on your API port and restart from a current checkout:
      <code class="text-xs">pip install -e &quot;.[server]&quot;</code>
      then
      <code class="text-xs">make serve-api</code>
      or
      <code class="text-xs">podcast serve --output-dir …</code>
      (Vite must proxy to that server — default port 8000).
    </p>
    <template v-else>
      <p v-if="!shell.hasCorpusPath" class="text-sm text-muted">
        Set <strong>Corpus path</strong> in the left panel (same as List files).
      </p>
      <div v-else class="flex min-h-0 flex-1 flex-col gap-2">
        <LibraryFilterBar
          :feed-filter-id="feedFilterId"
          :since-ymd="sinceYmd"
          :topic-cluster-only="topicClusterOnly"
          :feeds="feeds"
          :corpus-path="shell.corpusPath"
          :feeds-loading="feedsLoading"
          :feeds-error="feedsError"
          @update:feed-filter-id="(v) => (feedFilterId = v)"
          @update:since-ymd="(v) => (sinceYmd = v)"
          @update:topic-cluster-only="(v) => (topicClusterOnly = v)"
          @reset="clearAllLibraryFilters"
        />
        <div class="flex flex-wrap items-center gap-2 px-2 pb-2">
          <input
            id="lib-filter-title-q"
            v-model="titleQ"
            type="search"
            class="min-w-0 flex-1 rounded border border-border bg-elevated px-2 py-1 text-xs text-surface-foreground"
            placeholder="Title contains…"
            aria-label="Filter episodes by title"
            data-testid="library-filter-title"
            @keydown.enter="applyEpisodeFilters()"
          >
          <input
            id="lib-filter-topic-q"
            v-model="topicQ"
            type="search"
            class="min-w-0 flex-1 rounded border border-border bg-elevated px-2 py-1 text-xs text-surface-foreground"
            placeholder="Summary or bullets contain…"
            aria-label="Filter episodes by summary"
            data-testid="library-filter-summary"
            @keydown.enter="applyEpisodeFilters()"
          >
        </div>

        <div class="flex min-h-0 flex-1 flex-col gap-2">
        <!-- Episodes list (detail in right sidebar Episode panel) -->
        <div
          class="flex min-h-52 min-w-0 flex-1 flex-col rounded border border-border bg-surface lg:min-h-0"
          role="region"
          :aria-label="libraryEpisodesRegionAriaLabel"
        >
          <div class="border-b border-border p-2">
            <div class="flex items-center gap-1.5">
              <h2
                id="library-episodes-heading"
                class="flex flex-wrap items-baseline gap-x-1 text-xs font-semibold text-surface-foreground"
              >
                <span>Episodes</span>
                <span
                  v-if="libraryEpisodesCountDisplay"
                  class="font-normal tabular-nums text-muted"
                  :title="
                    nextCursor
                      ? 'More episodes load when you scroll or use Load more'
                      : undefined
                  "
                >{{ libraryEpisodesCountDisplay }}</span>
              </h2>
              <HelpTip
                class="shrink-0"
                :pref-width="304"
                button-aria-label="About the Library episode list"
              >
                <p class="mb-2 font-sans text-[11px] font-semibold text-surface-foreground">
                  Episode list
                </p>
                <p class="mb-2 font-sans text-[10px] text-muted">
                  Filters live in
                  <strong class="font-medium text-surface-foreground/90">Filters</strong>
                  above.
                </p>
                <p class="mb-2 font-sans text-[10px] text-muted">
                  When more pages exist, scroll the list to load them automatically, or use
                  <strong class="font-medium text-surface-foreground/90">Load more</strong>.
                </p>
                <p class="mb-2 font-sans text-[10px] text-muted">
                  <strong class="font-medium text-surface-foreground/90">Arrow up/down</strong>
                  and <strong class="font-medium text-surface-foreground/90">Home</strong> /
                  <strong class="font-medium text-surface-foreground/90">End</strong>
                  move between rows; the
                  <strong class="font-medium text-surface-foreground/90">Episode</strong>
                  panel updates as you move.
                </p>
                <p class="font-sans text-[10px] text-muted">
                  Episode summary, similar episodes, and actions open in the
                  <strong class="font-medium text-surface-foreground/90">Episode</strong>
                  panel on the right when you select a row.
                </p>
              </HelpTip>
            </div>
          </div>
          <div
            ref="episodesListScrollRootRef"
            class="min-h-0 flex-1 overflow-y-auto p-1"
          >
            <p v-if="episodesLoading && episodes.length === 0" class="px-1 text-xs text-muted">
              Loading…
            </p>
            <p v-else-if="episodesError" class="px-1 text-xs text-danger">
              {{ episodesError }}
            </p>
            <ul v-else-if="episodes.length === 0" class="px-1 text-xs text-muted">
              No episodes match.
            </ul>
            <ul v-else class="space-y-0.5 text-sm">
              <li v-for="(e, ei) in episodes" :key="episodeKey(e)">
                <div
                  role="button"
                  tabindex="0"
                  data-library-episode-row
                  class="flex w-full gap-2 rounded px-2 py-1.5 text-left outline-none ring-offset-1 focus-visible:ring-2 focus-visible:ring-primary"
                  :class="
                    isEpisodeSelected(e) ? 'bg-overlay' : 'hover:bg-overlay/35'
                  "
                  :aria-label="`${e.episode_title}, ${episodeRowFeedLabel(e)}`"
                  @click="selectEpisode(e)"
                  @keydown="onLibraryEpisodeRowKeydown($event, ei)"
                  @keydown.enter.prevent="selectEpisode(e)"
                  @keydown.space.prevent="selectEpisode(e)"
                >
                  <PodcastCover
                    :corpus-path="shell.corpusPath"
                    :episode-image-local-relpath="e.episode_image_local_relpath"
                    :feed-image-local-relpath="e.feed_image_local_relpath"
                    :episode-image-url="e.episode_image_url"
                    :feed-image-url="e.feed_image_url"
                    :alt="`Cover for ${e.episode_title}`"
                    size-class="h-9 w-9"
                  />
                  <div class="min-w-0 flex-1">
                  <div class="flex items-baseline justify-between gap-2">
                    <div class="flex min-w-0 flex-1 items-baseline gap-1.5">
                      <span
                        v-if="isPublishDateWithin24hRolling(e.publish_date)"
                        role="img"
                        class="mb-0.5 h-1.5 w-1.5 shrink-0 self-center rounded-full bg-success"
                        :title="recencyDotHoverTitle(e.publish_date) || undefined"
                        :aria-label="
                          recencyDotHoverTitle(e.publish_date) || 'Recently published'
                        "
                      />
                      <span class="min-w-0 flex-1 truncate font-medium text-surface-foreground">{{
                        e.episode_title
                      }}</span>
                    </div>
                    <div
                      class="flex min-w-0 max-w-[min(100%,14rem)] shrink-0 items-baseline justify-end gap-2 text-[10px] text-muted"
                    >
                      <span
                        v-if="episodeFeedInlineVisibleList(e)"
                        class="min-w-0 flex-1 truncate text-left font-semibold leading-tight text-surface-foreground"
                        :title="episodeListFeedHoverTitle(e) || undefined"
                      >{{ episodeRowFeedLabel(e) }}</span>
                      <span
                        v-if="
                          e.publish_date ||
                            e.episode_number != null ||
                            formatDurationSeconds(e.duration_seconds)
                        "
                        class="inline-flex shrink-0 flex-wrap items-baseline justify-end gap-x-1.5 text-right tabular-nums leading-tight"
                      >
                        <span v-if="e.publish_date">{{ e.publish_date }}</span>
                        <span v-if="e.episode_number != null">E{{ e.episode_number }}</span>
                        <span v-if="formatDurationSeconds(e.duration_seconds)">{{
                          formatDurationSeconds(e.duration_seconds)
                        }}</span>
                      </span>
                    </div>
                  </div>
                  <p
                    v-if="episodeListSummaryLine(e)"
                    class="mt-1 break-words text-[11px] leading-snug text-muted"
                    :class="isEpisodeSelected(e) ? 'whitespace-pre-wrap' : 'line-clamp-2'"
                    :title="
                      isEpisodeSelected(e)
                        ? undefined
                        : episodeListSummaryLine(e) || undefined
                    "
                  >
                    {{ episodeListSummaryLine(e) }}
                  </p>
                  </div>
                </div>
              </li>
            </ul>
            <div
              v-if="nextCursor"
              ref="episodesInfiniteSentinelRef"
              class="h-1 w-full shrink-0"
              aria-hidden="true"
            />
            <button
              v-if="nextCursor"
              type="button"
              class="mt-2 w-full rounded border border-border py-1 text-xs hover:bg-overlay"
              :disabled="episodesLoading"
              @click="loadEpisodes(true)"
            >
              {{ episodesLoading ? 'Loading…' : 'Load more' }}
            </button>
            <p
              v-if="nextCursor && episodesLoading && episodes.length > 0"
              class="mt-1 px-1 text-center text-[10px] text-muted"
            >
              Loading more episodes…
            </p>
          </div>
        </div>
        </div>
      </div>
    </template>
  </div>
</template>
