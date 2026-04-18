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
import { fetchIndexStats, type IndexStatsEnvelope } from '../../api/indexStatsApi'
import { fetchCorpusEpisodes, fetchCorpusFeeds, type CorpusEpisodeListItem, type CorpusFeedItem } from '../../api/corpusLibraryApi'
import CollapsibleSection from '../shared/CollapsibleSection.vue'
import HelpTip from '../shared/HelpTip.vue'
import PodcastCover from '../shared/PodcastCover.vue'
import { useCorpusLensStore } from '../../stores/corpusLens'
import { useEpisodeRailStore } from '../../stores/episodeRail'
import { useShellStore } from '../../stores/shell'
import { digestRowFeedLabelWithCatalog, libraryEpisodeSummaryLine } from '../../utils/digestRowDisplay'
import { feedNameHoverWithCatalogLookup } from '../../utils/feedHoverTitle'
import { formatDurationSeconds } from '../../utils/formatDuration'
import { normalizeFeedIdForViewer } from '../../utils/feedId'
import { handleVerticalListArrowKeydown } from '../../utils/listRowArrowNav'
import { StaleGeneration } from '../../utils/staleGeneration'

defineOptions({ name: 'LibraryView' })

/** ``GET /api/corpus/episodes`` page size (cursor pagination; scroll + Load more). */
const LIBRARY_EPISODES_PAGE_SIZE = 20

/**
 * Feed picker in Episode filters: cap list height so the filter column stays compact.
 * Approximate row: ``h-8`` cover + ``py-1`` + optional ``Indexed`` chip (use slightly generous rem).
 */
const LIBRARY_FEED_FILTER_VISIBLE_ROWS = 3
const LIBRARY_FEED_FILTER_ROW_REM = 2.625
const LIBRARY_FEED_FILTER_GAP_REM = 0.125
const LIBRARY_FEED_FILTER_LIST_MAX_HEIGHT_REM =
  LIBRARY_FEED_FILTER_VISIBLE_ROWS * LIBRARY_FEED_FILTER_ROW_REM +
  (LIBRARY_FEED_FILTER_VISIBLE_ROWS - 1) * LIBRARY_FEED_FILTER_GAP_REM

const emit = defineEmits<{
  'focus-search': [
    payload: { feed: string; query: string; since?: string; feedDisplayTitle?: string },
  ]
}>()

const shell = useShellStore()
const episodeRail = useEpisodeRailStore()
const corpusLens = useCorpusLensStore()
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
    episodeRail.clearEpisodeContext()
    void loadEpisodes(false)
  }, CORPUS_LENS_DEBOUNCE_MS)
}

function applySinceDateReloadEpisodesNow(): void {
  if (corpusLensDebounceTimer) {
    clearTimeout(corpusLensDebounceTimer)
    corpusLensDebounceTimer = null
  }
  episodeRail.clearEpisodeContext()
  void loadEpisodes(false)
}

function truncateSummaryPart(s: string, max: number): string {
  const t = s.trim()
  if (t.length <= max) {
    return t
  }
  return `${t.slice(0, max - 1)}…`
}

const filtersSectionSummary = computed(() => {
  const parts: string[] = []
  const n = feeds.value.length
  if (feedFilterId.value === null) {
    parts.push(n ? `All feeds (${n})` : 'No feeds')
  } else {
    const f = feeds.value.find((x) => x.feed_id === feedFilterId.value)
    parts.push(f ? feedRowVisibleLabel(f) : 'One feed')
  }
  parts.push(sinceYmd.value.trim() ? `≥ ${sinceYmd.value.trim()}` : 'Any date')
  if (titleQ.value.trim()) {
    parts.push(`title: ${truncateSummaryPart(titleQ.value, 18)}`)
  }
  if (topicQ.value.trim()) {
    parts.push(`topic: ${truncateSummaryPart(topicQ.value, 18)}`)
  }
  return parts.join(' · ')
})

const feeds = ref<CorpusFeedItem[]>([])
const feedsError = ref<string | null>(null)
const feedsLoading = ref(false)
/** Feed ids that appear in the vector index (from ``GET /api/index/stats``). */
const feedsIndexed = ref<Set<string>>(new Set())

/** Last index envelope from ``loadFeeds`` (detail diagnostics / troubleshooting). */
const indexStatsEnvelope = ref<IndexStatsEnvelope | null>(null)

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
/** Substring filter on summary title or bullets (set by topic pills or cleared manually). */
const topicQ = ref('')

/** Visible pill text cap; full string on hover via native `title`. */
const TOPIC_PILL_CHARS = 24

/** Visible line in the feed list: title when present, else id (no redundant long id in parens). */
function feedRowVisibleLabel(f: CorpusFeedItem): string {
  if (f.feed_id === '') {
    return '(No feed id)'
  }
  const t = f.display_title?.trim()
  if (t) {
    return t
  }
  return f.feed_id.trim()
}

/** Hover/native tooltip: title, id, RSS URL, description when present. */
function feedRowTitleAttr(f: CorpusFeedItem): string {
  const parts: string[] = []
  if (f.feed_id === '') {
    parts.push('(No feed id)')
  } else {
    const t = f.display_title?.trim()
    const id = f.feed_id.trim()
    if (t && id) {
      parts.push(`${t} · ${id}`)
    } else {
      parts.push(id || t || '')
    }
  }
  if (f.rss_url?.trim()) {
    parts.push(`RSS: ${f.rss_url.trim()}`)
  }
  if (f.description?.trim()) {
    parts.push(f.description.trim())
  }
  return parts.filter(Boolean).join('\n')
}

function feedRowAccessibleName(f: CorpusFeedItem): string {
  if (f.feed_id === '') {
    return `(No feed id), ${f.episode_count} episodes`
  }
  const t = f.display_title?.trim()
  const id = f.feed_id.trim()
  if (t) {
    return `${t}, feed id ${id}, ${f.episode_count} episodes`
  }
  return `${id}, ${f.episode_count} episodes`
}

function isFeedRowSelected(f: CorpusFeedItem): boolean {
  return feedFilterId.value !== null && feedFilterId.value === f.feed_id
}

function selectAllFeeds(): void {
  feedFilterId.value = null
}

function selectFeed(f: CorpusFeedItem): void {
  feedFilterId.value = f.feed_id
}

function feedHasVectorIndex(feedId: string): boolean {
  return feedsIndexed.value.has(normalizeFeedIdForViewer(feedId))
}

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

function topicPillShort(label: string): string {
  const s = label.trim()
  if (s.length <= TOPIC_PILL_CHARS) {
    return s
  }
  return `${s.slice(0, TOPIC_PILL_CHARS - 1)}…`
}

function onTopicPillClick(topic: string): void {
  const t = topic.trim()
  if (!t) {
    return
  }
  topicQ.value = t
  applyEpisodeFilters()
}

/** Preset lower bound for ``since`` (publish date on or after), local calendar day. */
function setSincePreset(kind: 'all' | 7 | 30 | 90): void {
  corpusLens.setPreset(kind)
}

/** Clear title and summary/topic inputs only (date uses **All time** / presets). */
function clearTextFilters(): void {
  titleQ.value = ''
  topicQ.value = ''
  episodeRail.clearEpisodeContext()
  void loadEpisodes(false)
}

const hasLibraryTextFilters = computed(
  () => Boolean(titleQ.value.trim() || topicQ.value.trim()),
)

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

/** Topic pills: same bullet source as Digest when API sends `summary_bullets_preview`. */
function episodeTopicPills(e: CorpusEpisodeListItem): string[] {
  const preview = e.summary_bullets_preview
  if (preview?.length) {
    return preview
  }
  return e.topics?.length ? e.topics : []
}

function episodeKey(e: CorpusEpisodeListItem): string {
  return e.metadata_relative_path
}

function isEpisodeSelected(e: CorpusEpisodeListItem): boolean {
  const p = e.metadata_relative_path?.trim()
  const cur = episodeRail.metadataRelativePath?.trim()
  return Boolean(p && cur && p === cur)
}

/**
 * Load feed catalog + index stats.
 * @returns false if a newer ``loadFeeds`` started before this one applied results.
 */
async function loadFeeds(): Promise<boolean> {
  const seq = libraryFeedsGate.bump()
  feedsError.value = null
  const root = shell.corpusPath.trim()
  if (!root || !shell.healthStatus) {
    feeds.value = []
    feedsIndexed.value = new Set()
    indexStatsEnvelope.value = null
    return libraryFeedsGate.isCurrent(seq)
  }
  feedsLoading.value = true
  try {
    const body = await fetchCorpusFeeds(root)
    if (libraryFeedsGate.isStale(seq)) {
      return false
    }
    feeds.value = body.feeds
    try {
      const env = await fetchIndexStats(root)
      if (libraryFeedsGate.isStale(seq)) {
        return false
      }
      indexStatsEnvelope.value = env
      if (env.available && env.stats?.feeds_indexed?.length) {
        feedsIndexed.value = new Set(
          env.stats.feeds_indexed
            .map((s) => normalizeFeedIdForViewer(s))
            .filter(Boolean),
        )
      } else {
        feedsIndexed.value = new Set()
      }
    } catch {
      if (libraryFeedsGate.isStale(seq)) {
        return false
      }
      feedsIndexed.value = new Set()
      indexStatsEnvelope.value = null
    }
  } catch (e) {
    if (libraryFeedsGate.isStale(seq)) {
      return false
    }
    feeds.value = []
    feedsIndexed.value = new Set()
    indexStatsEnvelope.value = null
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
      since: sinceYmd.value.trim() || undefined,
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
      const path = episodeRail.metadataRelativePath?.trim()
      if (path && nextCursor.value === null) {
        const inList = episodes.value.some((e) => e.metadata_relative_path === path)
        if (!inList) {
          episodeRail.clearEpisodeContext()
        }
      }
      if (
        !append &&
        episodes.value.length > 0 &&
        episodeRail.metadataRelativePath == null
      ) {
        void nextTick(() => {
          selectEpisode(episodes.value[0]!)
        })
      }
    }
  }
}

function selectEpisode(row: CorpusEpisodeListItem): void {
  episodeRail.openEpisodePanel(row.metadata_relative_path)
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
  episodeRail.clearEpisodeContext()
  void loadEpisodes(false)
}

watch(
  () => [shell.corpusPath, shell.healthStatus] as const,
  (_newV, oldV) => {
    if (oldV !== undefined) {
      shell.takePendingLibraryEpisode()
      shell.takePendingLibraryTopicQ()
      episodeRail.clearEpisodeContext()
      feedFilterId.value = null
      titleQ.value = ''
      topicQ.value = ''
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
  episodeRail.clearEpisodeContext()
  void loadEpisodes(false)
})

watch(sinceYmd, () => {
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

function applyPendingLibraryTopicFromShell(): void {
  const t = shell.takePendingLibraryTopicQ()
  if (!t) {
    return
  }
  topicQ.value = t
  applyEpisodeFilters()
}

function applyPendingLibraryFromShell(): void {
  const p = shell.takePendingLibraryEpisode()
  if (!p) {
    return
  }
  episodeRail.openEpisodePanel(p)
}

onMounted(() => {
  applyPendingLibraryTopicFromShell()
  applyPendingLibraryFromShell()
})

onActivated(() => {
  applyPendingLibraryTopicFromShell()
  applyPendingLibraryFromShell()
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
        <CollapsibleSection
          title="Episode filters"
          :summary="filtersSectionSummary"
          :default-open="true"
        >
          <template #subtitle>
            <div
              class="flex flex-wrap items-center justify-between gap-x-3 gap-y-2"
            >
              <span class="min-w-0 flex-1 text-xs leading-snug">
                Narrow episodes by publish date, title, summary topic, or feed (same
                title/date/topic fields as Search).
              </span>
              <button
                type="button"
                class="shrink-0 rounded bg-primary px-2 py-0.5 text-[10px] font-medium text-primary-foreground hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                title="Reload episodes using title and summary filters (same as Enter in those fields)"
                @click="applyEpisodeFilters()"
              >
                Apply
              </button>
            </div>
          </template>
          <div
            class="grid grid-cols-1 gap-3 border-b border-border pb-3 lg:grid-cols-3 lg:items-start lg:gap-x-4 lg:gap-y-2"
          >
            <!-- Left: date -->
            <div
              class="min-w-0 space-y-1.5 border-b border-border pb-3 lg:border-b-0 lg:border-r lg:pb-0 lg:pr-4"
            >
              <p class="text-[10px] font-medium text-muted">
                Dates
              </p>
              <div class="flex min-w-0 flex-row items-center gap-2">
                <label
                  for="lib-filter-since-q"
                  class="shrink-0 text-[10px] font-medium text-muted"
                >Published on or after</label>
                <input
                  id="lib-filter-since-q"
                  v-model="sinceYmd"
                  type="text"
                  inputmode="numeric"
                  class="min-w-0 flex-1 rounded border border-border bg-elevated px-2 py-1 text-xs text-surface-foreground"
                  placeholder="YYYY-MM-DD"
                  aria-label="Published on or after date"
                  @keydown.enter="applySinceDateReloadEpisodesNow()"
                >
              </div>
              <div class="flex flex-wrap items-center gap-1">
                <button
                  type="button"
                  class="rounded border border-border bg-surface px-2 py-0.5 text-[10px] text-surface-foreground hover:bg-overlay/40"
                  :class="activePreset === 'all' ? 'ring-2 ring-primary' : ''"
                  @click="setSincePreset('all')"
                >
                  All time
                </button>
                <button
                  type="button"
                  class="rounded border border-border bg-surface px-2 py-0.5 text-[10px] text-surface-foreground hover:bg-overlay/40"
                  :class="activePreset === '7' ? 'ring-2 ring-primary' : ''"
                  @click="setSincePreset(7)"
                >
                  7d
                </button>
                <button
                  type="button"
                  class="rounded border border-border bg-surface px-2 py-0.5 text-[10px] text-surface-foreground hover:bg-overlay/40"
                  :class="activePreset === '30' ? 'ring-2 ring-primary' : ''"
                  @click="setSincePreset(30)"
                >
                  30d
                </button>
                <button
                  type="button"
                  class="rounded border border-border bg-surface px-2 py-0.5 text-[10px] text-surface-foreground hover:bg-overlay/40"
                  :class="activePreset === '90' ? 'ring-2 ring-primary' : ''"
                  @click="setSincePreset(90)"
                >
                  90d
                </button>
              </div>
            </div>
            <!-- Center: title + summary (clear text in header, like Feed column) -->
            <div
              class="min-w-0 space-y-1.5 border-b border-border pb-3 lg:border-b-0 lg:border-r lg:pb-0 lg:pr-4"
            >
              <div class="flex min-w-0 items-start justify-between gap-2">
                <p class="text-[10px] font-medium text-muted">
                  Title &amp; summary
                </p>
                <button
                  type="button"
                  class="shrink-0 rounded border border-border bg-surface px-2 py-0.5 text-[10px] font-medium text-surface-foreground hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:bg-surface"
                  :disabled="!hasLibraryTextFilters"
                  :aria-label="
                    hasLibraryTextFilters
                      ? 'Clear title and summary or topic filter text'
                      : 'Clear text filters (enter title or summary text first)'
                  "
                  @click="clearTextFilters()"
                >
                  Clear text
                </button>
              </div>
              <div
                class="grid min-w-0 grid-cols-[auto_minmax(0,1fr)] items-center gap-x-2 gap-y-1.5"
              >
                <label
                  class="shrink-0 text-[10px] font-medium text-muted"
                  for="lib-filter-title-q"
                >Title</label>
                <input
                  id="lib-filter-title-q"
                  v-model="titleQ"
                  type="search"
                  class="min-w-0 rounded border border-border bg-elevated px-2 py-1 text-xs text-surface-foreground"
                  placeholder="Episode title…"
                  aria-label="Filter episodes by title"
                  @keydown.enter="applyEpisodeFilters()"
                >
                <label
                  class="shrink-0 text-[10px] font-medium text-muted"
                  for="lib-filter-topic-q"
                >Summary / topic</label>
                <input
                  id="lib-filter-topic-q"
                  v-model="topicQ"
                  type="search"
                  class="min-w-0 rounded border border-border bg-elevated px-2 py-1 text-xs text-surface-foreground"
                  placeholder="Summary or bullets…"
                  aria-label="Summary or topic filter"
                  @keydown.enter="applyEpisodeFilters()"
                >
              </div>
            </div>
            <!-- Right: feed -->
            <div class="flex min-h-0 min-w-0 flex-col">
              <div class="mb-1 flex min-w-0 items-start justify-between gap-2">
                <p class="text-[10px] font-medium text-muted">
                  Feed
                </p>
                <button
                  type="button"
                  class="shrink-0 rounded border border-border bg-surface px-2 py-0.5 text-[10px] font-medium text-surface-foreground hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary disabled:cursor-not-allowed disabled:opacity-45 disabled:hover:bg-surface"
                  :disabled="feedFilterId === null"
                  :aria-label="
                    feedFilterId === null
                      ? 'Clear feed filter (select a feed first)'
                      : 'Clear feed filter and show all feeds'
                  "
                  @click="selectAllFeeds()"
                >
                  Clear feed filter
                </button>
              </div>
              <div
                class="min-h-0 overflow-y-auto"
                :style="{ maxHeight: `${LIBRARY_FEED_FILTER_LIST_MAX_HEIGHT_REM}rem` }"
                role="region"
                aria-label="Feeds"
              >
            <p v-if="feedsLoading" class="px-1 text-xs text-muted">
              Loading…
            </p>
            <p v-else-if="feedsError" class="px-1 text-xs text-danger">
              {{ feedsError }}
            </p>
            <ul v-else class="space-y-0.5 text-sm">
              <li v-for="f in feeds" :key="f.feed_id || '__empty__'">
                <button
                  type="button"
                  class="flex w-full min-w-0 items-center gap-2 rounded px-2 py-1 text-left hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                  :class="
                    isFeedRowSelected(f) ? 'bg-overlay text-surface-foreground' : 'text-muted'
                  "
                  :title="feedRowTitleAttr(f)"
                  :aria-label="feedRowAccessibleName(f)"
                  :aria-pressed="isFeedRowSelected(f)"
                  @click="selectFeed(f)"
                >
                  <PodcastCover
                    :corpus-path="shell.corpusPath"
                    :feed-image-local-relpath="f.image_local_relpath"
                    :feed-image-url="f.image_url"
                    :alt="`Cover for ${feedRowVisibleLabel(f)}`"
                    size-class="h-8 w-8"
                  />
                  <span class="min-w-0 flex-1 truncate text-sm">{{
                    feedRowVisibleLabel(f)
                  }}</span>
                  <span class="shrink-0 text-[10px] text-muted">({{ f.episode_count }})</span>
                  <span
                    v-if="feedHasVectorIndex(f.feed_id)"
                    class="shrink-0 rounded bg-elevated px-1 text-[9px] font-medium text-muted"
                    title="This feed id appears in the vector index"
                  >Indexed</span>
                </button>
              </li>
            </ul>
              </div>
            </div>
          </div>
        </CollapsibleSection>

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
                  <strong class="font-medium text-surface-foreground/90">Episode filters</strong>
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
                    <span class="min-w-0 flex-1 truncate font-medium text-surface-foreground">{{
                      e.episode_title
                    }}</span>
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
                    class="mt-1 break-words whitespace-pre-wrap text-[11px] leading-snug text-muted"
                  >
                    {{ episodeListSummaryLine(e) }}
                  </p>
                  <div
                    v-if="episodeTopicPills(e).length"
                    class="mt-1 flex flex-wrap gap-1"
                  >
                    <button
                      v-for="(t, ti) in episodeTopicPills(e)"
                      :key="ti"
                      type="button"
                      class="max-w-[11rem] shrink-0 truncate rounded-full border border-border bg-canvas px-1.5 py-0.5 text-[10px] font-medium text-surface-foreground hover:bg-overlay"
                      :title="t.trim() || undefined"
                      :aria-label="`Filter episodes by topic: ${t}`"
                      @click.stop="onTopicPillClick(t)"
                    >
                      {{ topicPillShort(t) }}
                    </button>
                  </div>
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
