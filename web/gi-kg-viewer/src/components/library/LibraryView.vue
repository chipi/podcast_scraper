<script setup lang="ts">
import { computed, nextTick, onActivated, onMounted, ref, watch } from 'vue'
import { fetchIndexStats, type IndexStatsEnvelope } from '../../api/indexStatsApi'
import {
  fetchCorpusEpisodeDetail,
  fetchCorpusEpisodes,
  fetchCorpusFeeds,
  fetchCorpusSimilarEpisodes,
  type CorpusEpisodeDetailResponse,
  type CorpusEpisodeListItem,
  type CorpusFeedItem,
  type CorpusSimilarEpisodeItem,
} from '../../api/corpusLibraryApi'
import CollapsibleSection from '../shared/CollapsibleSection.vue'
import HelpTip from '../shared/HelpTip.vue'
import PodcastCover from '../shared/PodcastCover.vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { useShellStore } from '../../stores/shell'
import { buildLibrarySearchHandoffQuery } from '../../utils/corpusSearchHandoff'
import { digestRowFeedLabelWithCatalog, libraryEpisodeSummaryLine } from '../../utils/digestRowDisplay'
import { feedNameHoverWithCatalogLookup } from '../../utils/feedHoverTitle'
import { formatDurationSeconds } from '../../utils/formatDuration'
import { normalizeFeedIdForViewer } from '../../utils/feedId'

defineOptions({ name: 'LibraryView' })

const emit = defineEmits<{
  'switch-main-tab': [tab: 'graph' | 'dashboard']
  'focus-search': [payload: { feed: string; query: string }]
}>()

const shell = useShellStore()
const artifacts = useArtifactsStore()
const graphNav = useGraphNavigationStore()

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
  parts.push(sinceQ.value.trim() ? `≥ ${sinceQ.value.trim()}` : 'Any date')
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
const sinceQ = ref('')
/** Substring filter on summary title or bullets (set by topic pills or cleared manually). */
const topicQ = ref('')

/** Visible pill text cap; full string on hover via native `title`. */
const TOPIC_PILL_CHARS = 24

const selectedMetaPath = ref<string | null>(null)
const detail = ref<CorpusEpisodeDetailResponse | null>(null)
const detailError = ref<string | null>(null)
const detailLoading = ref(false)
const graphActionError = ref<string | null>(null)

const similarItems = ref<CorpusSimilarEpisodeItem[]>([])
const similarLoading = ref(false)
const similarError = ref<string | null>(null)
const similarQueryUsed = ref('')
/** Last similar request finished with HTTP 200 and no API ``error`` field. */
const similarRanOk = ref(false)

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

function isAllFeedsSelected(): boolean {
  return feedFilterId.value === null
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

const detailDiagnosticsEntries = computed(() => {
  const d = detail.value
  if (!d) {
    return [] as { label: string; value: string }[]
  }
  const cat = feeds.value.find((f) => f.feed_id === d.feed_id)
  const env = indexStatsEnvelope.value
  const inIndex = d.feed_id ? feedHasVectorIndex(d.feed_id) : false
  const rows: { label: string; value: string }[] = [
    { label: 'Metadata path', value: d.metadata_relative_path || '—' },
    { label: 'Feed ID', value: d.feed_id || '—' },
    {
      label: 'Feed title (feeds API)',
      value: cat?.display_title?.trim() || '—',
    },
    { label: 'Episode ID', value: d.episode_id ?? '—' },
    {
      label: 'GI artifact',
      value: d.has_gi ? `yes · ${d.gi_relative_path}` : 'no',
    },
    {
      label: 'KG artifact',
      value: d.has_kg ? `yes · ${d.kg_relative_path}` : 'no',
    },
    {
      label: 'Feed in vector index',
      value: d.feed_id
        ? inIndex
          ? 'yes (feed_id listed in GET /api/index/stats feeds_indexed)'
          : 'no'
        : '—',
    },
  ]
  if (env) {
    rows.push({
      label: 'Index API available',
      value: env.available ? 'yes' : 'no',
    })
    if (!env.available && env.reason) {
      rows.push({ label: 'Index reason', value: String(env.reason) })
    }
    const st = env.stats
    if (st) {
      rows.push({ label: 'Total vectors', value: String(st.total_vectors) })
      rows.push({ label: 'Embedding model', value: st.embedding_model || '—' })
      rows.push({ label: 'Embedding dim', value: String(st.embedding_dim) })
      rows.push({ label: 'Index last updated', value: st.last_updated || '—' })
    }
  } else {
    rows.push({
      label: 'Index stats',
      value: 'not loaded yet (reload Library after setting corpus path)',
    })
  }
  return rows
})

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

function detailFeedHoverTitle(): string {
  const d = detail.value
  if (!d) {
    return ''
  }
  return feedNameHoverWithCatalogLookup(
    {
      feed_id: d.feed_id,
      feed_rss_url: d.feed_rss_url,
      feed_description: d.feed_description,
    },
    feeds.value,
    normalizeFeedIdForViewer,
  )
}

function episodeFeedInlineVisibleList(row: EpisodeRowForFeedHover): boolean {
  const fid = row.feed_id?.trim()
  if (fid) {
    return true
  }
  const lab = episodeRowFeedLabel(row)
  return lab !== 'Unknown feed' && Boolean(lab.trim())
}

function detailFeedLine(): string {
  const d = detail.value
  if (!d?.feed_id?.trim()) {
    return '(no feed id)'
  }
  return digestRowFeedLabelWithCatalog({ feed_id: d.feed_id }, feedDisplayTitleById.value)
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

function formatLocalYMD(d: Date): string {
  const y = d.getFullYear()
  const m = String(d.getMonth() + 1).padStart(2, '0')
  const day = String(d.getDate()).padStart(2, '0')
  return `${y}-${m}-${day}`
}

/** Preset lower bound for ``since`` (publish date on or after), local calendar day. */
function setSincePreset(kind: 'all' | 7 | 30 | 90): void {
  if (kind === 'all') {
    sinceQ.value = ''
  } else {
    const d = new Date()
    d.setDate(d.getDate() - kind)
    sinceQ.value = formatLocalYMD(d)
  }
  applyEpisodeFilters()
}

function clearTextAndDateFilters(): void {
  titleQ.value = ''
  sinceQ.value = ''
  topicQ.value = ''
  applyEpisodeFilters()
}

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
  return selectedMetaPath.value === e.metadata_relative_path
}

async function loadFeeds(): Promise<void> {
  feedsError.value = null
  const root = shell.corpusPath.trim()
  if (!root || !shell.healthStatus) {
    feeds.value = []
    feedsIndexed.value = new Set()
    indexStatsEnvelope.value = null
    return
  }
  feedsLoading.value = true
  try {
    const body = await fetchCorpusFeeds(root)
    feeds.value = body.feeds
    try {
      const env = await fetchIndexStats(root)
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
      feedsIndexed.value = new Set()
      indexStatsEnvelope.value = null
    }
  } catch (e) {
    feeds.value = []
    feedsIndexed.value = new Set()
    indexStatsEnvelope.value = null
    feedsError.value = e instanceof Error ? e.message : String(e)
  } finally {
    feedsLoading.value = false
  }
}

async function loadEpisodes(append: boolean): Promise<void> {
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
      since: sinceQ.value.trim() || undefined,
      limit: 50,
      cursor,
    }
    if (feedFilterId.value !== null) {
      opts.feedId = feedFilterId.value
    }
    const body = await fetchCorpusEpisodes(root, opts)
    if (append) {
      episodes.value = [...episodes.value, ...body.items]
    } else {
      episodes.value = body.items
    }
    nextCursor.value = body.next_cursor
  } catch (e) {
    if (!append) {
      episodes.value = []
    }
    nextCursor.value = null
    episodesError.value = e instanceof Error ? e.message : String(e)
  } finally {
    episodesLoading.value = false
    if (!append && !episodesError.value && episodes.value.length > 0) {
      if (selectedMetaPath.value === null) {
        void nextTick(() => {
          selectEpisode(episodes.value[0]!)
        })
      }
    }
  }
}

async function loadDetail(metaPath: string): Promise<void> {
  detailError.value = null
  detail.value = null
  const root = shell.corpusPath.trim()
  if (!root || !shell.healthStatus) {
    return
  }
  detailLoading.value = true
  try {
    detail.value = await fetchCorpusEpisodeDetail(root, metaPath)
  } catch (e) {
    detailError.value = e instanceof Error ? e.message : String(e)
  } finally {
    detailLoading.value = false
  }
}

function selectEpisode(row: CorpusEpisodeListItem): void {
  selectedMetaPath.value = row.metadata_relative_path
  void loadDetail(row.metadata_relative_path)
}

async function openInGraph(): Promise<void> {
  graphActionError.value = null
  if (!detail.value) {
    return
  }
  const paths: string[] = []
  if (detail.value.has_gi) {
    paths.push(detail.value.gi_relative_path)
  }
  if (detail.value.has_kg) {
    paths.push(detail.value.kg_relative_path)
  }
  if (paths.length === 0) {
    graphActionError.value = 'No GI/KG artifacts on disk for this episode.'
    return
  }
  await artifacts.loadRelativeArtifacts(paths)
  graphNav.clearLibraryEpisodeHighlights()
  const eid = detail.value.episode_id?.trim()
  if (eid) {
    graphNav.requestFocusNode(eid)
    graphNav.setLibraryEpisodeHighlights([eid])
  } else {
    graphNav.clearPendingFocus()
  }
  emit('switch-main-tab', 'graph')
}

function openInSearch(): void {
  if (!detail.value) {
    return
  }
  emit('focus-search', {
    feed: normalizeFeedIdForViewer(detail.value.feed_id),
    query: buildLibrarySearchHandoffQuery(detail.value),
  })
}

function mapSimilarError(code: string, detailMsg: string | null): string {
  if (code === 'no_index') {
    return 'No vector index for this corpus yet. Run indexing to find similar episodes.'
  }
  if (code === 'insufficient_text') {
    return detailMsg || 'Add a longer summary or title to use similarity search.'
  }
  if (code === 'embed_failed') {
    return 'Embedding failed (model missing or offline).'
  }
  return detailMsg || code
}

async function loadSimilarEpisodes(): Promise<void> {
  similarError.value = null
  similarItems.value = []
  similarQueryUsed.value = ''
  similarRanOk.value = false
  if (!detail.value) {
    return
  }
  const root = shell.corpusPath.trim()
  if (!root) {
    return
  }
  similarLoading.value = true
  try {
    const body = await fetchCorpusSimilarEpisodes(root, detail.value.metadata_relative_path)
    similarQueryUsed.value = body.query_used || ''
    if (body.error) {
      similarError.value = mapSimilarError(body.error, body.detail)
      return
    }
    similarItems.value = body.items
    similarRanOk.value = true
  } catch (e) {
    similarError.value = e instanceof Error ? e.message : String(e)
  } finally {
    similarLoading.value = false
  }
}

function openSimilarEpisode(row: CorpusSimilarEpisodeItem): void {
  const p = row.metadata_relative_path?.trim()
  if (!p) {
    return
  }
  selectedMetaPath.value = p
  void loadDetail(p)
}

function applyEpisodeFilters(): void {
  selectedMetaPath.value = null
  detail.value = null
  void loadEpisodes(false)
}

watch(
  () => [shell.corpusPath, shell.healthStatus] as const,
  (_newV, oldV) => {
    if (oldV !== undefined) {
      shell.takePendingLibraryEpisode()
    }
    selectedMetaPath.value = null
    detail.value = null
    feedFilterId.value = null
    titleQ.value = ''
    sinceQ.value = ''
    topicQ.value = ''
    similarItems.value = []
    similarError.value = null
    similarQueryUsed.value = ''
    similarRanOk.value = false
    void loadFeeds().then(() => loadEpisodes(false))
  },
  { immediate: true },
)

watch(
  () => detail.value?.metadata_relative_path,
  () => {
    similarItems.value = []
    similarError.value = null
    similarQueryUsed.value = ''
    similarRanOk.value = false
  },
)

watch(feedFilterId, () => {
  selectedMetaPath.value = null
  detail.value = null
  void loadEpisodes(false)
})

function applyPendingLibraryFromShell(): void {
  const p = shell.takePendingLibraryEpisode()
  if (!p) {
    return
  }
  selectedMetaPath.value = p
  void loadDetail(p)
}

onMounted(() => {
  applyPendingLibraryFromShell()
})

onActivated(() => {
  applyPendingLibraryFromShell()
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
            <span
              >Narrow episodes by publish date, title, summary topic, or feed (same title/date/topic
              fields as Search).</span
            >
          </template>
          <div
            class="grid grid-cols-1 gap-3 border-b border-border pb-3 lg:grid-cols-3 lg:items-start lg:gap-x-4 lg:gap-y-2"
          >
            <!-- Left: date -->
            <div
              class="min-w-0 space-y-1.5 border-b border-border pb-3 lg:border-b-0 lg:border-r lg:pb-0 lg:pr-4"
            >
              <div class="flex min-w-0 flex-row items-center gap-2">
                <label
                  for="lib-filter-since-q"
                  class="shrink-0 text-[10px] font-medium text-muted"
                >Published on or after</label>
                <input
                  id="lib-filter-since-q"
                  v-model="sinceQ"
                  type="text"
                  inputmode="numeric"
                  class="min-w-0 flex-1 rounded border border-border bg-elevated px-2 py-1 text-xs text-surface-foreground"
                  placeholder="YYYY-MM-DD"
                  aria-label="Published on or after date"
                  @keydown.enter="applyEpisodeFilters()"
                >
              </div>
              <div class="flex flex-wrap gap-1">
                <button
                  type="button"
                  class="rounded border border-border bg-surface px-2 py-0.5 text-[10px] text-surface-foreground hover:bg-overlay/40"
                  @click="setSincePreset('all')"
                >
                  All time
                </button>
                <button
                  type="button"
                  class="rounded border border-border bg-surface px-2 py-0.5 text-[10px] text-surface-foreground hover:bg-overlay/40"
                  @click="setSincePreset(7)"
                >
                  7d
                </button>
                <button
                  type="button"
                  class="rounded border border-border bg-surface px-2 py-0.5 text-[10px] text-surface-foreground hover:bg-overlay/40"
                  @click="setSincePreset(30)"
                >
                  30d
                </button>
                <button
                  type="button"
                  class="rounded border border-border bg-surface px-2 py-0.5 text-[10px] text-surface-foreground hover:bg-overlay/40"
                  @click="setSincePreset(90)"
                >
                  90d
                </button>
              </div>
            </div>
            <!-- Center: text + actions -->
            <div
              class="min-w-0 space-y-1.5 border-b border-border pb-3 lg:border-b-0 lg:border-r lg:pb-0 lg:pr-4"
            >
              <div>
                <label
                  class="mb-0.5 block text-[10px] font-medium text-muted"
                  for="lib-filter-title-q"
                  >Title</label
                >
                <input
                  id="lib-filter-title-q"
                  v-model="titleQ"
                  type="search"
                  class="w-full rounded border border-border bg-elevated px-2 py-1 text-xs text-surface-foreground"
                  placeholder="Episode title…"
                  aria-label="Filter episodes by title"
                  @keydown.enter="applyEpisodeFilters()"
                >
              </div>
              <div>
                <label
                  class="mb-0.5 block text-[10px] font-medium text-muted"
                  for="lib-filter-topic-q"
                  >Summary / topic</label
                >
                <input
                  id="lib-filter-topic-q"
                  v-model="topicQ"
                  type="search"
                  class="w-full rounded border border-border bg-elevated px-2 py-1 text-xs text-surface-foreground"
                  placeholder="Summary or bullets…"
                  aria-label="Summary or topic filter"
                  @keydown.enter="applyEpisodeFilters()"
                >
              </div>
              <div class="flex flex-wrap items-center justify-center gap-1 pt-0.5">
                <button
                  type="button"
                  class="rounded bg-primary px-2 py-1 text-[10px] font-medium text-primary-foreground hover:opacity-90"
                  @click="applyEpisodeFilters()"
                >
                  Apply
                </button>
                <button
                  type="button"
                  class="rounded border border-border bg-surface px-2 py-1 text-[10px] text-surface-foreground hover:bg-overlay/40"
                  @click="clearTextAndDateFilters()"
                >
                  Clear text &amp; date
                </button>
              </div>
            </div>
            <!-- Right: feed -->
            <div class="flex min-h-0 min-w-0 flex-col">
              <p class="mb-1 text-[10px] font-medium text-muted">Feed</p>
              <div
                class="max-h-32 min-h-0 flex-1 overflow-y-auto sm:max-h-36 lg:max-h-32"
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
              <li>
                <button
                  type="button"
                  class="w-full rounded px-2 py-1 text-left hover:bg-overlay"
                  :class="isAllFeedsSelected() ? 'bg-overlay text-surface-foreground' : 'text-muted'"
                  @click="selectAllFeeds()"
                >
                  All feeds
                </button>
              </li>
              <li v-for="f in feeds" :key="f.feed_id || '__empty__'">
                <button
                  type="button"
                  class="flex w-full min-w-0 items-center gap-2 rounded px-2 py-1 text-left hover:bg-overlay"
                  :class="
                    isFeedRowSelected(f) ? 'bg-overlay text-surface-foreground' : 'text-muted'
                  "
                  :title="feedRowTitleAttr(f)"
                  :aria-label="feedRowAccessibleName(f)"
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

        <div class="flex min-h-0 flex-1 flex-col gap-2 lg:flex-row lg:items-stretch">
        <!-- Episodes + filters -->
        <div
          class="flex min-h-52 min-w-0 flex-1 flex-col rounded border border-border bg-surface lg:min-h-0"
          role="region"
          aria-label="Episodes"
        >
          <div class="border-b border-border p-2">
            <h2 id="library-episodes-heading" class="text-xs font-semibold text-surface-foreground">
              Episodes
            </h2>
            <p class="mt-0.5 text-[10px] text-muted">
              Filters live in
              <span class="font-medium text-surface-foreground">Episode filters</span>
              above.
            </p>
          </div>
          <div class="min-h-0 flex-1 overflow-y-auto p-1">
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
              <li v-for="e in episodes" :key="episodeKey(e)">
                <div
                  role="button"
                  tabindex="0"
                  class="flex w-full gap-2 rounded px-2 py-1.5 text-left outline-none ring-offset-1 hover:bg-overlay focus-visible:ring-2 focus-visible:ring-primary"
                  :class="isEpisodeSelected(e) ? 'bg-overlay' : ''"
                  :aria-label="`${e.episode_title}, ${episodeRowFeedLabel(e)}`"
                  @click="selectEpisode(e)"
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
                      class="flex min-w-0 max-w-[min(100%,14rem)] shrink-0 flex-col items-stretch gap-0.5 text-[10px] text-muted"
                    >
                      <div class="flex w-full min-w-0 items-baseline gap-2">
                        <span
                          v-if="episodeFeedInlineVisibleList(e)"
                          class="min-w-0 flex-1 truncate text-left font-semibold leading-tight text-surface-foreground"
                          :title="episodeListFeedHoverTitle(e) || undefined"
                        >{{ episodeRowFeedLabel(e) }}</span>
                        <span
                          v-if="e.publish_date"
                          class="shrink-0 text-right tabular-nums"
                          :class="episodeFeedInlineVisibleList(e) ? '' : 'ml-auto'"
                        >{{ e.publish_date }}</span>
                      </div>
                      <div class="flex flex-col items-end gap-0.5 text-right">
                        <span v-if="e.episode_number != null">E{{ e.episode_number }}</span>
                        <span v-if="formatDurationSeconds(e.duration_seconds)">{{
                          formatDurationSeconds(e.duration_seconds)
                        }}</span>
                      </div>
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
            <button
              v-if="nextCursor"
              type="button"
              class="mt-2 w-full rounded border border-border py-1 text-xs hover:bg-overlay"
              :disabled="episodesLoading"
              @click="loadEpisodes(true)"
            >
              {{ episodesLoading ? 'Loading…' : 'Load more' }}
            </button>
          </div>
        </div>

        <!-- Episode panel (selected episode) -->
        <div
          class="flex w-full shrink-0 flex-col rounded border border-border bg-surface lg:w-[22rem] lg:min-w-[22rem]"
          role="region"
          aria-label="Episode"
        >
          <h2 class="border-b border-border px-2 py-1.5 text-xs font-semibold text-surface-foreground">
            Episode
          </h2>
          <div class="min-h-0 flex-1 overflow-y-auto p-2 text-sm">
            <p v-if="detailLoading" class="text-xs text-muted">
              Loading…
            </p>
            <p v-else-if="detailError" class="text-xs text-danger">
              {{ detailError }}
            </p>
            <p v-else-if="!detail" class="text-xs text-muted">
              Select an episode.
            </p>
            <template v-else>
              <div class="flex gap-3">
                <PodcastCover
                  :corpus-path="shell.corpusPath"
                  :episode-image-local-relpath="detail.episode_image_local_relpath"
                  :feed-image-local-relpath="detail.feed_image_local_relpath"
                  :episode-image-url="detail.episode_image_url"
                  :feed-image-url="detail.feed_image_url"
                  :alt="`Cover for ${detail.episode_title}`"
                  size-class="h-20 w-20 sm:h-24 sm:w-24"
                />
                <div class="min-w-0 flex-1">
                  <div class="flex items-start justify-between gap-1.5">
                    <h3 class="min-w-0 flex-1 text-base font-semibold text-surface-foreground">
                      {{ detail.episode_title }}
                    </h3>
                    <HelpTip
                      class="shrink-0 pt-0.5"
                      :pref-width="300"
                      button-aria-label="Episode and feed diagnostics"
                    >
                      <p class="mb-2 font-sans text-[11px] font-semibold text-surface-foreground">
                        Troubleshooting
                      </p>
                      <p class="mb-2 font-sans text-[10px] text-muted">
                        Paths and ids for support — same data the viewer uses for search, similar
                        episodes, and graph loads.
                      </p>
                      <dl class="space-y-1.5 font-mono text-[10px] leading-snug">
                        <template
                          v-for="(row, di) in detailDiagnosticsEntries"
                          :key="di"
                        >
                          <dt class="font-sans font-medium text-muted">
                            {{ row.label }}
                          </dt>
                          <dd class="break-words text-elevated-foreground">
                            {{ row.value }}
                          </dd>
                        </template>
                      </dl>
                    </HelpTip>
                  </div>
                  <div
                    v-if="
                      detail.publish_date ||
                        detail.episode_number != null ||
                        formatDurationSeconds(detail.duration_seconds) ||
                        detail.feed_id
                    "
                    class="mt-0.5 space-y-0.5 text-xs text-muted"
                  >
                    <div class="flex min-w-0 flex-row items-baseline justify-between gap-2">
                      <span
                        class="min-w-0 flex-1 cursor-help truncate font-medium text-surface-foreground"
                        :title="detailFeedHoverTitle() || undefined"
                      >{{ detailFeedLine() }}</span>
                      <span
                        v-if="detail.publish_date"
                        class="shrink-0 tabular-nums"
                      >{{ detail.publish_date }}</span>
                    </div>
                    <p
                      v-if="
                        detail.episode_number != null ||
                          formatDurationSeconds(detail.duration_seconds)
                      "
                      class="text-right text-[10px] text-muted"
                    >
                      <span v-if="detail.episode_number != null">E{{ detail.episode_number }}</span>
                      <span
                        v-if="formatDurationSeconds(detail.duration_seconds)"
                        class="ml-1"
                      >
                        {{ formatDurationSeconds(detail.duration_seconds) }}
                      </span>
                    </p>
                  </div>
                </div>
              </div>
              <p v-if="detail.summary_title" class="mt-2 text-xs font-medium text-surface-foreground">
                {{ detail.summary_title }}
              </p>
              <p
                v-if="detail.summary_text"
                class="mt-2 whitespace-pre-wrap text-xs leading-relaxed text-muted"
              >
                {{ detail.summary_text }}
              </p>
              <div
                v-if="detail.summary_bullets.length"
                :class="
                  detail.summary_title || detail.summary_text
                    ? 'mt-3 border-t border-border pt-3'
                    : 'mt-2'
                "
              >
                <h4
                  v-if="detail.summary_title || detail.summary_text"
                  class="mb-2 text-[10px] font-semibold uppercase tracking-wide text-muted"
                >
                  Key points
                </h4>
                <ul class="list-inside list-disc space-y-0.5 text-xs leading-relaxed text-muted">
                  <li v-for="(b, i) in detail.summary_bullets" :key="i">
                    {{ b }}
                  </li>
                </ul>
              </div>
              <p
                v-if="!detail.summary_bullets.length && !detail.summary_text"
                class="mt-2 text-xs text-muted"
              >
                No summary text in metadata.
              </p>
              <div class="mt-3 flex items-stretch gap-2">
                <button
                  type="button"
                  class="min-w-0 flex-1 rounded bg-gi px-2 py-1.5 text-center text-xs font-medium leading-snug text-gi-foreground disabled:opacity-40"
                  :disabled="!detail.has_gi && !detail.has_kg"
                  @click="openInGraph()"
                >
                  Open in graph
                </button>
                <button
                  type="button"
                  class="min-w-0 flex-1 rounded bg-primary px-2 py-1.5 text-center text-xs font-medium leading-snug text-primary-foreground"
                  @click="openInSearch()"
                >
                  Prefill semantic search
                </button>
                <HelpTip class="shrink-0 self-center">
                  Opens the Search panel with feed filter and query filled from this episode
                  (summary text and bullets when present). Run Search to query the vector index.
                </HelpTip>
              </div>
              <div
                class="mt-4 border-t border-border pt-2"
                role="region"
                aria-label="Similar episodes"
                data-testid="library-similar"
              >
                <h4 class="text-xs font-semibold text-surface-foreground">
                  Similar episodes
                </h4>
                <button
                  type="button"
                  class="mt-1 w-full rounded border border-border px-2 py-1 text-xs hover:bg-overlay disabled:opacity-50"
                  :disabled="similarLoading"
                  @click="loadSimilarEpisodes()"
                >
                  {{ similarLoading ? 'Searching…' : 'Find similar episodes' }}
                </button>
                <p
                  v-if="similarQueryUsed"
                  class="mt-1 line-clamp-2 text-[10px] text-muted"
                >
                  Query: {{ similarQueryUsed }}
                </p>
                <p v-if="similarError" class="mt-1 text-xs text-danger">
                  {{ similarError }}
                </p>
                <p
                  v-else-if="
                    similarRanOk &&
                      !similarLoading &&
                      similarItems.length === 0
                  "
                  class="mt-1 text-xs text-muted"
                  data-testid="library-similar-empty"
                >
                  No similar episodes matched. Try another episode or rebuild the index.
                </p>
                <ul v-else-if="similarItems.length" class="mt-1 space-y-1 text-xs">
                  <li v-for="(s, si) in similarItems" :key="si">
                    <button
                      type="button"
                      class="flex w-full gap-2 rounded px-1 py-0.5 text-left hover:bg-overlay disabled:opacity-50"
                      :disabled="!s.metadata_relative_path"
                      @click="openSimilarEpisode(s)"
                    >
                      <PodcastCover
                        :corpus-path="shell.corpusPath"
                        :episode-image-local-relpath="s.episode_image_local_relpath"
                        :feed-image-local-relpath="s.feed_image_local_relpath"
                        :episode-image-url="s.episode_image_url"
                        :feed-image-url="s.feed_image_url"
                        :alt="`Cover for ${s.episode_title || 'episode'}`"
                        size-class="h-8 w-8"
                      />
                      <span class="min-w-0 flex-1">
                        <span class="font-medium text-surface-foreground">{{
                          s.episode_title || (s.snippet ? s.snippet.slice(0, 48) : '') || '(episode)'
                        }}</span>
                        <span class="ml-1 text-[10px] text-muted">{{ s.score.toFixed(3) }}</span>
                      </span>
                    </button>
                  </li>
                </ul>
              </div>
              <p v-if="graphActionError" class="mt-1 text-xs text-danger">
                {{ graphActionError }}
              </p>
            </template>
          </div>
        </div>
        </div>
      </div>
    </template>
  </div>
</template>
