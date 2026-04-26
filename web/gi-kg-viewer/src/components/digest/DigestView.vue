<script setup lang="ts">
import { computed, inject, onActivated, onBeforeUnmount, ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import {
  fetchCorpusDigest,
  type CorpusDigestResponse,
  type CorpusDigestRow,
  type CorpusDigestTopicBand,
  type CorpusDigestTopicHit,
} from '../../api/digestApi'
import { fetchCorpusFeeds, type CorpusFeedItem } from '../../api/corpusLibraryApi'
import CilTopicPillsRow from '../shared/CilTopicPillsRow.vue'
import HelpTip from '../shared/HelpTip.vue'
import PodcastCover from '../shared/PodcastCover.vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { useCorpusLensStore } from '../../stores/corpusLens'
import DateChip from '../shared/DateChip.vue'
import { useSubjectStore } from '../../stores/subject'
import { useGraphExplorerStore } from '../../stores/graphExplorer'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { useDashboardNavStore } from '../../stores/dashboardNav'
import { useShellStore } from '../../stores/shell'
import {
  digestRowFeedLabelWithCatalog,
  digestRowSummaryPreview,
  digestTopicHitSimilarityDisplay,
  libraryEpisodeSummaryLine,
} from '../../utils/digestRowDisplay'
import { isPublishDateWithin24hRolling, recencyDotHoverTitle } from '../../utils/digestRecency'
import { feedNameHoverWithCatalogLookup } from '../../utils/feedHoverTitle'
import { formatDurationSeconds } from '../../utils/formatDuration'
import { formatUtcDateTimeForDisplay } from '../../utils/formatting'
import { normalizeFeedIdForViewer } from '../../utils/feedId'
import { handleVerticalListArrowKeydown } from '../../utils/listRowArrowNav'
import { StaleGeneration } from '../../utils/staleGeneration'
import { corpusGraphBaselineLoaderKey } from '../../corpusGraphBaseline'
import {
  applyGraphFocusPlan,
  graphFocusPlanFromCilPill,
} from '../../utils/cilGraphFocus'

const emit = defineEmits<{
  'switch-main-tab': [tab: 'graph' | 'dashboard' | 'library']
  'focus-search': [payload: { feed: string; query: string; since?: string }]
  'open-library-episode': [payload: { metadata_relative_path: string }]
}>()

const shell = useShellStore()
const dashboardNav = useDashboardNavStore()
const artifacts = useArtifactsStore()
const graphExplorer = useGraphExplorerStore()
const loadCorpusGraphBaseline = inject(corpusGraphBaselineLoaderKey, null)
const graphNav = useGraphNavigationStore()

async function ensureDefaultCorpusGraphIfNeeded(): Promise<void> {
  if (!loadCorpusGraphBaseline) {
    return
  }
  if (graphExplorer.graphTabOpenedThisSession && artifacts.selectedRelPaths.length > 0) {
    return
  }
  await loadCorpusGraphBaseline()
}
const subject = useSubjectStore()
const corpusLens = useCorpusLensStore()
const { sinceYmd } = storeToRefs(corpusLens)

const CORPUS_LENS_DEBOUNCE_MS = 400
let digestLensDebounceTimer: ReturnType<typeof setTimeout> | null = null

function scheduleLoadDigestFromCorpusLens(): void {
  if (digestLensDebounceTimer) {
    clearTimeout(digestLensDebounceTimer)
  }
  digestLensDebounceTimer = setTimeout(() => {
    digestLensDebounceTimer = null
    void loadDigest()
  }, CORPUS_LENS_DEBOUNCE_MS)
}

const digest = ref<CorpusDigestResponse | null>(null)
const error = ref<string | null>(null)
const loading = ref(false)
const graphActionError = ref<string | null>(null)

/** Topic bands progressive reveal (digest UX). */
const DIGEST_TOPIC_BANDS_INITIAL = 3
/**
 * Max height for each topic band’s hit list (under the title / Search topic row).
 * ~65% of a typical tall stack — extra hits scroll inside the band.
 */
const DIGEST_TOPIC_BAND_HITS_MAX_HEIGHT = '10.5rem'
const topicBandsExpanded = ref(false)

watch(digest, () => {
  topicBandsExpanded.value = false
})

onActivated(() => {
  const h = dashboardNav.consumeHandoff()
  if (h?.kind === 'digest') {
    topicBandsExpanded.value = true
    void loadDigest()
  }
})

const visibleTopicBands = computed((): CorpusDigestTopicBand[] => {
  const all = digest.value?.topics ?? []
  if (topicBandsExpanded.value || all.length <= DIGEST_TOPIC_BANDS_INITIAL) {
    return all
  }
  return all.slice(0, DIGEST_TOPIC_BANDS_INITIAL)
})

const digestTopicBandsShowMoreCount = computed((): number => {
  const n = digest.value?.topics.length ?? 0
  if (topicBandsExpanded.value || n <= DIGEST_TOPIC_BANDS_INITIAL) {
    return 0
  }
  return n - DIGEST_TOPIC_BANDS_INITIAL
})

function topicBandIndex(band: CorpusDigestTopicBand): number {
  return digest.value?.topics.findIndex((b) => b.topic_id === band.topic_id) ?? -1
}

function bandCardClass(_band: CorpusDigestTopicBand): string {
  /** Same chrome for every column — first band used ``bg-elevated`` which reads as flat gray vs peers. */
  return 'flex min-h-0 min-w-0 flex-col rounded border border-border bg-surface p-1.5'
}

function bandTitleClass(band: CorpusDigestTopicBand): string {
  return topicBandIndex(band) === 0
    ? 'text-sm font-bold text-surface-foreground'
    : 'text-sm font-semibold text-surface-foreground'
}

function topicHitSimilarityUi(h: CorpusDigestTopicHit) {
  if (h.score == null) {
    return null
  }
  return digestTopicHitSimilarityDisplay(h.score)
}

const digestLoadGate = new StaleGeneration()
const digestCatalogGate = new StaleGeneration()
/** ``Open in graph`` from topic rows: stale loads must not focus/highlight after a newer click. */
const digestGraphOpenGate = new StaleGeneration()

/** Same source as Library feed list — human titles when metadata rows only have opaque ``feed_id``. */
const feedsCatalog = ref<CorpusFeedItem[]>([])

const feedDisplayTitleById = computed(() => {
  const m: Record<string, string> = {}
  for (const f of feedsCatalog.value) {
    const id = normalizeFeedIdForViewer(f.feed_id)
    const t = f.display_title?.trim()
    if (id && t) {
      m[id] = t
    }
  }
  return m
})

function episodeFeedLabel(row: CorpusDigestRow | CorpusDigestTopicHit): string {
  return digestRowFeedLabelWithCatalog(row, feedDisplayTitleById.value)
}

function episodeRowFeedHoverTitle(row: CorpusDigestRow | CorpusDigestTopicHit): string {
  return feedNameHoverWithCatalogLookup(row, feedsCatalog.value, normalizeFeedIdForViewer)
}

function episodeFeedInlineVisible(row: CorpusDigestRow | CorpusDigestTopicHit): boolean {
  const fid = row.feed_id?.trim()
  if (fid) {
    return true
  }
  const lab = episodeFeedLabel(row)
  return lab !== 'Unknown feed' && Boolean(lab.trim())
}

/**
 * #674 item 1 — clicking a feed name in a Digest row scopes the
 * Library tab to that feed and switches main tab. Library consumes
 * the handoff via ``dashboardNav.consumeLibraryHandoffIfPending()``
 * on activation.
 */
function onClickDigestFeedName(row: CorpusDigestRow | CorpusDigestTopicHit): void {
  const fid = row.feed_id?.trim()
  if (!fid) {
    return
  }
  dashboardNav.setHandoff({ kind: 'library', feedId: fid })
  emit('switch-main-tab', 'library')
}

/** Match Library episode list topic pills (visible cap; full string on `title`). */
const RECENT_TOPIC_PILL_CHARS = 24

function digestRecentCilPills(row: CorpusDigestRow) {
  const raw = row.cil_digest_topics
  return raw?.length ? raw : []
}

/** Open graph from a **Digest Recent** CIL topic pill (``topic:`` id). */
async function openDigestRecentTopicPillInGraph(
  row: CorpusDigestRow,
  pillIndex: number,
): Promise<void> {
  const seq = digestGraphOpenGate.bump()
  graphActionError.value = null
  const paths: string[] = []
  if (row.has_gi) {
    paths.push((row.gi_relative_path || '').trim())
  }
  if (row.has_kg) {
    paths.push((row.kg_relative_path || '').trim())
  }
  const cleaned = paths.filter(Boolean)
  if (cleaned.length === 0) {
    graphActionError.value = 'No GI/KG artifacts on disk for this episode.'
    return
  }
  const pill = row.cil_digest_topics?.[pillIndex]

  await ensureDefaultCorpusGraphIfNeeded()
  await artifacts.appendRelativeArtifacts(cleaned)
  if (digestGraphOpenGate.isStale(seq)) {
    return
  }
  graphNav.clearLibraryEpisodeHighlights()
  const plan = graphFocusPlanFromCilPill(pill, row.episode_id)
  applyGraphFocusPlan(graphNav, plan)
  const eid = row.episode_id?.trim()
  if (
    eid &&
    (plan.kind === 'episode_only' || (plan.kind === 'topic' && plan.fallback))
  ) {
    graphNav.setLibraryEpisodeHighlights([eid])
  }
  emit('switch-main-tab', 'graph')
}

/** Distinct from **Recent** digest cards (same episode can appear in both lists). */
function topicHitAriaLabel(h: CorpusDigestTopicHit): string {
  const base = `Open graph and episode details: ${h.episode_title || '(episode)'}, ${episodeFeedLabel(h)}`
  const sim = topicHitSimilarityUi(h)
  return sim ? `${base}, ${sim.label}` : base
}

/** Publish / E# / duration for topic-band hit native ``title`` (no longer under the cover). */
function topicHitTimingTooltipLines(h: CorpusDigestTopicHit): string[] {
  const lines: string[] = []
  const pd = h.publish_date?.trim()
  if (pd) {
    lines.push(`Published: ${pd}`)
  }
  if (h.episode_number != null) {
    lines.push(`Episode: E${h.episode_number}`)
  }
  const dur = formatDurationSeconds(h.duration_seconds)
  if (dur) {
    lines.push(`Duration: ${dur}`)
  }
  return lines
}

/** Native ``title`` for topic-band hit rows: readable lines only (no paths or opaque ids). */
function topicHitHoverNativeTitle(h: CorpusDigestTopicHit): string {
  const lines: string[] = []
  lines.push(...topicHitTimingTooltipLines(h))
  if (h.score != null) {
    // Similarity tier + raw score live on the row; keep native row title for catalog hints only.
  }
  if (episodeFeedInlineVisible(h)) {
    lines.push(`Feed: ${episodeFeedLabel(h)}`)
  }
  const desc = h.feed_description?.trim()
  if (desc) {
    const clipped = desc.length > 220 ? `${desc.slice(0, 217)}…` : desc
    lines.push(`About this feed: ${clipped}`)
  }
  const rss = h.feed_rss_url?.trim()
  if (rss) {
    lines.push(`RSS: ${rss}`)
  }
  if (lines.length === 0) {
    return ''
  }
  return lines.join('\n')
}

/** Same accessible name pattern as Library episode rows (`episode_title`, `feed`). */
function digestCardAriaLabel(row: CorpusDigestRow): string {
  return `${row.episode_title}, ${episodeFeedLabel(row)}`
}

/** Match Library list: `bg-overlay` when this episode is open in the Episode subject rail. */
function isDigestRowSelected(row: CorpusDigestRow): boolean {
  const p = row.metadata_relative_path?.trim()
  const cur = subject.episodeMetadataPath?.trim()
  return Boolean(p && cur && p === cur)
}

function isTopicHitSelected(h: CorpusDigestTopicHit): boolean {
  const p = h.metadata_relative_path?.trim()
  const cur = subject.episodeMetadataPath?.trim()
  return Boolean(p && cur && p === cur)
}

function topicsReasonLabel(code: string | null): string {
  if (!code) {
    return ''
  }
  if (code === 'no_index') {
    return 'Semantic topics need a vector index for this corpus. Rows below still work.'
  }
  return code
}

async function loadFeedsCatalog(): Promise<void> {
  const seq = digestCatalogGate.bump()
  const root = shell.corpusPath.trim()
  if (!root || !shell.healthStatus || !shell.corpusLibraryApiAvailable) {
    feedsCatalog.value = []
    return
  }
  try {
    const body = await fetchCorpusFeeds(root)
    if (digestCatalogGate.isStale(seq)) {
      return
    }
    feedsCatalog.value = body.feeds
  } catch {
    if (digestCatalogGate.isStale(seq)) {
      return
    }
    feedsCatalog.value = []
  }
}

function digestCoversMetadataPath(path: string): boolean {
  const d = digest.value
  if (!d) {
    return false
  }
  const p = path.trim()
  for (const r of d.rows) {
    if ((r.metadata_relative_path ?? '').trim() === p) {
      return true
    }
  }
  for (const band of d.topics) {
    for (const h of band.hits) {
      if ((h.metadata_relative_path ?? '').trim() === p) {
        return true
      }
    }
  }
  return false
}

async function loadDigest(): Promise<void> {
  const seq = digestLoadGate.bump()
  error.value = null
  digest.value = null
  const root = shell.corpusPath.trim()
  if (!root || !shell.healthStatus || !shell.corpusDigestApiAvailable) {
    return
  }
  const s = sinceYmd.value.trim()
  if (s && !/^\d{4}-\d{2}-\d{2}$/.test(s)) {
    error.value = 'Enter a publish date as YYYY-MM-DD (or clear the field for all time).'
    return
  }
  loading.value = true
  try {
    const digestWindow = s ? 'since' : 'all'
    const d = await fetchCorpusDigest(root, {
      window: digestWindow,
      since: s || undefined,
      includeTopics: true,
    })
    if (digestLoadGate.isStale(seq)) {
      return
    }
    digest.value = d
    const path = subject.episodeMetadataPath?.trim()
    if (path && !digestCoversMetadataPath(path)) {
      subject.clearSubject()
    }
  } catch (e) {
    if (digestLoadGate.isStale(seq)) {
      return
    }
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    if (digestLoadGate.isCurrent(seq)) {
      loading.value = false
    }
  }
}

/** Same lower bound as Library / digest date field (omit when all time). */
function corpusSinceForSearchEmit(): string | undefined {
  const s = sinceYmd.value.trim()
  if (s && /^\d{4}-\d{2}-\d{2}$/.test(s)) {
    return s
  }
  return undefined
}

function searchTopic(band: CorpusDigestTopicBand): void {
  const since = corpusSinceForSearchEmit()
  emit('focus-search', {
    feed: '',
    query: band.query.trim(),
    ...(since ? { since } : {}),
  })
}

function openRowInLibrary(row: CorpusDigestRow): void {
  const p = row.metadata_relative_path?.trim()
  if (!p) {
    return
  }
  emit('open-library-episode', { metadata_relative_path: p })
}

const digestRecentListScrollRef = ref<HTMLElement | null>(null)
const DIGEST_RECENT_ROW_SELECTOR = '[data-digest-recent-row]'

function onDigestRecentRowKeydown(e: KeyboardEvent, index: number): void {
  const rows = digest.value?.rows
  if (!rows?.length) {
    return
  }
  handleVerticalListArrowKeydown(e, index, {
    itemCount: rows.length,
    scrollRoot: digestRecentListScrollRef.value,
    rowSelector: DIGEST_RECENT_ROW_SELECTOR,
    activateIndex: (i) => openRowInLibrary(rows[i]!),
  })
}

function onTopicHitRowActivate(h: CorpusDigestTopicHit, band: CorpusDigestTopicBand): void {
  const p = h.metadata_relative_path?.trim()
  if (p) {
    emit('open-library-episode', { metadata_relative_path: p })
    void openTopicHitInGraph(h, { graphTopicNodeId: band.graph_topic_id?.trim() })
  }
}

async function openTopicHitInGraph(
  h: CorpusDigestTopicHit,
  opts?: { graphTopicNodeId?: string | null },
): Promise<void> {
  const seq = digestGraphOpenGate.bump()
  graphActionError.value = null
  const paths: string[] = []
  if (h.has_gi) {
    paths.push((h.gi_relative_path || '').trim())
  }
  if (h.has_kg) {
    paths.push((h.kg_relative_path || '').trim())
  }
  const cleaned = paths.filter(Boolean)
  if (cleaned.length === 0) {
    graphActionError.value = 'No GI/KG artifacts on disk for this episode.'
    return
  }
  await ensureDefaultCorpusGraphIfNeeded()
  await artifacts.appendRelativeArtifacts(cleaned)
  if (digestGraphOpenGate.isStale(seq)) {
    return
  }
  graphNav.clearLibraryEpisodeHighlights()
  const topicFocus = opts?.graphTopicNodeId?.trim()
  const eid = h.episode_id?.trim()
  if (topicFocus && eid) {
    graphNav.requestFocusNode(topicFocus, eid)
    graphNav.setLibraryEpisodeHighlights([eid])
  } else if (topicFocus) {
    graphNav.requestFocusNode(topicFocus)
  } else if (eid) {
    graphNav.requestFocusNode(eid)
    graphNav.setLibraryEpisodeHighlights([eid])
  } else {
    graphNav.clearPendingFocus()
  }
  emit('switch-main-tab', 'graph')
}

async function openTopicBandInGraph(band: CorpusDigestTopicBand): Promise<void> {
  const gid = band.graph_topic_id?.trim()
  for (const h of band.hits) {
    if (h.has_gi || h.has_kg) {
      await openTopicHitInGraph(h, { graphTopicNodeId: gid })
      if (gid) {
        subject.focusTopic(gid)
      }
      return
    }
  }
  graphActionError.value = "No GI/KG artifacts for this topic's hits."
}

watch(
  () =>
    [
      shell.corpusPath,
      shell.healthStatus,
      shell.corpusLibraryApiAvailable,
      shell.corpusDigestApiAvailable,
    ] as const,
  () => {
    digestGraphOpenGate.invalidate()
    void loadFeedsCatalog()
    void loadDigest()
  },
  { immediate: true },
)

watch(sinceYmd, () => {
  scheduleLoadDigestFromCorpusLens()
})

onBeforeUnmount(() => {
  if (digestLensDebounceTimer) {
    clearTimeout(digestLensDebounceTimer)
    digestLensDebounceTimer = null
  }
})
</script>

<template>
  <div
    class="flex h-full min-h-[280px] flex-col gap-2 overflow-y-auto p-3 text-surface-foreground"
    data-testid="digest-root"
  >
    <div class="flex flex-col gap-1.5">
      <div class="flex w-full min-w-0 flex-col gap-2">
        <div class="flex min-w-0 items-center gap-1.5">
          <h2
            id="digest-main-heading"
            class="text-sm font-semibold text-surface-foreground"
          >
            Digest
          </h2>
          <HelpTip
            class="shrink-0"
            :pref-width="320"
            button-aria-label="About Digest"
          >
            <p class="mb-2 font-sans text-[11px] font-semibold text-surface-foreground">
              Digest
            </p>
            <p class="font-sans text-[10px] leading-snug text-muted">
              <strong class="font-medium text-surface-foreground/90">Topic bands</strong>
              — click the
              <strong class="font-medium text-surface-foreground/90">topic title</strong>
              to open the
              <strong class="font-medium text-surface-foreground/90">Graph</strong>
              for the top GI/KG hit. Click a
              <strong class="font-medium text-surface-foreground/90">hit row</strong>
              to load the
              <strong class="font-medium text-surface-foreground/90">Episode</strong>
              panel on the right, then open the
              <strong class="font-medium text-surface-foreground/90">Graph</strong>
              and focus the
              <strong class="font-medium text-surface-foreground/90">digest topic node</strong>
              when it exists in the slice (otherwise the episode node).
              <strong class="font-medium text-surface-foreground/90">Search topic</strong>
              prefills semantic search. Each hit row shows a semantic
              <strong class="font-medium text-surface-foreground/90">match strength</strong>
              when a score exists (hover the label for the raw value). Hover the row for publish
              date, episode number, duration, feed name, and catalog hints (RSS and a short feed
              description when available).
              <strong class="font-medium text-surface-foreground/90">Recent</strong>
              — row click opens the
              <strong class="font-medium text-surface-foreground/90">Episode</strong>
              rail; when the API sends CIL topics,
              <strong class="font-medium text-surface-foreground/90">topic pills</strong>
              open the graph on
              <span class="font-mono text-[9px]">topic:…</span>
              when GI/KG list that node (see
              <strong class="font-medium text-surface-foreground/90">Library</strong>
              for the same catalog without list pills).
            </p>
          </HelpTip>
        </div>
        <div
          class="flex w-full min-w-0 flex-wrap items-center gap-x-2 gap-y-1"
          data-testid="digest-toolbar-filters"
        >
          <DateChip
            :model-value="sinceYmd"
            chip-testid="digest-chip-date"
            popover-testid="digest-popover-date"
            @update:model-value="(v) => (sinceYmd = v)"
          />
        </div>
      </div>
      <p
        v-if="digest"
        class="text-left text-[10px] leading-snug text-muted"
      >
        <time :datetime="digest.window_start_utc">{{
          formatUtcDateTimeForDisplay(digest.window_start_utc)
        }}</time>
        →
        <time :datetime="digest.window_end_utc">{{
          formatUtcDateTimeForDisplay(digest.window_end_utc)
        }}</time>
        · {{ digest.rows.length }} {{ digest.rows.length === 1 ? 'episode' : 'episodes' }}
      </p>
    </div>

    <p v-if="!shell.healthStatus" class="text-sm text-muted">
      Set corpus path and ensure the API is reachable.
    </p>
    <p
      v-else-if="!shell.corpusLibraryApiAvailable"
      class="text-sm text-danger"
    >
      This API does not expose corpus catalog routes (<code class="text-xs">/api/corpus/*</code>).
      Restart from a current checkout with <code class="text-xs">.[server]</code> installed.
    </p>
    <p
      v-else-if="!shell.corpusDigestApiAvailable"
      class="text-sm text-danger"
    >
      This API build does not expose the digest endpoint (<code class="text-xs">GET /api/corpus/digest</code>).
      Upgrade the viewer API process; <strong>Library</strong> may still work if the catalog is available.
    </p>
    <p v-else-if="!shell.hasCorpusPath" class="text-sm text-muted">
      Set <strong>Corpus path</strong> in the left panel (same as List files).
    </p>
    <p v-else-if="loading && !digest" class="text-sm text-muted">
      Loading digest…
    </p>
    <p v-else-if="error" class="text-sm text-danger">
      {{ error }}
    </p>
    <template v-else-if="digest">
      <div class="flex min-h-0 flex-1 flex-col gap-2">
      <p
        v-if="digest.topics_unavailable_reason"
        class="text-xs text-muted"
      >
        {{ topicsReasonLabel(digest.topics_unavailable_reason) }}
      </p>

      <div
        v-if="digest.topics.length"
        class="min-w-0 overflow-x-hidden"
        role="region"
        aria-label="Topic bands"
      >
        <div
          class="grid gap-2 [grid-template-columns:repeat(auto-fit,minmax(min(100%,12rem),1fr))]"
        >
          <section
            v-for="band in visibleTopicBands"
            :key="band.topic_id"
            :class="bandCardClass(band)"
            :aria-label="`Topic ${band.label}`"
          >
            <div class="flex flex-wrap items-center justify-between gap-1.5">
              <button
                type="button"
                class="min-w-0 flex-1 rounded px-0.5 py-0.5 text-left hover:bg-overlay/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                :aria-label="`Open graph for topic ${band.label} (top hit with GI or KG)`"
                @click="void openTopicBandInGraph(band)"
              >
                <span :class="bandTitleClass(band)">{{
                  band.label
                }}</span>
              </button>
              <button
                type="button"
                class="shrink-0 rounded bg-primary px-2 py-0.5 text-[10px] font-medium text-primary-foreground"
                @click="searchTopic(band)"
              >
                Search topic
              </button>
            </div>
            <ul
              class="mt-0.5 min-h-0 space-y-0.5 overflow-x-hidden overflow-y-auto overscroll-y-contain text-xs [scrollbar-width:thin] [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-border"
              :style="{ maxHeight: DIGEST_TOPIC_BAND_HITS_MAX_HEIGHT }"
            >
              <li v-for="(h, hi) in band.hits" :key="hi">
                <div
                  :role="h.metadata_relative_path ? 'button' : undefined"
                  :tabindex="h.metadata_relative_path ? 0 : undefined"
                  class="grid grid-cols-[auto,minmax(0,1fr)] gap-x-1.5 gap-y-0.5 rounded px-0.5 py-px outline-none"
                  :class="[
                    h.metadata_relative_path
                      ? 'cursor-pointer ring-offset-1 hover:bg-overlay focus-visible:ring-2 focus-visible:ring-primary'
                      : '',
                    isTopicHitSelected(h) ? 'bg-overlay' : '',
                  ]"
                  :title="topicHitHoverNativeTitle(h) || undefined"
                  :aria-label="h.metadata_relative_path ? topicHitAriaLabel(h) : undefined"
                  @click="onTopicHitRowActivate(h, band)"
                  @keydown.enter.prevent="onTopicHitRowActivate(h, band)"
                  @keydown.space.prevent="onTopicHitRowActivate(h, band)"
                >
                  <div class="row-start-1 flex w-8 shrink-0 self-start">
                    <PodcastCover
                      class="shrink-0"
                      :corpus-path="shell.corpusPath"
                      :episode-image-local-relpath="h.episode_image_local_relpath"
                      :feed-image-local-relpath="h.feed_image_local_relpath"
                      :episode-image-url="h.episode_image_url"
                      :feed-image-url="h.feed_image_url"
                      :alt="`Cover for ${h.episode_title || 'episode'}`"
                      size-class="h-8 w-8 shrink-0"
                    />
                  </div>
                  <div class="row-start-1 min-w-0">
                    <div class="flex w-full min-w-0 items-start justify-between gap-2">
                      <div class="flex min-w-0 flex-1 items-start gap-1.5">
                        <span
                          v-if="isPublishDateWithin24hRolling(h.publish_date)"
                          role="img"
                          class="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-success"
                          :title="recencyDotHoverTitle(h.publish_date) || undefined"
                          :aria-label="
                            recencyDotHoverTitle(h.publish_date) || 'Recently published'
                          "
                        />
                        <span
                          class="min-w-0 break-words font-medium text-surface-foreground"
                        >{{ h.episode_title || '(episode)' }}</span>
                      </div>
                      <template
                        v-for="sim in [topicHitSimilarityUi(h)]"
                        :key="'digest-topic-sim-' + band.topic_id + '-' + hi"
                      >
                        <span
                          v-if="sim"
                          class="shrink-0 text-[10px] leading-tight"
                          :class="sim.labelClass"
                          :title="sim.rawTitle"
                        >{{ sim.label }}</span>
                      </template>
                    </div>
                    <p
                      v-if="digestRowSummaryPreview(h)"
                      class="mt-0.5 min-w-0 break-words text-[10px] leading-snug text-muted"
                      :class="isTopicHitSelected(h) ? '' : 'line-clamp-1'"
                      :title="
                        isTopicHitSelected(h)
                          ? undefined
                          : digestRowSummaryPreview(h) || undefined
                      "
                    >
                      {{ digestRowSummaryPreview(h) }}
                    </p>
                  </div>
                </div>
              </li>
            </ul>
          </section>
        </div>
        <button
          v-if="digestTopicBandsShowMoreCount > 0"
          type="button"
          class="mt-2 text-left text-xs text-primary underline decoration-transparent underline-offset-2 hover:decoration-primary"
          data-testid="digest-topic-bands-show-more"
          @click="topicBandsExpanded = true"
        >
          Show {{ digestTopicBandsShowMoreCount }} more topics
        </button>
      </div>

      <!-- Match Library: collapsible/filters above, then `flex min-h-0 flex-1 flex-col gap-2` + Episodes panel. -->
      <div class="flex min-h-0 flex-1 flex-col gap-2">
      <!-- Same panel chrome + list structure as Library **Episodes** (`LibraryView.vue`). -->
      <div
        class="flex min-h-72 min-w-0 flex-1 flex-col rounded border border-border bg-surface lg:min-h-0"
        role="region"
        :aria-label="
          digest.rows.length === 1
            ? 'Recent episodes, 1 item'
            : `Recent episodes, ${digest.rows.length} items`
        "
      >
        <div class="border-b border-border p-2">
          <div class="flex items-center gap-1.5">
            <h2
              id="digest-recent-heading"
              class="flex flex-wrap items-baseline gap-x-1 text-sm font-semibold text-surface-foreground"
            >
              <span>Recent</span>
              <span class="font-normal tabular-nums text-muted">({{ digest.rows.length }})</span>
            </h2>
            <HelpTip
              class="shrink-0"
              :pref-width="304"
              button-aria-label="About the Recent digest list"
            >
              <p class="mb-2 font-sans text-[11px] font-semibold text-surface-foreground">
                Recent (diverse)
              </p>
              <p class="mb-2 font-sans text-[10px] text-muted">
                This list is <strong class="font-medium text-surface-foreground/90">not</strong> the same as
                <strong class="font-medium text-surface-foreground/90">topic bands</strong> above. Those run
                separate <strong class="font-medium text-surface-foreground/90">semantic searches</strong> (one per
                configured topic); the same episode can rank high in several topics.
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
                <strong class="font-medium text-surface-foreground/90">Recent</strong> is built from episodes whose
                publish date falls in the digest window, then <strong class="font-medium text-surface-foreground/90">diversified</strong>
                so one feed cannot fill the list (round-robin style with a per-feed cap and a total row cap on the server).
                It is a sample of what is new, spread across feeds — not every episode in the window.
              </p>
            </HelpTip>
          </div>
        </div>
        <div
          ref="digestRecentListScrollRef"
          class="min-h-0 flex-1 overflow-y-auto p-1"
        >
          <p
            v-if="digest.rows.length === 0"
            class="px-1 text-xs text-muted"
          >
            No episodes in this window.
          </p>
          <ul
            v-else
            class="space-y-0.5 text-sm"
          >
            <li
              v-for="(row, ri) in digest.rows"
              :key="row.metadata_relative_path"
            >
              <div
                role="button"
                tabindex="0"
                data-digest-recent-row
                class="flex w-full items-start gap-2 rounded px-2 py-1.5 text-left outline-none ring-offset-1 focus-visible:ring-2 focus-visible:ring-primary"
                :class="
                  isDigestRowSelected(row) ? 'bg-overlay' : 'hover:bg-overlay/35'
                "
                :aria-label="digestCardAriaLabel(row)"
                @click="openRowInLibrary(row)"
                @keydown="onDigestRecentRowKeydown($event, ri)"
                @keydown.enter.prevent="openRowInLibrary(row)"
                @keydown.space.prevent="openRowInLibrary(row)"
              >
                <PodcastCover
                  :corpus-path="shell.corpusPath"
                  :episode-image-local-relpath="row.episode_image_local_relpath"
                  :feed-image-local-relpath="row.feed_image_local_relpath"
                  :episode-image-url="row.episode_image_url"
                  :feed-image-url="row.feed_image_url"
                  :alt="`Cover for ${row.episode_title}`"
                  size-class="h-9 w-9 shrink-0"
                />
                <div class="min-w-0 flex-1">
                  <div class="flex flex-col gap-1 sm:flex-row sm:items-start sm:justify-between sm:gap-2">
                    <div
                      class="flex min-w-0 items-start gap-1.5 sm:max-w-[min(100%,24rem)] sm:flex-1"
                    >
                      <span
                        v-if="isPublishDateWithin24hRolling(row.publish_date)"
                        role="img"
                        class="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-success"
                        :title="recencyDotHoverTitle(row.publish_date) || undefined"
                        :aria-label="
                          recencyDotHoverTitle(row.publish_date) || 'Recently published'
                        "
                      />
                      <span
                        class="min-w-0 break-words font-medium text-surface-foreground"
                      >{{ row.episode_title }}</span>
                    </div>
                    <div
                      class="flex min-w-0 max-w-full shrink-0 flex-wrap items-baseline justify-end gap-x-1.5 gap-y-0.5 text-[10px] sm:max-w-[min(100%,20rem)]"
                    >
                      <button
                        v-if="episodeFeedInlineVisible(row)"
                        type="button"
                        class="min-w-0 break-words text-left font-semibold leading-tight text-surface-foreground hover:text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                        :title="episodeRowFeedHoverTitle(row) || undefined"
                        :aria-label="`Open Library scoped to feed ${episodeFeedLabel(row)}`"
                        data-testid="digest-feed-name-link"
                        @click.stop="onClickDigestFeedName(row)"
                      >
                        {{ episodeFeedLabel(row) }}
                      </button>
                      <span
                        v-if="
                          row.publish_date ||
                            row.episode_number != null ||
                            formatDurationSeconds(row.duration_seconds)
                        "
                        class="inline-flex shrink-0 flex-wrap items-baseline gap-x-1.5 tabular-nums leading-tight text-muted"
                      >
                        <span v-if="row.publish_date">{{ row.publish_date }}</span>
                        <span v-if="row.episode_number != null">E{{ row.episode_number }}</span>
                        <span v-if="formatDurationSeconds(row.duration_seconds)">{{
                          formatDurationSeconds(row.duration_seconds)
                        }}</span>
                      </span>
                    </div>
                  </div>
                  <p
                    v-if="libraryEpisodeSummaryLine(row)"
                    class="mt-0.5 break-words text-[11px] leading-relaxed text-muted"
                    :class="isDigestRowSelected(row) ? 'whitespace-pre-wrap' : 'line-clamp-2'"
                    :title="
                      isDigestRowSelected(row)
                        ? undefined
                        : libraryEpisodeSummaryLine(row) || undefined
                    "
                  >
                    {{ libraryEpisodeSummaryLine(row) }}
                  </p>
                  <CilTopicPillsRow
                    v-if="digestRecentCilPills(row).length"
                    class="mt-0.5"
                    cluster-member-appearance="kg"
                    :pills="digestRecentCilPills(row)"
                    :max-pill-chars="RECENT_TOPIC_PILL_CHARS"
                    truncation="wrap"
                    max-width-class="auto"
                    data-testid="digest-recent-cil-pills"
                    @pill-click="(i) => void openDigestRecentTopicPillInGraph(row, i)"
                  />
                </div>
              </div>
            </li>
          </ul>
        </div>
      </div>
      </div>
      </div>
      <p v-if="graphActionError" class="text-xs text-danger">
        {{ graphActionError }}
      </p>
    </template>
  </div>
</template>
