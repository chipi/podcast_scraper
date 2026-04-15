<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import {
  fetchCorpusDigest,
  type CorpusDigestResponse,
  type CorpusDigestRow,
  type CorpusDigestTopicBand,
  type CorpusDigestTopicHit,
} from '../../api/digestApi'
import { fetchCorpusFeeds, type CorpusFeedItem } from '../../api/corpusLibraryApi'
import HelpTip from '../shared/HelpTip.vue'
import PodcastCover from '../shared/PodcastCover.vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { useCorpusLensStore } from '../../stores/corpusLens'
import { useEpisodeRailStore } from '../../stores/episodeRail'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { useShellStore } from '../../stores/shell'
import {
  digestRowFeedLabelWithCatalog,
  digestRowSummaryPreview,
  libraryEpisodeSummaryLine,
} from '../../utils/digestRowDisplay'
import { feedNameHoverWithCatalogLookup } from '../../utils/feedHoverTitle'
import { formatDurationSeconds } from '../../utils/formatDuration'
import { formatUtcDateTimeForDisplay } from '../../utils/formatting'
import { normalizeFeedIdForViewer } from '../../utils/feedId'
import { handleVerticalListArrowKeydown } from '../../utils/listRowArrowNav'
import { StaleGeneration } from '../../utils/staleGeneration'

const emit = defineEmits<{
  'switch-main-tab': [tab: 'graph' | 'dashboard']
  'focus-search': [payload: { feed: string; query: string; since?: string }]
  'open-library-episode': [payload: { metadata_relative_path: string }]
}>()

const shell = useShellStore()
const artifacts = useArtifactsStore()
const graphNav = useGraphNavigationStore()
const episodeRail = useEpisodeRailStore()
const corpusLens = useCorpusLensStore()
const { sinceYmd, activePreset } = storeToRefs(corpusLens)

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

function applyDigestDateNow(): void {
  if (digestLensDebounceTimer) {
    clearTimeout(digestLensDebounceTimer)
    digestLensDebounceTimer = null
  }
  void loadDigest()
}
const digest = ref<CorpusDigestResponse | null>(null)
const error = ref<string | null>(null)
const loading = ref(false)
const graphActionError = ref<string | null>(null)

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

/** Match Library episode list topic pills (visible cap; full string on `title`). */
const RECENT_TOPIC_PILL_CHARS = 24

function digestRecentTopicPills(row: CorpusDigestRow): string[] {
  const preview = row.summary_bullets_preview
  if (preview?.length) {
    return preview
  }
  return []
}

function recentTopicPillShort(label: string): string {
  const s = label.trim()
  if (s.length <= RECENT_TOPIC_PILL_CHARS) {
    return s
  }
  return `${s.slice(0, RECENT_TOPIC_PILL_CHARS - 1)}…`
}

async function openDigestRecentTopicPillInGraph(row: CorpusDigestRow, bulletIndex: number): Promise<void> {
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
  const ids = row.summary_bullet_graph_topic_ids
  const topicHint = ids?.[bulletIndex]?.trim() ?? ''

  await artifacts.loadRelativeArtifacts(cleaned)
  if (digestGraphOpenGate.isStale(seq)) {
    return
  }
  graphNav.clearLibraryEpisodeHighlights()
  const eid = row.episode_id?.trim()
  if (topicHint && eid) {
    graphNav.requestFocusNode(topicHint, eid)
    graphNav.setLibraryEpisodeHighlights([eid])
  } else if (eid) {
    graphNav.requestFocusNode(eid)
    graphNav.setLibraryEpisodeHighlights([eid])
  } else if (topicHint) {
    graphNav.requestFocusNode(topicHint)
  } else {
    graphNav.clearPendingFocus()
  }
  emit('switch-main-tab', 'graph')
}

/** Shown on hover for topic-hit similarity numbers (vector search). */
const TOPIC_HIT_SCORE_TOOLTIP =
  'Search similarity from the corpus vector index for this topic query (higher = closer match). ' +
  'Depends on the embedding model; use only to rank hits within this digest, not across runs or models.'

/** Distinct from **Recent** digest cards (same episode can appear in both lists). */
function topicHitAriaLabel(h: CorpusDigestTopicHit): string {
  return `Open graph: ${h.episode_title || '(episode)'}, ${episodeFeedLabel(h)}`
}

/** Same accessible name pattern as Library episode rows (`episode_title`, `feed`). */
function digestCardAriaLabel(row: CorpusDigestRow): string {
  return `${row.episode_title}, ${episodeFeedLabel(row)}`
}

/** Match Library list: `bg-overlay` when this episode is open in the Episode rail. */
function isDigestRowSelected(row: CorpusDigestRow): boolean {
  const p = row.metadata_relative_path?.trim()
  const cur = episodeRail.metadataRelativePath?.trim()
  return Boolean(p && cur && p === cur)
}

function isTopicHitSelected(h: CorpusDigestTopicHit): boolean {
  const p = h.metadata_relative_path?.trim()
  const cur = episodeRail.metadataRelativePath?.trim()
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
    const path = episodeRail.metadataRelativePath?.trim()
    if (path && !digestCoversMetadataPath(path)) {
      episodeRail.clearEpisodeContext()
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
  if (h.metadata_relative_path?.trim()) {
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
  await artifacts.loadRelativeArtifacts(cleaned)
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
    <div class="flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
      <p
        v-if="digest"
        class="min-w-0 flex-1 basis-full text-left text-[10px] leading-snug text-muted sm:basis-auto"
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
      <div
        class="flex flex-wrap items-center justify-end gap-2 sm:shrink-0"
        :class="digest ? '' : 'ml-auto'"
      >
        <label
          class="text-[10px] font-medium text-muted"
          for="digest-filter-since"
        >Published on or after</label>
        <input
          id="digest-filter-since"
          v-model="sinceYmd"
          type="text"
          inputmode="numeric"
          placeholder="YYYY-MM-DD"
          class="w-32 rounded border border-border bg-elevated px-2 py-1 text-xs text-surface-foreground"
          aria-label="Published on or after date"
          @keydown.enter="applyDigestDateNow()"
        >
        <div class="flex flex-wrap gap-1">
          <button
            type="button"
            class="rounded border border-border bg-surface px-2 py-0.5 text-[10px] text-surface-foreground hover:bg-overlay/40"
            :class="activePreset === 'all' ? 'ring-2 ring-primary' : ''"
            @click="corpusLens.setPreset('all')"
          >
            All time
          </button>
          <button
            type="button"
            class="rounded border border-border bg-surface px-2 py-0.5 text-[10px] text-surface-foreground hover:bg-overlay/40"
            :class="activePreset === '7' ? 'ring-2 ring-primary' : ''"
            @click="corpusLens.setPreset(7)"
          >
            7d
          </button>
          <button
            type="button"
            class="rounded border border-border bg-surface px-2 py-0.5 text-[10px] text-surface-foreground hover:bg-overlay/40"
            :class="activePreset === '30' ? 'ring-2 ring-primary' : ''"
            @click="corpusLens.setPreset(30)"
          >
            30d
          </button>
          <button
            type="button"
            class="rounded border border-border bg-surface px-2 py-0.5 text-[10px] text-surface-foreground hover:bg-overlay/40"
            :class="activePreset === '90' ? 'ring-2 ring-primary' : ''"
            @click="corpusLens.setPreset(90)"
          >
            90d
          </button>
        </div>
      </div>
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
      <p class="text-[10px] leading-snug text-muted">
        <strong class="font-medium text-surface-foreground/90">Topic bands</strong>
        — click the <strong class="font-medium text-surface-foreground/90">topic title</strong> or a
        <strong class="font-medium text-surface-foreground/90">hit row</strong> to open the
        <strong class="font-medium text-surface-foreground/90">Graph</strong> for that episode, focus the
        <strong class="font-medium text-surface-foreground/90">digest topic node</strong> when it exists in the
        slice (otherwise the episode node), and show details in the graph panel.
        <strong class="font-medium text-surface-foreground/90">Search topic</strong>
        prefills semantic search. Hover the score for what it means.
        <strong class="font-medium text-surface-foreground/90">Recent</strong>
        — row click opens the <strong class="font-medium text-surface-foreground/90">Episode</strong> rail; a
        <strong class="font-medium text-surface-foreground/90">summary topic pill</strong> opens the graph with
        a matching <span class="font-mono text-[9px]">topic:…</span> hint when GI/KG lists that node.
      </p>
      <p
        v-if="digest.topics_unavailable_reason"
        class="text-xs text-muted"
      >
        {{ topicsReasonLabel(digest.topics_unavailable_reason) }}
      </p>

      <div
        v-if="digest.topics.length"
        class="min-h-0 max-h-[min(50vh,24rem)] overflow-x-hidden overflow-y-auto rounded-sm"
        role="region"
        aria-label="Topic bands"
      >
        <div class="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
          <section
            v-for="band in digest.topics"
            :key="band.topic_id"
            class="flex min-w-0 flex-col rounded border border-border bg-surface p-1.5"
            :aria-label="`Topic ${band.label}`"
          >
            <div class="flex flex-wrap items-center justify-between gap-1.5">
              <button
                type="button"
                class="min-w-0 flex-1 rounded px-0.5 py-0.5 text-left hover:bg-overlay/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                :aria-label="`Open graph for topic ${band.label} (top hit with GI or KG)`"
                @click="void openTopicBandInGraph(band)"
              >
                <span class="text-xs font-semibold text-surface-foreground">{{
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
            <ul class="mt-0.5 space-y-0.5 text-xs">
              <li v-for="(h, hi) in band.hits" :key="hi">
                <div
                  :role="h.metadata_relative_path ? 'button' : undefined"
                  :tabindex="h.metadata_relative_path ? 0 : undefined"
                  class="flex items-start gap-1.5 rounded px-0.5 py-0.5 outline-none"
                  :class="[
                    h.metadata_relative_path
                      ? 'cursor-pointer ring-offset-1 hover:bg-overlay focus-visible:ring-2 focus-visible:ring-primary'
                      : '',
                    isTopicHitSelected(h) ? 'bg-overlay' : '',
                  ]"
                  :aria-label="h.metadata_relative_path ? topicHitAriaLabel(h) : undefined"
                  @click="onTopicHitRowActivate(h, band)"
                  @keydown.enter.prevent="onTopicHitRowActivate(h, band)"
                  @keydown.space.prevent="onTopicHitRowActivate(h, band)"
                >
                  <PodcastCover
                    :corpus-path="shell.corpusPath"
                    :episode-image-local-relpath="h.episode_image_local_relpath"
                    :feed-image-local-relpath="h.feed_image_local_relpath"
                    :episode-image-url="h.episode_image_url"
                    :feed-image-url="h.feed_image_url"
                    :alt="`Cover for ${h.episode_title || 'episode'}`"
                    size-class="h-7 w-7"
                  />
                  <div class="min-w-0 flex-1">
                    <span class="block font-medium break-words text-surface-foreground">{{
                      h.episode_title || '(episode)'
                    }}</span>
                    <p
                      v-if="digestRowSummaryPreview(h)"
                      class="mt-0.5 break-words whitespace-pre-wrap text-[11px] leading-snug text-muted"
                    >
                      {{ digestRowSummaryPreview(h) }}
                    </p>
                  </div>
                  <!-- Narrow meta column: score, date+duration line, feed (frees width for title). -->
                  <div
                    class="flex max-w-[5.75rem] shrink-0 flex-col items-end gap-0.5 text-right text-[10px] leading-tight sm:max-w-[7rem]"
                  >
                    <span
                      v-if="h.score != null"
                      class="cursor-help rounded bg-overlay px-1 py-px font-mono text-[9px] leading-none text-muted"
                      :title="TOPIC_HIT_SCORE_TOOLTIP"
                      :aria-label="`Similarity ${h.score.toFixed(3)}. ${TOPIC_HIT_SCORE_TOOLTIP}`"
                    >{{ h.score.toFixed(3) }}</span>
                    <span
                      v-if="h.publish_date || formatDurationSeconds(h.duration_seconds)"
                      class="inline-flex w-full flex-wrap items-baseline justify-end gap-x-1 tabular-nums text-muted"
                    >
                      <span v-if="h.publish_date" class="break-words">{{ h.publish_date }}</span>
                      <span v-if="formatDurationSeconds(h.duration_seconds)">{{
                        formatDurationSeconds(h.duration_seconds)
                      }}</span>
                    </span>
                    <span
                      v-if="episodeFeedInlineVisible(h)"
                      class="w-full break-words font-semibold text-surface-foreground"
                      :title="episodeRowFeedHoverTitle(h) || undefined"
                    >{{ episodeFeedLabel(h) }}</span>
                  </div>
                </div>
              </li>
            </ul>
          </section>
        </div>
      </div>

      <!-- Match Library: collapsible/filters above, then `flex min-h-0 flex-1 flex-col gap-2` + Episodes panel. -->
      <div class="flex min-h-0 flex-1 flex-col gap-2">
      <!-- Same panel chrome + list structure as Library **Episodes** (`LibraryView.vue`). -->
      <div
        class="flex min-h-52 min-w-0 flex-1 flex-col rounded border border-border bg-surface lg:min-h-0"
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
              class="flex flex-wrap items-baseline gap-x-1 text-xs font-semibold text-surface-foreground"
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
                class="flex w-full gap-2 rounded px-2 py-1.5 text-left outline-none ring-offset-1 focus-visible:ring-2 focus-visible:ring-primary"
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
                  size-class="h-9 w-9"
                />
                <div class="min-w-0 flex-1">
                  <div class="flex items-baseline justify-between gap-2">
                    <span class="min-w-0 flex-1 truncate font-medium text-surface-foreground">{{
                      row.episode_title
                    }}</span>
                    <div
                      class="flex min-w-0 max-w-[min(100%,14rem)] shrink-0 items-baseline justify-end gap-2 text-[10px] text-muted"
                    >
                      <span
                        v-if="episodeFeedInlineVisible(row)"
                        class="min-w-0 flex-1 truncate text-left font-semibold leading-tight text-surface-foreground"
                        :title="episodeRowFeedHoverTitle(row) || undefined"
                      >{{ episodeFeedLabel(row) }}</span>
                      <span
                        v-if="
                          row.publish_date ||
                            row.episode_number != null ||
                            formatDurationSeconds(row.duration_seconds)
                        "
                        class="inline-flex shrink-0 flex-wrap items-baseline justify-end gap-x-1.5 text-right tabular-nums leading-tight"
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
                    class="mt-1 break-words whitespace-pre-wrap text-[11px] leading-snug text-muted"
                  >
                    {{ libraryEpisodeSummaryLine(row) }}
                  </p>
                  <div
                    v-if="digestRecentTopicPills(row).length"
                    class="mt-1 flex flex-wrap gap-1"
                  >
                    <button
                      v-for="(t, ti) in digestRecentTopicPills(row)"
                      :key="ti"
                      type="button"
                      class="max-w-[11rem] shrink-0 truncate rounded-full border border-border bg-canvas px-1.5 py-0.5 text-[10px] font-medium text-surface-foreground hover:bg-overlay"
                      :title="t.trim() || undefined"
                      :aria-label="`Open graph for summary line as topic (topic node when present): ${t}`"
                      @click.stop="void openDigestRecentTopicPillInGraph(row, ti)"
                    >
                      {{ recentTopicPillShort(t) }}
                    </button>
                  </div>
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
