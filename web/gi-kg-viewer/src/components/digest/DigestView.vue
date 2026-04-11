<script setup lang="ts">
import { computed, ref, watch } from 'vue'
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
import { useEpisodeRailStore } from '../../stores/episodeRail'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { useShellStore } from '../../stores/shell'
import {
  digestRowFeedLabelWithCatalog,
  digestRowSummaryPreview,
} from '../../utils/digestRowDisplay'
import { feedNameHoverWithCatalogLookup } from '../../utils/feedHoverTitle'
import { formatDurationSeconds } from '../../utils/formatDuration'
import { formatUtcDateTimeForDisplay } from '../../utils/formatting'
import { normalizeFeedIdForViewer } from '../../utils/feedId'

const emit = defineEmits<{
  'switch-main-tab': [tab: 'library' | 'graph' | 'dashboard']
  'focus-search': [payload: { feed: string; query: string; since?: string }]
  'open-library-episode': [payload: { metadata_relative_path: string }]
}>()

const shell = useShellStore()
const artifacts = useArtifactsStore()
const graphNav = useGraphNavigationStore()
const episodeRail = useEpisodeRailStore()

const windowChoice = ref<'24h' | '7d' | 'since'>('7d')
const sinceInput = ref('')
const digest = ref<CorpusDigestResponse | null>(null)
const error = ref<string | null>(null)
const loading = ref(false)
const graphActionError = ref<string | null>(null)

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

/** Shown on hover for topic-hit similarity numbers (vector search). */
const TOPIC_HIT_SCORE_TOOLTIP =
  'Search similarity from the corpus vector index for this topic query (higher = closer match). ' +
  'Depends on the embedding model; use only to rank hits within this digest, not across runs or models.'

/** Distinct from **Recent** digest cards (same episode can appear in both lists). */
function topicHitAriaLabel(h: CorpusDigestTopicHit): string {
  return `${h.episode_title || '(episode)'}, ${episodeFeedLabel(h)} — topic band hit`
}

function digestCardHoverTitle(row: CorpusDigestRow): string {
  const parts: string[] = []
  const label = episodeFeedLabel(row)
  const fid = row.feed_id?.trim()
  if (label && label !== 'Unknown feed') {
    parts.push(fid && label !== fid ? `${label} (${fid})` : label)
  } else if (fid) {
    parts.push(`Feed: ${fid}`)
  }
  return parts.join(' · ')
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
  const root = shell.corpusPath.trim()
  if (!root || !shell.healthStatus || !shell.corpusLibraryApiAvailable) {
    feedsCatalog.value = []
    return
  }
  try {
    const body = await fetchCorpusFeeds(root)
    feedsCatalog.value = body.feeds
  } catch {
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
  error.value = null
  digest.value = null
  const root = shell.corpusPath.trim()
  if (!root || !shell.healthStatus || !shell.corpusDigestApiAvailable) {
    return
  }
  const w = windowChoice.value
  if (w === 'since' && !/^\d{4}-\d{2}-\d{2}$/.test(sinceInput.value.trim())) {
    error.value = 'Enter a since date as YYYY-MM-DD.'
    return
  }
  loading.value = true
  try {
    digest.value = await fetchCorpusDigest(root, {
      window: w,
      since: w === 'since' ? sinceInput.value.trim() : undefined,
      includeTopics: true,
    })
    const path = episodeRail.metadataRelativePath?.trim()
    if (path && !digestCoversMetadataPath(path)) {
      episodeRail.clearEpisodeContext()
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

/** YYYY-MM-DD from digest ``window_start_utc`` (ISO) for Search ``since`` filter. */
function digestWindowStartDateForSearch(): string | undefined {
  const raw = digest.value?.window_start_utc?.trim()
  if (!raw || raw.length < 10) {
    return undefined
  }
  const day = raw.slice(0, 10)
  return /^\d{4}-\d{2}-\d{2}$/.test(day) ? day : undefined
}

function searchTopic(band: CorpusDigestTopicBand): void {
  const since = digestWindowStartDateForSearch()
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

function openTopicHitInLibrary(h: CorpusDigestTopicHit): void {
  const p = h.metadata_relative_path?.trim()
  if (!p) {
    return
  }
  emit('open-library-episode', { metadata_relative_path: p })
}

function onTopicHitRowActivate(h: CorpusDigestTopicHit): void {
  if (h.metadata_relative_path?.trim()) {
    openTopicHitInLibrary(h)
  }
}

async function openTopicHitInGraph(h: CorpusDigestTopicHit): Promise<void> {
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
  graphNav.clearLibraryEpisodeHighlights()
  const eid = h.episode_id?.trim()
  if (eid) {
    graphNav.requestFocusNode(eid)
    graphNav.setLibraryEpisodeHighlights([eid])
  } else {
    graphNav.clearPendingFocus()
  }
  emit('switch-main-tab', 'graph')
}

async function openTopicBandInGraph(band: CorpusDigestTopicBand): Promise<void> {
  for (const h of band.hits) {
    if (h.has_gi || h.has_kg) {
      await openTopicHitInGraph(h)
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
    void loadFeedsCatalog()
    void loadDigest()
  },
  { immediate: true },
)

watch(windowChoice, () => {
  void loadDigest()
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
        <label class="text-[10px] font-medium text-muted">Window</label>
        <select
          v-model="windowChoice"
          class="rounded border border-border bg-elevated px-2 py-1 text-xs text-surface-foreground"
          aria-label="Digest time window"
        >
          <option value="24h">
            Last 24 hours
          </option>
          <option value="7d">
            Last 7 days
          </option>
          <option value="since">
            Since date…
          </option>
        </select>
        <input
          v-if="windowChoice === 'since'"
          v-model="sinceInput"
          type="text"
          placeholder="YYYY-MM-DD"
          class="w-32 rounded border border-border bg-elevated px-2 py-1 text-xs"
          aria-label="Since date"
          @keydown.enter="loadDigest()"
        >
        <button
          v-if="windowChoice === 'since'"
          type="button"
          class="rounded bg-primary px-2 py-1 text-[10px] font-medium text-primary-foreground"
          @click="loadDigest()"
        >
          Load
        </button>
        <button
          type="button"
          class="rounded border border-border bg-surface px-2 py-1 text-xs hover:bg-overlay"
          @click="emit('switch-main-tab', 'library')"
        >
          Open Library
        </button>
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
      <p class="text-[10px] leading-snug text-muted">
        <strong class="font-medium text-surface-foreground/90">Topic bands</strong>
        — click the <strong class="font-medium text-surface-foreground/90">topic title</strong> to open the
        <strong class="font-medium text-surface-foreground/90">top</strong> hit in the
        <strong class="font-medium text-surface-foreground/90">Graph</strong> (focus + highlight); click a
        <strong class="font-medium text-surface-foreground/90">hit row</strong> to open the
        <strong class="font-medium text-surface-foreground/90">Episode</strong> panel on the right (stay on
        <strong class="font-medium text-surface-foreground/90">Digest</strong>; same as
        <strong class="font-medium text-surface-foreground/90">Recent</strong> below).
        <strong class="font-medium text-surface-foreground/90">Search topic</strong>
        prefills semantic search. Hover the score for what it means.
        <strong class="font-medium text-surface-foreground/90">Episode cards</strong>
        — click for <strong class="font-medium text-surface-foreground/90">Episode</strong> detail on the right; use
        <strong class="font-medium text-surface-foreground/90">Open in graph</strong>
        / <strong class="font-medium text-surface-foreground/90">Prefill semantic search</strong> there.
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
                :aria-label="`Open top hit for topic ${band.label} in graph`"
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
                  @click="onTopicHitRowActivate(h)"
                  @keydown.enter.prevent="onTopicHitRowActivate(h)"
                  @keydown.space.prevent="onTopicHitRowActivate(h)"
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

      <div
        class="mt-4 border-t border-border pt-3"
        role="region"
        aria-label="Recent episodes"
      >
        <div class="flex items-center gap-1.5">
          <h3 class="text-xs font-semibold text-surface-foreground">
            Recent
          </h3>
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
            <p class="font-sans text-[10px] text-muted">
              <strong class="font-medium text-surface-foreground/90">Recent</strong> is built from episodes whose
              publish date falls in the digest window, then <strong class="font-medium text-surface-foreground/90">diversified</strong>
              so one feed cannot fill the list (round-robin style with a per-feed cap and a total row cap on the server).
              It is a sample of what is new, spread across feeds — not every episode in the window.
            </p>
          </HelpTip>
        </div>
        <p v-if="digest.rows.length === 0" class="mt-2 text-sm text-muted">
          No episodes in this window.
        </p>
        <ul v-else class="mt-2 space-y-2">
          <li
            v-for="row in digest.rows"
            :key="row.metadata_relative_path"
            class="overflow-hidden rounded border border-border bg-elevated"
          >
            <div
              role="button"
              tabindex="0"
              class="cursor-pointer px-2 py-1.5 outline-none ring-offset-1 hover:bg-overlay focus-visible:ring-2 focus-visible:ring-primary"
              :class="isDigestRowSelected(row) ? 'bg-overlay' : ''"
              :title="digestCardHoverTitle(row)"
              :aria-label="digestCardAriaLabel(row)"
              @click="openRowInLibrary(row)"
              @keydown.enter.prevent="openRowInLibrary(row)"
              @keydown.space.prevent="openRowInLibrary(row)"
            >
              <div class="flex gap-2">
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
                    v-if="digestRowSummaryPreview(row)"
                    class="mt-1 break-words whitespace-pre-wrap text-[11px] leading-snug text-muted"
                  >
                    {{ digestRowSummaryPreview(row) }}
                  </p>
                </div>
              </div>
            </div>
          </li>
        </ul>
      </div>
      <p v-if="graphActionError" class="text-xs text-danger">
        {{ graphActionError }}
      </p>
    </template>
  </div>
</template>
