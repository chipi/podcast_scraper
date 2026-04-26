<script setup lang="ts">
import { computed, inject, ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import { fetchIndexStats, type IndexStatsEnvelope } from '../../api/indexStatsApi'
import {
  fetchCorpusEpisodeDetail,
  fetchCorpusFeeds,
  fetchCorpusSimilarEpisodes,
  type CorpusEpisodeDetailResponse,
  type CorpusFeedItem,
  type CorpusSimilarEpisodeItem,
} from '../../api/corpusLibraryApi'
import CilTopicPillsRow from '../shared/CilTopicPillsRow.vue'
import DiagnosticRow from '../shared/DiagnosticRow.vue'
import EpisodeBridgePartition from './EpisodeBridgePartition.vue'
import HelpTip from '../shared/HelpTip.vue'
import PodcastCover from '../shared/PodcastCover.vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { useGraphExplorerStore } from '../../stores/graphExplorer'
import { useGraphFilterStore } from '../../stores/graphFilters'
import { useGraphNavigationStore } from '../../stores/graphNavigation'
import { useSubjectStore } from '../../stores/subject'
import { useShellStore } from '../../stores/shell'
import { buildLibrarySearchHandoffQuery } from '../../utils/corpusSearchHandoff'
import { digestRowFeedLabelWithCatalog } from '../../utils/digestRowDisplay'
import { feedNameHoverWithCatalogLookup } from '../../utils/feedHoverTitle'
import { formatDurationSeconds } from '../../utils/formatDuration'
import { normalizeFeedIdForViewer } from '../../utils/feedId'
import {
  SEARCH_RESULT_COPY_TITLE_CHIP_CLASS,
  SEARCH_RESULT_DIAGNOSTICS_HELP_CHIP_CLASS,
  SEARCH_RESULT_EPISODE_ID_BUTTON_CLASS,
} from '../../utils/searchResultActionStyles'
import { StaleGeneration } from '../../utils/staleGeneration'
import {
  applyGraphFocusPlan,
  graphFocusPlanFromCilPill,
} from '../../utils/cilGraphFocus'
import { corpusGraphBaselineLoaderKey } from '../../corpusGraphBaseline'
import { findEpisodeGraphNodeIdForMetadataPathOrEpisodeId } from '../../utils/graphEpisodeMetadata'

const props = withDefaults(
  defineProps<{
    /** When true, **Details** / **Neighbourhood** slot sits under hero (graph rail parity). */
    railNeighbourhoodEnabled?: boolean
    railDetailTab?: 'details' | 'neighbourhood'
  }>(),
  {
    railNeighbourhoodEnabled: false,
    railDetailTab: 'details',
  },
)

const emit = defineEmits<{
  'focus-search': [
    payload: { feed: string; query: string; since?: string; feedDisplayTitle?: string },
  ]
  'switch-main-tab': [tab: 'graph' | 'dashboard']
}>()

const shell = useShellStore()
const artifacts = useArtifactsStore()
const graphExplorer = useGraphExplorerStore()
const loadCorpusGraphBaseline = inject(corpusGraphBaselineLoaderKey, null)

/** Same merged-graph load as first Graph visit — skip if corpus slice is already loaded. */
async function ensureDefaultCorpusGraphIfNeeded(): Promise<void> {
  if (!loadCorpusGraphBaseline) {
    return
  }
  if (graphExplorer.graphTabOpenedThisSession && artifacts.selectedRelPaths.length > 0) {
    return
  }
  await loadCorpusGraphBaseline()
}
const graphFilters = useGraphFilterStore()
const graphNav = useGraphNavigationStore()
const subject = useSubjectStore()
const { episodeMetadataPath: metadataRelativePath } = storeToRefs(subject)

const feeds = ref<CorpusFeedItem[]>([])
const indexStatsEnvelope = ref<IndexStatsEnvelope | null>(null)
const detail = ref<CorpusEpisodeDetailResponse | null>(null)
const detailError = ref<string | null>(null)
const detailLoading = ref(false)
const graphActionError = ref<string | null>(null)

const similarItems = ref<CorpusSimilarEpisodeItem[]>([])
const similarLoading = ref(false)
const similarError = ref<string | null>(null)
const similarQueryUsed = ref('')
const similarRanOk = ref(false)

const detailLoadGate = new StaleGeneration()

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

function feedHasVectorIndex(feedId: string): boolean {
  const env = indexStatsEnvelope.value
  if (!env?.available || !env.stats?.feeds_indexed?.length) {
    return false
  }
  return new Set(
    env.stats.feeds_indexed.map((s) => normalizeFeedIdForViewer(s)).filter(Boolean),
  ).has(normalizeFeedIdForViewer(feedId))
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
      value: 'not loaded',
    })
  }
  return rows
})

const episodeIdChipTooltip = computed((): string => {
  const d = detail.value
  const id = d?.episode_id
  if (typeof id !== 'string' || !id.trim()) {
    return ''
  }
  return (
    `Episode id (corpus-stable, from metadata / vector index): ${id.trim()}. ` +
    'Same episode across search chunks; opening Library uses the metadata file path.'
  )
})

const episodeTitleForCopy = computed(() => detail.value?.episode_title?.trim() ?? '')

type EpisodeTitleCopyUi = 'idle' | 'copied' | 'failed'

const episodeTitleCopyUi = ref<EpisodeTitleCopyUi>('idle')
let episodeTitleCopyResetTimer: ReturnType<typeof setTimeout> | null = null

function resetEpisodeTitleCopyUi(): void {
  if (episodeTitleCopyResetTimer !== null) {
    clearTimeout(episodeTitleCopyResetTimer)
    episodeTitleCopyResetTimer = null
  }
  episodeTitleCopyUi.value = 'idle'
}

watch(
  () => [metadataRelativePath.value, episodeTitleForCopy.value] as const,
  () => {
    resetEpisodeTitleCopyUi()
  },
)

async function copyTextToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    try {
      const ta = document.createElement('textarea')
      ta.value = text
      ta.setAttribute('readonly', '')
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      ta.style.left = '-9999px'
      document.body.appendChild(ta)
      ta.focus()
      ta.select()
      const ok = document.execCommand('copy')
      document.body.removeChild(ta)
      return ok
    } catch {
      return false
    }
  }
}

async function copyEpisodeTitle(): Promise<void> {
  const text = episodeTitleForCopy.value
  if (!text) return
  resetEpisodeTitleCopyUi()
  const ok = await copyTextToClipboard(text)
  episodeTitleCopyUi.value = ok ? 'copied' : 'failed'
  episodeTitleCopyResetTimer = setTimeout(() => {
    episodeTitleCopyUi.value = 'idle'
    episodeTitleCopyResetTimer = null
  }, 2000)
}

const episodeTitleCopyAriaLabel = computed((): string => {
  if (episodeTitleCopyUi.value === 'copied') return 'Copied to clipboard'
  if (episodeTitleCopyUi.value === 'failed') return 'Copy failed; try again'
  return 'Copy title'
})

const episodeTitleCopyTitleTooltip = computed((): string => episodeTitleCopyAriaLabel.value)

function detailFeedLine(): string {
  const d = detail.value
  if (!d?.feed_id?.trim()) {
    return '(no feed id)'
  }
  return digestRowFeedLabelWithCatalog({ feed_id: d.feed_id }, feedDisplayTitleById.value)
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

async function loadFeedsAndIndex(seq: number): Promise<void> {
  const root = shell.corpusPath.trim()
  if (!root || !shell.healthStatus) {
    if (detailLoadGate.isCurrent(seq)) {
      feeds.value = []
      indexStatsEnvelope.value = null
    }
    return
  }
  try {
    const body = await fetchCorpusFeeds(root)
    if (detailLoadGate.isStale(seq)) {
      return
    }
    feeds.value = body.feeds
    try {
      const env = await fetchIndexStats(root)
      if (detailLoadGate.isStale(seq)) {
        return
      }
      indexStatsEnvelope.value = env
    } catch {
      if (detailLoadGate.isStale(seq)) {
        return
      }
      indexStatsEnvelope.value = null
    }
  } catch {
    if (detailLoadGate.isStale(seq)) {
      return
    }
    feeds.value = []
    indexStatsEnvelope.value = null
  }
}

async function loadDetail(metaPath: string): Promise<void> {
  const seq = detailLoadGate.bump()
  detailError.value = null
  detail.value = null
  similarItems.value = []
  similarError.value = null
  similarQueryUsed.value = ''
  similarRanOk.value = false
  similarLoading.value = false
  const root = shell.corpusPath.trim()
  if (!root || !shell.healthStatus) {
    return
  }
  detailLoading.value = true
  try {
    await loadFeedsAndIndex(seq)
    if (detailLoadGate.isStale(seq)) {
      return
    }
    const d = await fetchCorpusEpisodeDetail(root, metaPath)
    if (detailLoadGate.isStale(seq)) {
      return
    }
    if (metadataRelativePath.value?.trim() !== metaPath) {
      return
    }
    detail.value = d
    void loadSimilarEpisodes(seq, metaPath)
  } catch (e) {
    if (detailLoadGate.isStale(seq)) {
      return
    }
    detailError.value = e instanceof Error ? e.message : String(e)
  } finally {
    if (detailLoadGate.isCurrent(seq)) {
      detailLoading.value = false
    }
  }
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

async function loadSimilarEpisodes(forSeq: number, forMetaPath: string): Promise<void> {
  similarError.value = null
  similarItems.value = []
  similarQueryUsed.value = ''
  similarRanOk.value = false
  if (detailLoadGate.isStale(forSeq)) {
    return
  }
  if (!detail.value || detail.value.metadata_relative_path !== forMetaPath) {
    return
  }
  const root = shell.corpusPath.trim()
  if (!root) {
    return
  }
  similarLoading.value = true
  try {
    const body = await fetchCorpusSimilarEpisodes(root, forMetaPath)
    if (detailLoadGate.isStale(forSeq)) {
      return
    }
    if (metadataRelativePath.value?.trim() !== forMetaPath) {
      return
    }
    similarQueryUsed.value = body.query_used || ''
    if (body.error) {
      similarError.value = mapSimilarError(body.error, body.detail)
      return
    }
    similarItems.value = body.items
    similarRanOk.value = true
  } catch (e) {
    if (detailLoadGate.isStale(forSeq)) {
      return
    }
    similarError.value = e instanceof Error ? e.message : String(e)
  } finally {
    if (detailLoadGate.isCurrent(forSeq)) {
      similarLoading.value = false
    }
  }
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
  /** Baseline + append await graph/corpus reloads; ``detail`` can swap if another fetch completes — must not focus using a stale ref after awaits. */
  const episodeMeta = detail.value.metadata_relative_path?.trim() ?? ''
  const episodeIdForGraph = detail.value.episode_id
  const episodeUiTitle = detail.value.episode_title?.trim() || null
  await ensureDefaultCorpusGraphIfNeeded()
  await artifacts.appendRelativeArtifacts(paths)
  graphNav.clearLibraryEpisodeHighlights()
  if (episodeMeta) {
    const epCy =
      findEpisodeGraphNodeIdForMetadataPathOrEpisodeId(
        graphFilters.filteredArtifact,
        episodeMeta,
        episodeIdForGraph,
      ) || ''
    subject.focusEpisode(episodeMeta, {
      uiTitle: episodeUiTitle,
      ...(epCy ? { graphConnectionsCyId: epCy } : {}),
    })
    if (epCy) {
      graphNav.requestFocusNode(epCy)
    }
  } else {
    subject.setEpisodeUiLabel(episodeUiTitle)
  }
  emit('switch-main-tab', 'graph')
}

async function openDetailCilTopicInGraph(pillIndex: number): Promise<void> {
  graphActionError.value = null
  if (!detail.value) {
    return
  }
  const pills = detail.value.cil_digest_topics ?? []
  const pill = pills[pillIndex]
  if (!pill) {
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
  const episodeIdForPlan = detail.value.episode_id
  await ensureDefaultCorpusGraphIfNeeded()
  await artifacts.appendRelativeArtifacts(paths)
  graphNav.clearLibraryEpisodeHighlights()
  const plan = graphFocusPlanFromCilPill(pill, episodeIdForPlan)
  applyGraphFocusPlan(graphNav, plan)
  const eid = typeof episodeIdForPlan === 'string' ? episodeIdForPlan.trim() : ''
  if (
    eid &&
    (plan.kind === 'episode_only' || (plan.kind === 'topic' && plan.fallback))
  ) {
    graphNav.setLibraryEpisodeHighlights([eid])
  }
  emit('switch-main-tab', 'graph')
}

function openInSearch(): void {
  if (!detail.value) {
    return
  }
  const fid = normalizeFeedIdForViewer(detail.value.feed_id)
  const catalogTitle = fid ? feedDisplayTitleById.value[fid]?.trim() : ''
  emit('focus-search', {
    feed: fid,
    query: buildLibrarySearchHandoffQuery(detail.value),
    ...(catalogTitle ? { feedDisplayTitle: catalogTitle } : {}),
  })
}

function openSimilarEpisode(row: CorpusSimilarEpisodeItem): void {
  const p = row.metadata_relative_path?.trim()
  if (!p) {
    return
  }
  subject.focusEpisode(p, { uiTitle: row.episode_title?.trim() || null })
}

watch(
  () => [shell.corpusPath, shell.healthStatus] as const,
  () => {
    detailLoadGate.invalidate()
    feeds.value = []
    indexStatsEnvelope.value = null
    detail.value = null
    detailError.value = null
    similarItems.value = []
    similarError.value = null
    similarQueryUsed.value = ''
    similarRanOk.value = false
  },
)

watch(
  metadataRelativePath,
  (p) => {
    if (p?.trim()) {
      void loadDetail(p.trim())
    } else {
      detailLoadGate.invalidate()
      detail.value = null
      detailError.value = null
      detailLoading.value = false
      similarItems.value = []
      similarError.value = null
      similarQueryUsed.value = ''
      similarRanOk.value = false
    }
  },
  { immediate: true },
)
</script>

<template>
  <div
    class="flex min-h-0 flex-1 flex-col overflow-hidden text-sm text-surface-foreground"
    data-testid="episode-detail-rail-body"
  >
    <p
      v-if="detailLoading"
      class="shrink-0 border-b border-border px-2 py-2 text-xs text-muted"
    >
      Loading…
    </p>
    <p
      v-else-if="detailError"
      class="shrink-0 border-b border-border px-2 py-2 text-xs text-danger"
    >
      {{ detailError }}
    </p>
    <p
      v-else-if="!detail"
      class="shrink-0 border-b border-border px-2 py-2 text-xs text-muted"
    >
      No episode selected.
    </p>
    <template v-else>
      <div class="shrink-0 border-b border-border px-2 py-2">
        <div class="flex min-w-0 items-start gap-3">
          <PodcastCover
            class="shrink-0"
            :corpus-path="shell.corpusPath"
            :episode-image-local-relpath="detail.episode_image_local_relpath"
            :feed-image-local-relpath="detail.feed_image_local_relpath"
            :episode-image-url="detail.episode_image_url"
            :feed-image-url="detail.feed_image_url"
            :alt="`Cover for ${detail.episode_title}`"
            size-class="h-[4.5rem] w-[4.5rem]"
          />
          <div class="min-h-0 min-w-0 flex-1">
            <div class="flex min-w-0 items-start gap-x-1.5">
              <div class="flex min-h-0 min-w-0 flex-1 flex-col gap-y-1">
                <h3
                  class="node-detail-primary-title min-w-0 select-text text-base font-semibold leading-snug text-surface-foreground"
                >
                  {{ detail.episode_title }}
                </h3>
                <div
                  v-if="
                    detail.publish_date ||
                      detail.episode_number != null ||
                      formatDurationSeconds(detail.duration_seconds) ||
                      detail.feed_id
                  "
                  class="min-w-0 text-xs text-muted"
                >
                  <div class="flex min-w-0 flex-col items-start gap-1 text-left">
                    <span
                      class="w-full min-w-0 cursor-help break-words font-medium text-surface-foreground"
                      :title="detailFeedHoverTitle() || undefined"
                    >{{ detailFeedLine() }}</span>
                    <span
                      v-if="
                        detail.publish_date ||
                          detail.episode_number != null ||
                          formatDurationSeconds(detail.duration_seconds)
                      "
                      class="inline-flex w-full min-w-0 flex-wrap items-baseline gap-x-1.5 text-left text-[10px] tabular-nums leading-tight text-muted"
                    >
                      <span v-if="detail.publish_date">{{ detail.publish_date }}</span>
                      <span v-if="detail.episode_number != null">E{{ detail.episode_number }}</span>
                      <span v-if="formatDurationSeconds(detail.duration_seconds)">{{
                        formatDurationSeconds(detail.duration_seconds)
                      }}</span>
                    </span>
                  </div>
                </div>
              </div>
              <div class="flex shrink-0 flex-col items-end gap-0.5 self-start pt-0.5">
                <button
                  v-if="detail.episode_id?.trim()"
                  type="button"
                  :class="SEARCH_RESULT_EPISODE_ID_BUTTON_CLASS"
                  :aria-label="episodeIdChipTooltip"
                  :title="episodeIdChipTooltip"
                  @click.stop.prevent
                >
                  E
                </button>
                <HelpTip
                  :pref-width="300"
                  :button-class="SEARCH_RESULT_DIAGNOSTICS_HELP_CHIP_CLASS"
                  button-aria-label="Episode and feed diagnostics"
                >
                  <p class="mb-2 font-sans text-[11px] font-semibold text-surface-foreground">
                    Troubleshooting
                  </p>
                  <p class="mb-2 font-sans text-[10px] text-muted">
                    Paths and ids for support — same data the viewer uses for search, similar
                    episodes, and graph loads.
                  </p>
                  <!--
                    #656-foundation: migrate this troubleshooting panel to
                    ``DiagnosticRow`` so the three #656 per-episode
                    diagnostics (bridge partition, pipeline-cleanup
                    counters, ad-excision) share one visual language.
                  -->
                  <dl class="space-y-0 font-mono text-[10px] leading-snug">
                    <DiagnosticRow
                      v-for="(row, di) in detailDiagnosticsEntries"
                      :key="di"
                      :label="row.label"
                      :value="row.value"
                    />
                  </dl>
                </HelpTip>
                <button
                  v-if="episodeTitleForCopy"
                  type="button"
                  :class="SEARCH_RESULT_COPY_TITLE_CHIP_CLASS"
                  :aria-label="episodeTitleCopyAriaLabel"
                  :title="episodeTitleCopyTitleTooltip"
                  data-testid="episode-detail-header-title-copy"
                  @click="copyEpisodeTitle"
                >
                  C
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
      <slot name="episode-rail-tabs" />
      <div class="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
        <div
          v-show="!railNeighbourhoodEnabled || railDetailTab === 'details'"
          id="episode-detail-rail-panel-details"
          class="min-h-0 flex-1 overflow-y-auto p-2"
          :role="railNeighbourhoodEnabled ? 'tabpanel' : undefined"
          :aria-labelledby="
            railNeighbourhoodEnabled ? 'episode-detail-rail-tab-details' : undefined
          "
          :tabindex="railNeighbourhoodEnabled ? -1 : undefined"
        >
      <p v-if="detail.summary_title" class="mt-0 text-xs font-medium text-surface-foreground">
        {{ detail.summary_title }}
      </p>
      <div v-if="(detail.cil_digest_topics ?? []).length" class="mt-2">
        <!--
          #656 Stage B: chip/pill polish. ``truncation="wrap"`` +
          ``max-width-class="auto"`` lets the pill shrink-wrap around
          post-#653 canonical labels ("oil prices", "shadow fleet")
          instead of padding to 11rem. ``max-pill-chars`` stays as a
          defensive cap for legacy corpora whose labels weren't
          backfilled yet.
        -->
        <CilTopicPillsRow
          :pills="detail.cil_digest_topics ?? []"
          :max-pill-chars="40"
          truncation="wrap"
          max-width-class="auto"
          data-testid="episode-detail-cil-pills"
          @pill-click="(i) => void openDetailCilTopicInGraph(i)"
        />
      </div>
      <!--
        #656 Stage B: per-episode bridge {gi_only, kg_only, both}
        indicator (post-#654). Component hides itself when the
        partition is missing — legacy episodes without bridge.json
        don't render an empty placeholder.
      -->
      <EpisodeBridgePartition :partition="detail.bridge_partition" />
      <p
        v-if="detail.summary_text"
        class="mt-2 whitespace-pre-wrap text-xs leading-relaxed text-muted"
      >
        {{ detail.summary_text }}
      </p>
      <div
        v-if="(detail.summary_bullets ?? []).length"
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
          <li v-for="(b, i) in detail.summary_bullets ?? []" :key="i">
            {{ b }}
          </li>
        </ul>
      </div>
      <p
        v-if="!(detail.summary_bullets ?? []).length && !detail.summary_text"
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
          Opens Search with this feed scoped and the same field order as
          <strong class="font-medium text-surface-foreground/90">Similar episodes</strong>
          (summary title + bullets, else episode title). Long titles or bullets are clipped so the
          query stays short (~480 chars max). Run Search to query the vector index.
        </HelpTip>
      </div>
      <div
        class="mt-4 border-t border-border pt-2"
        role="region"
        aria-label="Similar episodes"
        data-testid="library-similar"
      >
        <div class="flex items-center gap-1.5">
          <h4 class="text-xs font-semibold text-surface-foreground">
            Similar episodes
          </h4>
          <HelpTip
            class="shrink-0"
            :pref-width="320"
            button-aria-label="About similar episodes"
          >
            <p class="mb-2 font-sans text-[11px] font-semibold text-surface-foreground">
              How this works
            </p>
            <p class="mb-2 font-sans text-[10px] text-muted">
              Peers come from the same
              <strong class="font-medium text-surface-foreground/90">vector index</strong> as
              semantic search. The server turns summary text from this episode into an embedding and
              returns nearest neighbors (excluding this episode).
            </p>
            <p class="mb-1 font-sans text-[10px] font-medium text-muted">
              Text embedded for this request
            </p>
            <p
              v-if="similarQueryUsed"
              class="whitespace-pre-wrap break-words font-mono text-[10px] leading-snug text-elevated-foreground"
            >
              {{ similarQueryUsed }}
            </p>
            <p v-else-if="similarLoading" class="font-sans text-[10px] text-muted">
              Loading…
            </p>
            <p v-else class="font-sans text-[10px] text-muted">
              Open this tip after results load to see the exact string the server embedded.
            </p>
          </HelpTip>
        </div>
        <p v-if="similarLoading" class="mt-1 text-xs text-muted" aria-live="polite">
          Searching similar episodes…
        </p>
        <p v-if="similarError" class="mt-1 text-xs text-danger">
          {{ similarError }}
        </p>
        <p
          v-else-if="similarRanOk && !similarLoading && similarItems.length === 0"
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
        </div>
        <div
          v-if="railNeighbourhoodEnabled && railDetailTab === 'neighbourhood'"
          id="episode-detail-rail-panel-neighbourhood"
          class="flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto"
          role="tabpanel"
          aria-labelledby="episode-detail-rail-tab-neighbourhood"
          tabindex="-1"
        >
          <slot name="episode-rail-neighbourhood" />
        </div>
      </div>
    </template>
  </div>
</template>
