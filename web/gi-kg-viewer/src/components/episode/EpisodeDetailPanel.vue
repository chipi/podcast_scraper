<script setup lang="ts">
import { computed, inject, ref, useSlots, watch } from 'vue'
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
import { copyTextToClipboard } from '../../utils/clipboard'
import CilTopicPillsRow from '../shared/CilTopicPillsRow.vue'
import DiagnosticRow from '../shared/DiagnosticRow.vue'
import EpisodeBridgePartition from './EpisodeBridgePartition.vue'
import EpisodeEnrichmentSection from './EpisodeEnrichmentSection.vue'
import HelpTip from '../shared/HelpTip.vue'
import PodcastCover from '../shared/PodcastCover.vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { themeMemberTopicIdSet } from '../../utils/topicClustersOverlay'
import { useGraphExplorerStore } from '../../stores/graphExplorer'
import { useGraphFilterStore } from '../../stores/graphFilters'
import { useGraphHandoffStore } from '../../stores/graphHandoff'
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
import { fetchEpisodeRelatedInsights, type RelatedNode } from '../../api/relationalApi'
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
    railDetailTab?: 'details' | 'enrichment' | 'neighbourhood'
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
  /**
   * Search v3 §S6 — episode-scoped rail launcher. Switches to the Search
   * main tab, sets ``search.filters.episodeId`` to the current episode,
   * and (when a query is provided) runs immediately. App-level handler
   * publishes to activeSearchContext for cross-tab reactivity.
   */
  'open-search-in-episode': [payload: { episodeId: string; query?: string }]
  'switch-main-tab': [tab: 'graph' | 'dashboard']
  /** Bubbled from EpisodeEnrichmentSection so the rail can hide the Enrichment tab. */
  'enrichment-has-content': [boolean]
}>()

// The episode rail (SubjectRail) always injects a Details/Enrichment(+Neighbourhood)
// tablist via #episode-rail-tabs. When that tablist is present the detail panels act
// as ARIA tabpanels; mounted standalone (no tablist) they render plainly. Prop-driven
// mounts that enable the neighbourhood tab are treated as tabbed too.
const slots = useSlots()
const railTabsEnabled = computed(
  () => props.railNeighbourhoodEnabled || Boolean(slots['episode-rail-tabs']),
)

const shell = useShellStore()
const artifacts = useArtifactsStore()
// Topic ids in any co-occurrence THEME cluster — teal ring on the episode pills.
const themeMemberIds = computed(() => themeMemberTopicIdSet(artifacts.themeClustersDoc))
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
const graphHandoff = useGraphHandoffStore()
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
const episodeOpenGraphGate = new StaleGeneration()

// PRD-033 FR4.3 — insights related to this episode, from the relational layer.
const relatedInsights = ref<RelatedNode[]>([])
const relatedLoading = ref(false)
const relatedError = ref<string | null>(null)
const relatedGate = new StaleGeneration()

async function loadRelatedInsights(episodeId: string): Promise<void> {
  const id = episodeId.trim()
  const root = shell.corpusPath.trim()
  if (!id || !root || !shell.healthStatus) {
    relatedInsights.value = []
    relatedError.value = null
    return
  }
  const seq = relatedGate.bump()
  relatedLoading.value = true
  relatedInsights.value = []
  relatedError.value = null
  try {
    const body = await fetchEpisodeRelatedInsights(root, id, 10)
    if (relatedGate.isStale(seq)) return
    relatedError.value = body.error ?? null
    relatedInsights.value = body.results ?? []
  } catch (e) {
    if (relatedGate.isStale(seq)) return
    relatedError.value = e instanceof Error ? e.message : String(e)
    relatedInsights.value = []
  } finally {
    if (relatedGate.isCurrent(seq)) relatedLoading.value = false
  }
}

watch(
  () => detail.value?.episode_id ?? '',
  (id) => void loadRelatedInsights(id),
  { immediate: true },
)

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
  // Best-effort (memoized): load THEME clusters so topic pills can show the teal
  // theme ring without requiring a prior graph visit.
  void artifacts.syncTopicClustersForCurrentCorpus()
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
  // P5.1 stale-gate: rapid re-clicks must supersede the prior run so we
  // never apply a stale ``focusEpisode`` / ``requestFocusNode`` after a
  // newer handoff has taken over. Matches the DigestView P4.1 pattern.
  const seq = episodeOpenGraphGate.bump()
  // F1.1 — fire FSM event synchronously at click time so the handoff is observable
  // before any await. Source `episode-panel` per decision #2; subject-external load
  // source per medium-granularity rule.
  if (episodeMeta) {
    artifacts.setLoadSource('subject-external')
    graphHandoff.handoffRequested({
      kind: 'episode',
      metadataPath: episodeMeta,
      episodeId: episodeIdForGraph || undefined,
      source: 'episode-panel',
      loadSource: 'subject-external',
      camera: { kind: 'center-on-target' },
    })
  }
  // P5.1 race fix: emit the tab switch SYNCHRONOUSLY now (before any
  // ``await``). Same rationale as the DigestView P4.1 fix — the user
  // intent to land on Graph is honored immediately, and a fast follow-up
  // click that supersedes this envelope can't be raced by a deferred emit
  // landing back on Graph after their navigation away.
  emit('switch-main-tab', 'graph')
  await ensureDefaultCorpusGraphIfNeeded()
  if (episodeOpenGraphGate.isStale(seq)) {
    return
  }
  const beforeCount = artifacts.selectedRelPaths.length
  await artifacts.appendRelativeArtifacts(paths)
  if (episodeOpenGraphGate.isStale(seq)) {
    return
  }
  graphNav.clearLibraryEpisodeHighlights()
  if (episodeMeta) {
    // #775 — resolve epCy with retry to bridge the
    // ``appendRelativeArtifacts`` → ``filteredArtifact`` (Pinia
    // computed) propagation gap. The first attempt may race the Pinia
    // re-computation (especially on the second hot-state click after
    // KG-second-wave loads). Three attempts with microtask spacing
    // gives the computed time to update without blocking on a real
    // timer. Without this: the second hot-state Library → Library
    // handoff finds no epCy → requestFocusNode never fires → only the
    // FSM ``pending`` envelope drives apply → ``finishLayoutPass``'s
    // 3-tier resolver may still miss in some race windows.
    let epCy = ''
    for (let attempt = 0; attempt < 3; attempt++) {
      const candidate =
        findEpisodeGraphNodeIdForMetadataPathOrEpisodeId(
          graphFilters.filteredArtifact,
          episodeMeta,
          episodeIdForGraph,
        ) || ''
      if (candidate) {
        epCy = candidate
        break
      }
      // Yield to the microtask queue so any pending Pinia computed
      // re-evaluation can run. ``Promise.resolve()`` is the cheapest
      // way to defer; ``await`` ensures we wait before the next read.
      await Promise.resolve()
      if (episodeOpenGraphGate.isStale(seq)) {
        return
      }
    }
    subject.focusEpisode(episodeMeta, {
      uiTitle: episodeUiTitle,
      ...(episodeIdForGraph ? { episodeId: episodeIdForGraph } : {}),
      ...(epCy ? { graphConnectionsCyId: epCy } : {}),
    })
    if (epCy) {
      graphNav.requestFocusNode(epCy)
    }
  } else {
    subject.setEpisodeUiLabel(episodeUiTitle)
  }
  // P5.1 no-op append rescue: when the second rapid click targets the
  // same artifacts already loaded by the first, ``appendRelativeArtifacts``
  // is a no-op and the natural redraw chain doesn't fire — leaving the
  // FSM stuck in a load state until the 15s wall clock. Force a
  // ``loadSelected`` follow-up only when the append was a no-op AND the
  // FSM is still in a load state. Mirrors the DigestView fix.
  if (artifacts.selectedRelPaths.length === beforeCount) {
    await new Promise<void>((r) => setTimeout(r, 600))
    if (episodeOpenGraphGate.isStale(seq)) {
      return
    }
    const stillStuck =
      graphHandoff.state === 'loading_fetch' ||
      graphHandoff.state === 'loading_bootstrap' ||
      graphHandoff.state === 'loading_merge'
    if (stillStuck) {
      await artifacts.loadSelected({ preserveExpansion: true })
    }
  }
  // (No final ``emit('switch-main-tab', 'graph')`` — fired synchronously
  // above to avoid the P5.1 cross-tab race.)
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

/**
 * Search v3 §S6 — episode-scoped rail launcher. Sets the exact
 * ``episode_id`` filter (server-side scope) so the top-k comes from
 * this episode only. Uses the summary title / bullets as the pre-filled
 * query so the user has something to run immediately (same
 * ``build_similarity_query`` shape as Similar episodes).
 */
function openSearchInEpisode(): void {
  const d = detail.value
  if (!d) return
  const ep = d.episode_id?.trim()
  if (!ep) return
  emit('open-search-in-episode', {
    episodeId: ep,
    query: buildLibrarySearchHandoffQuery(d),
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
  (curr, prev) => {
    // Guard against health-hydration transitions that race with a
    // focus-episode + loadDetail sequence — the initial mount fetchHealth
    // (or a corpus-path-triggered re-probe) can resolve AFTER the row
    // click has already populated ``detail``, and firing this reset would
    // wipe the just-loaded episode data (HANDOFF_MATRIX H1.1 class of
    // regression). Skip when the corpus path is unchanged AND the health
    // transitions FROM a not-usable state (null/''/unknown) TO 'ok' —
    // that's always a hydration, never a corpus swap. Real corpus swaps
    // and real ok→error transitions still reset the panel below.
    const prevPath = prev?.[0]
    const currPath = curr[0]
    const prevHealth = prev?.[1]
    const currHealth = curr[1]
    const pathUnchanged = (prevPath ?? '') === (currPath ?? '')
    const prevWasNotUsable =
      prevHealth == null || prevHealth === '' || prevHealth === 'unknown'
    const currIsOk = currHealth === 'ok'
    if (pathUnchanged && prevWasNotUsable && currIsOk) {
      return
    }
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
          v-show="!railTabsEnabled || railDetailTab === 'details'"
          id="episode-detail-rail-panel-details"
          class="min-h-0 flex-1 overflow-y-auto p-2"
          :role="railTabsEnabled ? 'tabpanel' : undefined"
          :aria-labelledby="
            railTabsEnabled ? 'episode-detail-rail-tab-details' : undefined
          "
          :tabindex="railTabsEnabled ? -1 : undefined"
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
          cluster-member-appearance="kg"
          :theme-member-ids="themeMemberIds"
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
        <button
          type="button"
          class="min-w-0 flex-1 rounded border border-primary px-2 py-1.5 text-center text-xs font-medium leading-snug text-primary hover:bg-primary/10 disabled:opacity-40"
          data-testid="episode-detail-search-in-episode"
          :disabled="!detail.episode_id?.trim()"
          title="Search within this episode — filters /api/search by exact episode_id (Search v3 §S6)."
          @click="openSearchInEpisode()"
        >
          Search within episode
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

      <!-- PRD-033 FR4.3 — insights related to this episode (relational layer). -->
      <div
        v-if="relatedLoading || relatedError || relatedInsights.length"
        class="mt-2 border-t border-border pt-2"
        role="region"
        aria-label="Related insights"
        data-testid="episode-related-insights"
      >
        <strong class="text-xs font-medium text-surface-foreground/90">Related insights</strong>
        <p
          v-if="relatedLoading"
          data-testid="episode-related-insights-loading"
          class="mt-1 text-xs text-muted"
        >
          Loading…
        </p>
        <p
          v-else-if="relatedError"
          class="mt-1 text-xs text-warning"
        >
          {{ relatedError }}
        </p>
        <ul
          v-else
          class="mt-1 space-y-1 text-xs"
          data-testid="episode-related-insights-list"
        >
          <li
            v-for="row in relatedInsights"
            :key="row.id"
            data-testid="episode-related-insights-row"
            class="rounded border-l-2 border-primary/40 pl-2 text-[11px] leading-snug text-muted"
            :title="row.text"
          >
            <span class="line-clamp-2">{{ row.text || row.id }}</span>
          </li>
        </ul>
      </div>

      <p v-if="graphActionError" class="mt-1 text-xs text-danger">
        {{ graphActionError }}
      </p>
        </div>
        <!-- RFC-088 chunk-9: episode-scope enrichment signals (insight density
             bars + per-episode topic-pair chips), lifted into a dedicated
             Enrichment tab (#1128 follow-up). The section hides itself when
             neither envelope is present. Shown standalone when no tablist. -->
        <div
          v-show="!railTabsEnabled || railDetailTab === 'enrichment'"
          id="episode-detail-rail-panel-enrichment"
          class="min-h-0 flex-1 overflow-y-auto p-2"
          :role="railTabsEnabled ? 'tabpanel' : undefined"
          :aria-labelledby="
            railTabsEnabled ? 'episode-detail-rail-tab-enrichment' : undefined
          "
          :tabindex="railTabsEnabled ? -1 : undefined"
        >
          <EpisodeEnrichmentSection
            :corpus-path="shell.corpusPath"
            :metadata-relpath="detail.metadata_relative_path || ''"
            @has-content="emit('enrichment-has-content', $event)"
          />
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
