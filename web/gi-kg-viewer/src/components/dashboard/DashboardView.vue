<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { CorpusManifestDocument } from '../../api/corpusMetricsApi'
import { fetchCorpusFeeds, type CorpusFeedItem } from '../../api/corpusLibraryApi'
import {
  fetchCorpusManifest,
  fetchCorpusRunsSummary,
  fetchCorpusStats,
} from '../../api/corpusMetricsApi'
import { fetchCorpusDigest, type CorpusDigestResponse } from '../../api/digestApi'
import { useArtifactsStore } from '../../stores/artifacts'
import { useIndexStatsStore } from '../../stores/indexStats'
import { useShellStore } from '../../stores/shell'
import {
  bucketGiKgMtimesByDay,
  cumulativeGiKgByDay,
  MAX_ARTIFACT_ROWS_FOR_CLIENT_TIMELINE,
  sortedMonthHistogram,
} from '../../utils/artifactMtimeBuckets'
import { normalizeFeedIdForViewer } from '../../utils/feedId'
import { formatBytes, formatUtcDateTimeForDisplay } from '../../utils/formatting'
import { computeArtifactMetrics } from '../../utils/metrics'
import CategoryLineChart from './CategoryLineChart.vue'
import MultiSeriesLineChart from './MultiSeriesLineChart.vue'
import SimpleDoughnutChart from './SimpleDoughnutChart.vue'
import StackedStageBarChart from './StackedStageBarChart.vue'
import TypeCountBarChart from './TypeCountBarChart.vue'
import VerticalBarChart from './VerticalBarChart.vue'

const shell = useShellStore()
const artifacts = useArtifactsStore()
const indexStats = useIndexStatsStore()

const corpusStats = ref<Awaited<ReturnType<typeof fetchCorpusStats>> | null>(null)
const manifestDoc = ref<CorpusManifestDocument | null>(null)
const runsSummary = ref<Awaited<ReturnType<typeof fetchCorpusRunsSummary>> | null>(null)
const feedsCatalog = ref<CorpusFeedItem[]>([])
/** Compact digest snapshot for dashboard copy (GET /api/corpus/digest?compact=true). */
const digestGlance = ref<CorpusDigestResponse | null>(null)
const digestGlanceError = ref<string | null>(null)
const dashLoading = ref(false)
const dashError = ref<string | null>(null)
/** Bumps on each refresh so stale async work cannot leave `dashLoading` stuck or clobber state. */
let dashboardRefreshGeneration = 0

/** Sub-views: pipeline execution vs content intelligence (corpus / graph / index). */
const dashboardPanel = ref<'pipeline' | 'contentIntelligence'>('pipeline')

function setDashboardPanel(panel: 'pipeline' | 'contentIntelligence'): void {
  dashboardPanel.value = panel
}

const DASH_IDX_FEEDS = '__dash_feeds_in_index'
const DASH_CAT_FEEDS = '__dash_feeds_catalog'

function parseIsoMs(iso: string | undefined | null): number | null {
  const raw = iso?.trim()
  if (!raw) {
    return null
  }
  const t = Date.parse(raw)
  return Number.isNaN(t) ? null : t
}

const artifactMetrics = computed(() => {
  const art = artifacts.displayArtifact
  if (!art) return null
  return computeArtifactMetrics(art)
})

const indexDocTypeCounts = computed(
  () => indexStats.indexEnvelope?.stats?.doc_type_counts ?? {},
)

const artifactTimelineRowCount = computed(
  () => shell.artifactList.filter((a) => a.kind === 'gi' || a.kind === 'kg').length,
)

const artifactTimelineOverCap = computed(
  () => artifactTimelineRowCount.value > MAX_ARTIFACT_ROWS_FOR_CLIENT_TIMELINE,
)

const artifactDayBuckets = computed(() => bucketGiKgMtimesByDay(shell.artifactList))

const artifactTimelineLabels = computed(() => artifactDayBuckets.value.map((b) => b.day))

const artifactTimelineValues = computed(() => artifactDayBuckets.value.map((b) => b.count))

const publishMonthSeries = computed(() => {
  const h = corpusStats.value?.publish_month_histogram ?? {}
  return sortedMonthHistogram(h)
})

const publishHistogramSum = computed(() => {
  const h = corpusStats.value?.publish_month_histogram ?? {}
  return Object.values(h).reduce((a, b) => a + b, 0)
})

const manifestFeedCounts = computed(() => {
  const feeds = manifestDoc.value?.feeds ?? []
  const out: Record<string, number> = {}
  feeds.forEach((f, i) => {
    const key = (f.stable_feed_dir?.trim() || `feed_${i + 1}`).slice(0, 48)
    out[key] = f.episodes_processed ?? 0
  })
  return out
})

const feedTitleByCatalogId = computed(() => {
  const m: Record<string, string> = {}
  for (const row of feedsCatalog.value) {
    const id = normalizeFeedIdForViewer(row.feed_id)
    if (!id) {
      continue
    }
    const t = row.display_title?.trim()
    m[id] = t && t.length ? t : id
  }
  return m
})

function hostnameFromFeedUrl(url: string | undefined): string | null {
  const u = url?.trim()
  if (!u) {
    return null
  }
  try {
    const parsed = new URL(u)
    const h = parsed.hostname.replace(/^www\./i, '')
    return h || null
  } catch {
    return null
  }
}

/** Map manifest `stable_feed_dir` → display label for chart Y axis. */
const manifestFeedLabelMap = computed(() => {
  const cat = feedTitleByCatalogId.value
  const out: Record<string, string> = {}
  for (const f of manifestDoc.value?.feeds ?? []) {
    const dir = (f.stable_feed_dir ?? '').trim()
    if (!dir) {
      continue
    }
    const fromCat = cat[dir]
    const host = hostnameFromFeedUrl(f.feed_url)
    out[dir] = fromCat ?? host ?? dir
  }
  return out
})

/** Cumulative metrics from successive `run.json` summaries (oldest → newest). */
const runCorpusGrowth = computed(() => {
  const raw = runsSummary.value?.runs ?? []
  const dated = raw.filter((r) => (r.created_at?.length ?? 0) >= 10)
  const chrono = [...dated].sort((a, b) => (a.created_at ?? '').localeCompare(b.created_at ?? ''))
  const cap = chrono.slice(-48)
  const labels = cap.map((r) => (r.created_at ?? '').slice(0, 10))
  let ep = 0
  let gi = 0
  let kg = 0
  const epR: number[] = []
  const giR: number[] = []
  const kgR: number[] = []
  for (const r of cap) {
    ep += r.episodes_scraped_total ?? 0
    gi += r.gi_artifacts_generated ?? 0
    kg += r.kg_artifacts_generated ?? 0
    epR.push(ep)
    giR.push(gi)
    kgR.push(kg)
  }
  const hasAny = epR.some((v) => v > 0) || giR.some((v) => v > 0) || kgR.some((v) => v > 0)
  if (!hasAny || labels.length === 0) {
    return null
  }
  return {
    labels,
    series: [
      {
        label: 'Episodes scraped (cumulative)',
        values: epR,
        colorToken: '--ps-primary',
      },
      { label: 'GI artifacts (cumulative)', values: giR, colorToken: '--ps-gi' },
      { label: 'KG artifacts (cumulative)', values: kgR, colorToken: '--ps-kg' },
    ],
  }
})

/** Cumulative GI vs KG artifact files by UTC write day (under artifact cap only). */
const artifactGiKgCumulative = computed(() => {
  if (artifactTimelineOverCap.value) {
    return null
  }
  const rows = cumulativeGiKgByDay(shell.artifactList)
  if (rows.length === 0) {
    return null
  }
  return {
    labels: rows.map((r) => r.day),
    series: [
      { label: 'GI files (cumulative)', values: rows.map((r) => r.gi), colorToken: '--ps-gi' },
      { label: 'KG files (cumulative)', values: rows.map((r) => r.kg), colorToken: '--ps-kg' },
    ],
  }
})

const runDurationTrend = computed(() => {
  const runs = runsSummary.value?.runs ?? []
  const usable = runs.filter(
    (r) => r.run_duration_seconds != null && (r.created_at?.length ?? 0) > 0,
  )
  const chrono = usable.slice(0, 40).reverse()
  return {
    labels: chrono.map((r) => (r.created_at ?? '').slice(0, 10)),
    values: chrono.map((r) => r.run_duration_seconds ?? 0),
  }
})

const latestRun = computed(() => runsSummary.value?.runs[0] ?? null)

const latestRunStages = computed(() => {
  const r = latestRun.value
  if (!r) return []
  return [
    { label: 'Scraping', seconds: Math.max(0, r.time_scraping_seconds ?? 0) },
    { label: 'Parsing', seconds: Math.max(0, r.time_parsing_seconds ?? 0) },
    { label: 'Normalizing', seconds: Math.max(0, r.time_normalizing_seconds ?? 0) },
    { label: 'I/O & waiting', seconds: Math.max(0, r.time_io_and_waiting_seconds ?? 0) },
  ]
})

const latestEpisodeOutcomes = computed(() => latestRun.value?.episode_outcomes ?? {})

const latestEpisodeOutcomeTotal = computed(() => {
  const o = latestEpisodeOutcomes.value
  return (o.ok ?? 0) + (o.failed ?? 0) + (o.skipped ?? 0)
})

/** One-line takeaways for chart cards (Tufte: annotate the insight). */
const insightArtifactTimeline = computed(() => {
  const buckets = artifactDayBuckets.value
  if (buckets.length === 0) {
    return undefined
  }
  let max = buckets[0]!
  for (const b of buckets) {
    if (b.count > max.count) {
      max = b
    }
  }
  if (max.count <= 0) {
    return undefined
  }
  return `Peak: ${max.day} — ${max.count.toLocaleString()} GI/KG file(s) touched that UTC day.`
})

const insightPublishMonth = computed(() => {
  const s = publishMonthSeries.value
  const catalogN = corpusStats.value?.catalog_episode_count
  const histSum = publishHistogramSum.value
  let peak = ''
  if (s.length > 0) {
    let max = s[0]!
    for (const row of s) {
      if (row.count > max.count) {
        max = row
      }
    }
    peak = `Peak month: ${max.label} — ${max.count.toLocaleString()} episodes (catalog).`
  }
  let catalogGap = ''
  if (catalogN != null && s.length > 0) {
    if (histSum !== catalogN) {
      catalogGap = ` Catalog reports ${catalogN.toLocaleString()} episodes total; bars sum to ${histSum.toLocaleString()} — the gap is episodes without a publish month in metadata.`
    } else {
      catalogGap = ` Bar heights sum to the catalog episode count (${catalogN.toLocaleString()}).`
    }
  }
  const combined = `${peak}${catalogGap}`.trim()
  return combined.length > 0 ? combined : undefined
})

const insightManifestFeeds = computed(() => {
  const c = manifestFeedCounts.value
  const map = manifestFeedLabelMap.value
  const entries = Object.entries(c).sort((a, b) => b[1] - a[1])
  if (entries.length === 0) {
    return undefined
  }
  const [k, v] = entries[0]!
  const label = map[k] ?? k
  const short = label.length > 42 ? `${label.slice(0, 41)}…` : label
  return `Top feed: ${short} — ${v.toLocaleString()} episodes processed (manifest).`
})

const insightRunDuration = computed(() => {
  const labels = runDurationTrend.value.labels
  const values = runDurationTrend.value.values
  if (labels.length === 0 || values.length === 0) {
    return undefined
  }
  let maxI = 0
  for (let i = 1; i < values.length; i += 1) {
    if ((values[i] ?? 0) > (values[maxI] ?? 0)) {
      maxI = i
    }
  }
  const sec = values[maxI] ?? 0
  return `Longest run: ${labels[maxI]} — ${sec.toLocaleString()} s wall time.`
})

const insightRunGrowth = computed(() => {
  const g = runCorpusGrowth.value
  if (!g || g.labels.length === 0) {
    return undefined
  }
  const last = g.labels.length - 1
  const ep = g.series[0]?.values[last]
  const giv = g.series[1]?.values[last]
  const kgv = g.series[2]?.values[last]
  if (ep == null) {
    return undefined
  }
  return `End of series: ${ep.toLocaleString()} episodes scraped (cum.), ${String(giv)} GI, ${String(kgv)} KG artifacts.`
})

const insightGiKgCumulative = computed(() => {
  const g = artifactGiKgCumulative.value
  if (!g || g.labels.length === 0) {
    return undefined
  }
  const last = g.labels.length - 1
  const giv = g.series[0]?.values[last]
  const kgv = g.series[1]?.values[last]
  return `Latest point: ${String(giv)} GI and ${String(kgv)} KG files on disk (cumulative).`
})

const insightPipelineStages = computed(() => {
  const stages = latestRunStages.value.filter((s) => s.seconds > 0)
  if (stages.length === 0) {
    return undefined
  }
  let max = stages[0]!
  for (const s of stages) {
    if (s.seconds > max.seconds) {
      max = s
    }
  }
  return `Slowest stage: ${max.label} — ${max.seconds.toFixed(1)} s wall time.`
})

const insightEpisodeOutcomes = computed(() => {
  const o = latestEpisodeOutcomes.value
  const total = latestEpisodeOutcomeTotal.value
  if (total <= 0) {
    return undefined
  }
  const entries = Object.entries(o)
    .filter(([, v]) => (v ?? 0) > 0)
    .sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0))
  if (entries.length === 0) {
    return undefined
  }
  const [k, v] = entries[0]!
  const n = v ?? 0
  const pct = ((n / total) * 100).toFixed(1)
  return `Dominant: ${k} — ${n.toLocaleString()} (${pct}% of this run).`
})

const insightNodeTypes = computed(() => {
  const m = artifactMetrics.value?.visualNodeTypeCounts
  if (!m) {
    return undefined
  }
  const e = Object.entries(m)
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1])
  if (e.length === 0) {
    return undefined
  }
  const [k, v] = e[0]!
  return `Most common node bucket: ${k} — ${v.toLocaleString()} nodes.`
})

const insightIndexDocTypes = computed(() => {
  const c = indexDocTypeCounts.value
  const totalVec = indexStats.indexEnvelope?.stats?.total_vectors ?? 0
  const e = Object.entries(c)
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1])
  if (e.length === 0) {
    return undefined
  }
  const [k, v] = e[0]!
  if (totalVec > 0) {
    const pct = ((v / totalVec) * 100).toFixed(1)
    return `Largest doc_type: ${k} — ${v.toLocaleString()} chunks (${pct}% of reported index vectors).`
  }
  return `Largest doc_type in index: ${k} — ${v.toLocaleString()} chunks (snapshot).`
})

const indexFreshnessBlock = computed(() => {
  const env = indexStats.indexEnvelope
  if (!env) {
    return null
  }
  if (!env.available) {
    const reason = env.reason?.replace(/_/g, ' ') ?? 'No index for this corpus.'
    return {
      headline: 'Vector index unavailable',
      lines: [] as string[],
      insight: reason,
    }
  }
  const s = env.stats
  if (!s) {
    return null
  }
  const lines: string[] = []
  if (s.last_updated) {
    lines.push(`Index last updated: ${s.last_updated}`)
  }
  if (env.artifact_newest_mtime) {
    lines.push(`Newest index-related artifact on disk: ${env.artifact_newest_mtime}`)
  }
  const idxMs = parseIsoMs(s.last_updated)
  const artMs = parseIsoMs(env.artifact_newest_mtime)
  let insight = ''
  if (idxMs != null && artMs != null) {
    const deltaH = (artMs - idxMs) / 3600000
    if (deltaH > 48) {
      insight = `Tracked artifacts are roughly ${Math.round(deltaH / 24)} day(s) newer than the index timestamp — consider rebuilding if embeddings should match latest files.`
    } else if (deltaH < -48) {
      insight =
        'Index timestamp is newer than tracked artifact mtimes (common right after a fresh index build).'
    } else {
      insight = 'Index and artifact mtimes are within a few days; snapshot looks reasonably current.'
    }
  }
  return { headline: 'Index freshness', lines, insight }
})

const indexRebuildNotes = computed((): string[] => {
  const env = indexStats.indexEnvelope
  if (!env?.available) {
    return []
  }
  const notes: string[] = []
  if (env.rebuild_in_progress) {
    notes.push('Index rebuild is running for this corpus.')
  }
  const err = env.rebuild_last_error?.trim()
  if (err) {
    notes.push(`Last rebuild error: ${err.length > 200 ? `${err.slice(0, 199)}…` : err}`)
  }
  const banner = indexStats.indexHealthBanner
  if (banner?.lines.length) {
    for (const line of banner.lines) {
      if (!notes.includes(line)) {
        notes.push(line)
      }
    }
  } else if (env.reindex_recommended) {
    notes.push(
      'Reindex recommended; open API · Data → Vector index for actions and detailed reason codes.',
    )
  }
  return notes
})

const indexFootprintRows = computed(() => {
  const env = indexStats.indexEnvelope
  const s = env?.stats
  if (!env?.available || !s) {
    return [] as { k: string; v: string }[]
  }
  return [
    { k: 'Total vectors', v: s.total_vectors.toLocaleString() },
    {
      k: 'Embedding',
      v:
        s.embedding_model || s.embedding_dim
          ? `${s.embedding_model || '—'} · dim ${s.embedding_dim}`
          : '—',
    },
    { k: 'On-disk size', v: formatBytes(s.index_size_bytes) },
    { k: 'Index path', v: env.index_path?.trim() || '—' },
  ]
})

const indexFeedsComparisonCounts = computed((): Record<string, number> | null => {
  const env = indexStats.indexEnvelope
  if (!env?.available || !env.stats || !corpusStats.value) {
    return null
  }
  return {
    [DASH_IDX_FEEDS]: env.stats.feeds_indexed.length,
    [DASH_CAT_FEEDS]: corpusStats.value.catalog_feed_count,
  }
})

const indexFeedsComparisonLabels = computed(() => ({
  [DASH_IDX_FEEDS]: 'Feeds represented in index',
  [DASH_CAT_FEEDS]: 'Distinct feeds in catalog',
}))

const insightIndexFeedsVsCatalog = computed(() => {
  const env = indexStats.indexEnvelope
  const cat = corpusStats.value?.catalog_feed_count
  if (!env?.available || !env.stats || cat == null) {
    return undefined
  }
  const nIdx = env.stats.feeds_indexed.length
  if (nIdx === cat) {
    return `Same feed count in index list and catalog (${nIdx}).`
  }
  if (nIdx < cat) {
    return `Index lists fewer feeds (${nIdx}) than the catalog (${cat}) — some feeds may lack indexed chunks yet.`
  }
  return `Index lists more feed ids (${nIdx}) than distinct catalog feeds (${cat}) — check id normalization in API · Data.`
})

const digestGlanceLine = computed(() => {
  if (digestGlanceError.value) {
    return `Digest glance failed: ${digestGlanceError.value}`
  }
  const d = digestGlance.value
  if (!d) {
    return ''
  }
  const a = formatUtcDateTimeForDisplay(d.window_start_utc)
  const b = formatUtcDateTimeForDisplay(d.window_end_utc)
  return `Digest (${d.window}, compact): ${a} → ${b} · ${d.rows.length} diversified row(s). Open the Digest tab for the full view.`
})

const showCorpusSummaryStrip = computed(
  () => shell.hasCorpusPath && Boolean(shell.healthStatus),
)

function displayCatalogSummaryField(count: number | undefined): string {
  if (!shell.corpusLibraryApiAvailable) {
    return '—'
  }
  if (dashLoading.value && corpusStats.value === null) {
    return '…'
  }
  if (count === undefined) {
    return '—'
  }
  return String(count)
}

const displaySummaryFeeds = computed(() =>
  displayCatalogSummaryField(corpusStats.value?.catalog_feed_count),
)

const displaySummaryEpisodes = computed(() =>
  displayCatalogSummaryField(corpusStats.value?.catalog_episode_count),
)

const displaySummaryTopics = computed(() =>
  displayCatalogSummaryField(corpusStats.value?.digest_topics_configured),
)

const displaySummaryInsights = computed(() => {
  if (shell.artifactsError) {
    return '—'
  }
  if (shell.artifactsLoading) {
    return '…'
  }
  return String(shell.artifactList.filter((a) => a.kind === 'gi').length)
})

async function refreshDashboardMetrics(): Promise<void> {
  const gen = (dashboardRefreshGeneration += 1)
  dashError.value = null
  corpusStats.value = null
  manifestDoc.value = null
  runsSummary.value = null
  feedsCatalog.value = []
  digestGlance.value = null
  digestGlanceError.value = null
  if (!shell.healthStatus || !shell.hasCorpusPath) {
    dashLoading.value = false
    return
  }
  dashLoading.value = true
  try {
    await shell.fetchArtifactList()
    if (gen !== dashboardRefreshGeneration) {
      return
    }
    const p = shell.corpusPath.trim()
    if (shell.corpusDigestApiAvailable) {
      try {
        const d = await fetchCorpusDigest(p, { compact: true, includeTopics: false })
        if (gen === dashboardRefreshGeneration) {
          digestGlance.value = d
        }
      } catch (e) {
        if (gen === dashboardRefreshGeneration) {
          digestGlanceError.value = e instanceof Error ? e.message : String(e)
        }
      }
    }
    if (shell.corpusLibraryApiAvailable) {
      const [stats, runs] = await Promise.all([
        fetchCorpusStats(p),
        fetchCorpusRunsSummary(p),
      ])
      if (gen !== dashboardRefreshGeneration) {
        return
      }
      corpusStats.value = stats
      runsSummary.value = runs
      try {
        const fr = await fetchCorpusFeeds(p)
        if (gen === dashboardRefreshGeneration) {
          feedsCatalog.value = fr.feeds
        }
      } catch {
        if (gen === dashboardRefreshGeneration) {
          feedsCatalog.value = []
        }
      }
      try {
        const doc = await fetchCorpusManifest(p)
        if (gen === dashboardRefreshGeneration) {
          manifestDoc.value = doc
        }
      } catch {
        if (gen === dashboardRefreshGeneration) {
          manifestDoc.value = null
        }
      }
    }
  } catch (e) {
    if (gen === dashboardRefreshGeneration) {
      dashError.value = e instanceof Error ? e.message : String(e)
    }
  } finally {
    if (gen === dashboardRefreshGeneration) {
      dashLoading.value = false
    }
  }
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
    void refreshDashboardMetrics()
  },
  { immediate: true },
)
</script>

<template>
  <div class="space-y-4">
    <p class="text-xs text-muted">
      Corpus root, catalog snapshot, graph metrics, and vector index live under
      <span class="font-medium text-surface-foreground">API · Data</span>
      in the left panel.
      <span class="font-medium text-surface-foreground">Pipeline</span>
      covers runs and throughput;
      <span class="font-medium text-surface-foreground">Content intelligence</span>
      covers catalog shape, artifacts, graph, and index. Charts refresh when the corpus path and API are available.
    </p>
    <p
      v-if="dashLoading"
      class="text-xs text-muted"
    >
      Loading corpus charts…
    </p>
    <p
      v-if="dashError"
      class="text-xs text-danger"
    >
      {{ dashError }}
    </p>

    <div
      v-if="showCorpusSummaryStrip"
      class="grid grid-cols-2 gap-3 sm:grid-cols-4"
      role="group"
      aria-label="Corpus summary counts"
    >
      <div class="rounded border border-border bg-surface p-3 text-surface-foreground">
        <div class="text-xs text-muted">
          Feeds
        </div>
        <div class="text-2xl font-semibold tabular-nums">
          {{ displaySummaryFeeds }}
        </div>
      </div>
      <div class="rounded border border-border bg-surface p-3 text-surface-foreground">
        <div class="text-xs text-muted">
          Episodes
        </div>
        <div class="text-2xl font-semibold tabular-nums">
          {{ displaySummaryEpisodes }}
        </div>
      </div>
      <div class="rounded border border-border bg-surface p-3 text-surface-foreground">
        <div class="text-xs text-muted">
          Topics
        </div>
        <div class="text-2xl font-semibold tabular-nums">
          {{ displaySummaryTopics }}
        </div>
        <p class="mt-1 text-[10px] leading-snug text-muted">
          Digest bands (server config)
        </p>
      </div>
      <div class="rounded border border-border bg-surface p-3 text-surface-foreground">
        <div class="text-xs text-muted">
          Insights
        </div>
        <div class="text-2xl font-semibold tabular-nums">
          {{ displaySummaryInsights }}
        </div>
        <p class="mt-1 text-[10px] leading-snug text-muted">
          GI JSON files (artifact list)
        </p>
      </div>
    </div>

    <div class="flex flex-col gap-2">
      <nav
        class="relative z-[1] flex flex-wrap gap-1 rounded border border-border bg-elevated p-0.5 text-xs font-medium"
        role="tablist"
        aria-label="Dashboard sections"
      >
        <button
          id="dash-tab-pipeline"
          type="button"
          role="tab"
          :aria-selected="dashboardPanel === 'pipeline'"
          tabindex="0"
          class="rounded px-3 py-1"
          :class="
            dashboardPanel === 'pipeline'
              ? 'bg-primary text-primary-foreground'
              : 'text-elevated-foreground hover:bg-overlay'
          "
          @click.stop="setDashboardPanel('pipeline')"
        >
          Pipeline
        </button>
        <button
          id="dash-tab-content-intelligence"
          type="button"
          role="tab"
          :aria-selected="dashboardPanel === 'contentIntelligence'"
          tabindex="0"
          class="rounded px-3 py-1"
          :class="
            dashboardPanel === 'contentIntelligence'
              ? 'bg-primary text-primary-foreground'
              : 'text-elevated-foreground hover:bg-overlay'
          "
          @click.stop="setDashboardPanel('contentIntelligence')"
        >
          Content intelligence
        </button>
      </nav>
      <p class="text-[11px] leading-snug text-muted">
        <template v-if="dashboardPanel === 'pipeline'">
          Runs, manifest throughput, cumulative growth from
          <span class="font-mono text-[10px]">run.json</span>
          summaries, and the latest run snapshot.
        </template>
        <template v-else>
          Catalog and on-disk shape: artifact activity, publish timing, graph node mix, and vector index snapshot.
        </template>
      </p>
    </div>

    <!-- Pipeline: execution & run.json metrics -->
    <div
      v-if="dashboardPanel === 'pipeline'"
      id="dashboard-panel-pipeline"
      class="space-y-4"
      role="tabpanel"
      aria-labelledby="dash-tab-pipeline"
    >
      <div
        v-if="shell.corpusLibraryApiAvailable"
        class="grid gap-4 lg:grid-cols-2"
      >
        <TypeCountBarChart
          v-if="Object.keys(manifestFeedCounts).length > 0"
          title="Episodes processed per feed (manifest)"
          help-text="Rows use feed titles from the library catalog when the manifest directory matches feed_id; otherwise the RSS hostname or stable directory id. Hover for count and raw id."
          :insight-text="insightManifestFeeds"
          :label-map="manifestFeedLabelMap"
          :counts="manifestFeedCounts"
        />
        <div
          v-else
          class="rounded border border-border bg-surface p-3 text-surface-foreground"
        >
          <h3 class="mb-2 text-sm font-semibold">
            Episodes processed per feed (manifest)
          </h3>
          <p class="text-xs text-muted">
            No <span class="font-mono text-[11px]">corpus_manifest.json</span> at corpus root,
            or it has no feeds.
          </p>
        </div>

        <CategoryLineChart
          v-if="runDurationTrend.labels.length > 0"
          title="Run duration (recent run.json, oldest → newest)"
          y-label="Seconds"
          help-text="Wall time per pipeline run from metrics.run_duration_seconds. X axis is the run created_at date (UTC, truncated to day)."
          :insight-text="insightRunDuration"
          :labels="runDurationTrend.labels"
          :values="runDurationTrend.values"
        />
        <div
          v-else
          class="rounded border border-border bg-surface p-3 text-surface-foreground"
        >
          <h3 class="mb-2 text-sm font-semibold">
            Run duration (run.json)
          </h3>
          <p class="text-xs text-muted">
            No <span class="font-mono text-[11px]">run.json</span> files found under the corpus.
          </p>
        </div>
      </div>
      <div
        v-else
        class="rounded border border-border bg-surface p-3 text-surface-foreground"
      >
        <p class="text-xs text-muted">
          Pipeline charts need the corpus library API
          (<span class="font-medium text-surface-foreground">/api/corpus/*</span>).
        </p>
      </div>

      <div
        v-if="shell.corpusLibraryApiAvailable"
        class="grid gap-4 lg:grid-cols-1"
      >
        <MultiSeriesLineChart
          v-if="runCorpusGrowth"
          title="Corpus growth from run summaries (cumulative)"
          help-text="Sums episodes_scraped_total, gi_artifacts_generated, and kg_artifacts_generated across successive run.json files (chronological). Useful for multi-batch corpora; a single run shows one point per metric."
          y-label="Cumulative count"
          :insight-text="insightRunGrowth"
          :labels="runCorpusGrowth.labels"
          :series="runCorpusGrowth.series"
        />
        <div
          v-else
          class="rounded border border-border bg-surface p-3 text-surface-foreground"
        >
          <h3 class="mb-2 text-sm font-semibold">
            Corpus growth from run summaries (cumulative)
          </h3>
          <p class="text-xs text-muted">
            No run.json metrics with non-zero episode or artifact counts, or no dated runs.
          </p>
        </div>
      </div>

      <div
        v-if="shell.corpusLibraryApiAvailable && latestRun"
        class="grid gap-4 lg:grid-cols-2"
      >
        <StackedStageBarChart
          title="Latest run — stage time (s)"
          help-text="One horizontal bar per pipeline stage from the most recent run.json metrics (wall seconds: scraping, parsing, normalizing, I/O & waiting)."
          :insight-text="insightPipelineStages"
          :stages="latestRunStages"
        />
        <SimpleDoughnutChart
          v-if="latestEpisodeOutcomeTotal > 0"
          title="Latest run — episode outcomes"
          help-text="Per-episode status counts (ok / failed / skipped) from episode_statuses in the latest run.json; sorted horizontal bars with count and share at bar ends."
          :insight-text="insightEpisodeOutcomes"
          :segments="latestEpisodeOutcomes"
        />
        <div
          v-else
          class="rounded border border-border bg-surface p-3 text-surface-foreground"
        >
          <h3 class="mb-2 text-sm font-semibold">
            Latest run — episode outcomes
          </h3>
          <p class="text-xs text-muted">
            No episode outcome rows in this <span class="font-mono text-[11px]">run.json</span>
            (metrics empty or no <span class="font-mono text-[11px]">episode_statuses</span>).
            Re-run the pipeline to refresh; older summaries may lack per-episode rows.
          </p>
        </div>
        <p class="col-span-full text-[11px] text-muted">
          Latest:
          <span class="font-mono">{{ latestRun.relative_path }}</span>
          <span v-if="latestRun.created_at"> · {{ latestRun.created_at }}</span>
        </p>
      </div>
    </div>

    <!-- Content intelligence: corpus shape, catalog, graph & index -->
    <div
      v-if="dashboardPanel === 'contentIntelligence'"
      id="dashboard-panel-content-intelligence"
      class="space-y-4"
      role="tabpanel"
      aria-labelledby="dash-tab-content-intelligence"
    >
      <div
        v-if="shell.hasCorpusPath && shell.healthStatus"
        class="space-y-3 rounded border border-border bg-surface p-3 text-surface-foreground"
        role="region"
        aria-label="Vector index and digest glance"
      >
        <h3 class="text-sm font-semibold">
          Vector index and digest glance
        </h3>
        <p
          v-if="indexStats.indexLoading && !indexStats.indexEnvelope"
          class="text-xs text-muted"
        >
          Loading index stats…
        </p>
        <p
          v-else-if="indexStats.indexError"
          class="text-xs text-danger"
        >
          {{ indexStats.indexError }}
        </p>
        <template v-if="indexFreshnessBlock">
          <div class="border-t border-border pt-2 first:border-t-0 first:pt-0">
            <div class="text-xs font-medium text-surface-foreground">
              {{ indexFreshnessBlock.headline }}
            </div>
            <ul
              v-if="indexFreshnessBlock.lines.length"
              class="mt-1 list-none space-y-0.5 text-[11px] text-muted"
            >
              <li
                v-for="(ln, i) in indexFreshnessBlock.lines"
                :key="i"
              >
                {{ ln }}
              </li>
            </ul>
            <p
              v-if="indexFreshnessBlock.insight"
              class="mt-1.5 text-[11px] font-medium leading-snug text-surface-foreground"
            >
              {{ indexFreshnessBlock.insight }}
            </p>
          </div>
        </template>
        <div
          v-if="indexRebuildNotes.length"
          class="border-t border-border pt-2"
        >
          <div class="text-xs font-medium text-surface-foreground">
            Index status
          </div>
          <ul class="mt-1 list-disc pl-4 text-[11px] leading-snug text-muted">
            <li
              v-for="(note, i) in indexRebuildNotes"
              :key="i"
            >
              {{ note }}
            </li>
          </ul>
        </div>
        <dl
          v-if="indexFootprintRows.length"
          class="grid grid-cols-1 gap-x-4 gap-y-1 border-t border-border pt-2 text-[11px] sm:grid-cols-2"
        >
          <template
            v-for="row in indexFootprintRows"
            :key="row.k"
          >
            <dt class="text-muted">
              {{ row.k }}
            </dt>
            <dd class="font-mono text-surface-foreground">
              {{ row.v }}
            </dd>
          </template>
        </dl>
        <div
          v-if="indexFeedsComparisonCounts && shell.corpusLibraryApiAvailable"
          class="border-t border-border pt-2"
        >
          <TypeCountBarChart
            title="Feeds in index vs catalog"
            help-text="Feed ids in the vector index vs distinct feeds in the on-disk catalog. Same horizontal scale from zero (GET /api/index/stats, /api/corpus/stats)."
            :insight-text="insightIndexFeedsVsCatalog"
            :counts="indexFeedsComparisonCounts"
            :label-map="indexFeedsComparisonLabels"
          />
        </div>
        <template v-if="shell.corpusDigestApiAvailable">
          <p
            v-if="dashLoading && !digestGlance && !digestGlanceError"
            class="border-t border-border pt-2 text-[11px] text-muted"
          >
            Loading digest glance…
          </p>
          <p
            v-else-if="digestGlanceLine"
            class="border-t border-border pt-2 text-[11px] leading-snug"
            :class="digestGlanceError ? 'text-danger' : 'text-muted'"
          >
            {{ digestGlanceLine }}
          </p>
        </template>
        <p
          v-else
          class="border-t border-border pt-2 text-[11px] text-muted"
        >
          Digest API unavailable on this server — compact digest glance is not loaded.
        </p>
      </div>

      <div class="grid gap-4 lg:grid-cols-2">
        <div
          v-if="!shell.hasCorpusPath || !shell.healthStatus"
          class="rounded border border-border bg-surface p-3 text-surface-foreground"
        >
          <h3 class="mb-2 text-sm font-semibold">
            GI + KG artifacts by write day (UTC)
          </h3>
          <p class="text-xs text-muted">
            Set corpus path and ensure the API is up to load artifact mtimes.
          </p>
        </div>
        <div
          v-else-if="artifactTimelineOverCap"
          class="rounded border border-border bg-surface p-3 text-surface-foreground"
        >
          <h3 class="mb-2 text-sm font-semibold">
            GI + KG artifacts by write day (UTC)
          </h3>
          <p class="text-xs text-warning">
            Too many artifact files ({{ artifactTimelineRowCount }}) to chart in the browser (cap
            {{ MAX_ARTIFACT_ROWS_FOR_CLIENT_TIMELINE }}). Use a narrower path or a future
            server-side aggregate.
          </p>
        </div>
        <div
          v-else-if="shell.artifactsError"
          class="rounded border border-border bg-surface p-3 text-surface-foreground"
        >
          <h3 class="mb-2 text-sm font-semibold">
            GI + KG artifacts by write day (UTC)
          </h3>
          <p class="text-xs text-danger">
            {{ shell.artifactsError }}
          </p>
          <p class="mt-2 text-xs leading-snug text-muted">
            The API only accepts corpus paths under the server's configured output root
            (<code class="rounded bg-overlay px-0.5 font-mono text-[10px]">--output-dir</code>
            /
            <code class="rounded bg-overlay px-0.5 font-mono text-[10px]">PODCAST_SERVE_OUTPUT_DIR</code>).
            For acceptance or multi-root trees, set that to a parent directory (e.g. repo root):
            <code class="mt-1 block rounded bg-overlay px-1 py-0.5 font-mono text-[10px]">make serve-api SERVE_OUTPUT_DIR=.</code>
          </p>
        </div>
        <CategoryLineChart
          v-else-if="artifactTimelineLabels.length > 0"
          title="GI + KG artifacts by write day (UTC)"
          y-label="Files"
          help-text="Each point is how many GI/KG JSON files were last modified on that UTC day (from the artifact list). Hover for the exact count."
          :insight-text="insightArtifactTimeline"
          :labels="artifactTimelineLabels"
          :values="artifactTimelineValues"
        />
        <div
          v-else
          class="rounded border border-border bg-surface p-3 text-surface-foreground"
        >
          <h3 class="mb-2 text-sm font-semibold">
            GI + KG artifacts by write day (UTC)
          </h3>
          <p class="text-xs text-muted">
            No GI/KG JSON files in the artifact list for this path (or still loading).
          </p>
        </div>

        <VerticalBarChart
          v-if="shell.corpusLibraryApiAvailable && publishMonthSeries.length > 0"
          title="Episodes by publish month (catalog)"
          help-text="Episode counts grouped by publish month from on-disk metadata (catalog scan)."
          y-axis-label="Episodes"
          :insight-text="insightPublishMonth"
          :labels="publishMonthSeries.map((x) => x.label)"
          :values="publishMonthSeries.map((x) => x.count)"
        />
        <div
          v-else-if="shell.corpusLibraryApiAvailable"
          class="rounded border border-border bg-surface p-3 text-surface-foreground"
        >
          <h3 class="mb-2 text-sm font-semibold">
            Episodes by publish month (catalog)
          </h3>
          <p class="text-xs text-muted">
            No publish dates in catalog, or corpus is empty.
          </p>
        </div>
        <div
          v-else
          class="rounded border border-border bg-surface p-3 text-surface-foreground"
        >
          <h3 class="mb-2 text-sm font-semibold">
            Episodes by publish month (catalog)
          </h3>
          <p class="text-xs text-muted">
            Corpus library API unavailable — publish histogram needs
            <span class="font-medium text-surface-foreground">/api/corpus/stats</span>.
          </p>
        </div>
      </div>

      <div
        v-if="shell.corpusLibraryApiAvailable"
        class="grid gap-4 lg:grid-cols-1"
      >
        <MultiSeriesLineChart
          v-if="artifactGiKgCumulative"
          title="GI vs KG artifacts over time (cumulative by write day)"
          help-text="Cumulative count of .gi.json / .kg.json files by UTC modification day from the artifact list. Distinct from vector index doc types (see below)."
          y-label="Files (cumulative)"
          :insight-text="insightGiKgCumulative"
          :labels="artifactGiKgCumulative.labels"
          :series="artifactGiKgCumulative.series"
        />
        <div
          v-else-if="artifactTimelineOverCap"
          class="rounded border border-border bg-surface p-3 text-surface-foreground"
        >
          <h3 class="mb-2 text-sm font-semibold">
            GI vs KG artifacts over time (cumulative)
          </h3>
          <p class="text-xs text-warning">
            Artifact list too large for this chart (same cap as the daily timeline).
          </p>
        </div>
        <div
          v-else-if="shell.artifactsError"
          class="rounded border border-border bg-surface p-3 text-surface-foreground"
        >
          <h3 class="mb-2 text-sm font-semibold">
            GI vs KG artifacts over time (cumulative)
          </h3>
          <p class="text-xs text-danger">
            {{ shell.artifactsError }}
          </p>
          <p class="mt-2 text-xs text-muted">
            Fix the artifact list error above; this chart uses the same
            <code class="font-mono text-[10px]">GET /api/artifacts</code> data.
          </p>
        </div>
        <div
          v-else
          class="rounded border border-border bg-surface p-3 text-surface-foreground"
        >
          <h3 class="mb-2 text-sm font-semibold">
            GI vs KG artifacts over time (cumulative)
          </h3>
          <p class="text-xs text-muted">
            No GI/KG artifacts in the list for this path, or set corpus path and ensure the API listed files.
          </p>
        </div>
      </div>

      <div class="grid gap-4 lg:grid-cols-2">
        <template v-if="artifactMetrics">
          <TypeCountBarChart
            title="Node types (visual groups)"
            help-text="Counts nodes in the graph loaded on the Graph tab, grouped by viewer legend buckets (not RSS feed ids)."
            :insight-text="insightNodeTypes"
            :counts="artifactMetrics.visualNodeTypeCounts"
          />
        </template>
        <div
          v-else
          class="rounded border border-dashed border-border bg-surface p-6 text-center text-sm text-muted"
        >
          Load artifacts on the Graph tab to see charts.
        </div>

        <template v-if="indexStats.indexEnvelope?.available && indexStats.indexEnvelope.stats">
          <TypeCountBarChart
            title="Indexed document types (vector index snapshot)"
            help-text="Current FAISS / vector index chunk doc_type tallies from GET /api/index/stats. Historical per-day breakdown is not stored in run.json yet — this is a point-in-time snapshot only."
            :insight-text="insightIndexDocTypes"
            :counts="indexDocTypeCounts"
            :bar-end-percent-total="indexStats.indexEnvelope.stats.total_vectors"
          />
        </template>
      </div>
    </div>
  </div>
</template>
