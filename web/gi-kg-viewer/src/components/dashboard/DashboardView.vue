<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { CorpusManifestDocument } from '../../api/corpusMetricsApi'
import { fetchCorpusFeeds, type CorpusFeedItem } from '../../api/corpusLibraryApi'
import {
  fetchCorpusManifest,
  fetchCorpusRunsSummary,
  fetchCorpusStats,
} from '../../api/corpusMetricsApi'
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
const dashLoading = ref(false)
const dashError = ref<string | null>(null)

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
  dashError.value = null
  corpusStats.value = null
  manifestDoc.value = null
  runsSummary.value = null
  feedsCatalog.value = []
  if (!shell.healthStatus || !shell.hasCorpusPath) {
    return
  }
  dashLoading.value = true
  try {
    await shell.fetchArtifactList()
    const p = shell.corpusPath.trim()
    if (shell.corpusLibraryApiAvailable) {
      const [stats, runs] = await Promise.all([
        fetchCorpusStats(p),
        fetchCorpusRunsSummary(p),
      ])
      corpusStats.value = stats
      runsSummary.value = runs
      try {
        const fr = await fetchCorpusFeeds(p)
        feedsCatalog.value = fr.feeds
      } catch {
        feedsCatalog.value = []
      }
      try {
        manifestDoc.value = await fetchCorpusManifest(p)
      } catch {
        manifestDoc.value = null
      }
    }
  } catch (e) {
    dashError.value = e instanceof Error ? e.message : String(e)
  } finally {
    dashLoading.value = false
  }
}

watch(
  () =>
    [shell.corpusPath, shell.healthStatus, shell.corpusLibraryApiAvailable] as const,
  () => {
    void refreshDashboardMetrics()
  },
  { immediate: true },
)
</script>

<template>
  <div class="space-y-4">
    <p class="text-xs text-muted">
      Corpus path, graph metrics, and vector index live under
      <span class="font-medium text-surface-foreground">API · Data</span>
      in the left panel. Charts below refresh when the corpus path and API are available.
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

    <!-- Timelines & corpus shape -->
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

    <!-- Manifest + run trends -->
    <div
      v-if="shell.corpusLibraryApiAvailable"
      class="grid gap-4 lg:grid-cols-2"
    >
      <TypeCountBarChart
        v-if="Object.keys(manifestFeedCounts).length > 0"
        title="Episodes processed per feed (manifest)"
        help-text="Rows use feed titles from the library catalog when the manifest directory matches feed_id; otherwise the RSS hostname or stable directory id. Hover for count and raw id."
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

    <!-- Corpus growth (runs + artifact write curve) -->
    <div
      v-if="shell.corpusLibraryApiAvailable"
      class="grid gap-4 lg:grid-cols-2"
    >
      <MultiSeriesLineChart
        v-if="runCorpusGrowth"
        title="Corpus growth from run summaries (cumulative)"
        help-text="Sums episodes_scraped_total, gi_artifacts_generated, and kg_artifacts_generated across successive run.json files (chronological). Useful for multi-batch corpora; a single run shows one point per metric."
        y-label="Cumulative count"
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

      <MultiSeriesLineChart
        v-if="artifactGiKgCumulative"
        title="GI vs KG artifacts over time (cumulative by write day)"
        help-text="Cumulative count of .gi.json / .kg.json files by UTC modification day from the artifact list. Distinct from vector index doc types (see below)."
        y-label="Files (cumulative)"
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

    <!-- Latest run pipeline -->
    <div
      v-if="shell.corpusLibraryApiAvailable && latestRun"
      class="grid gap-4 lg:grid-cols-2"
    >
      <StackedStageBarChart
        title="Latest run — stage time (s)"
        help-text="Stacked wall-time segments from the most recent run.json metrics (scraping, parsing, normalizing, I/O & waiting)."
        :stages="latestRunStages"
      />
      <SimpleDoughnutChart
        v-if="latestEpisodeOutcomeTotal > 0"
        title="Latest run — episode outcomes"
        help-text="Per-episode status counts (ok / failed / skipped) parsed from episode_statuses in the latest run.json."
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

    <!-- Graph + index (existing) -->
    <div class="grid gap-4 lg:grid-cols-2">
        <template v-if="artifactMetrics">
        <TypeCountBarChart
          title="Node types (visual groups)"
          help-text="Counts nodes in the graph loaded on the Graph tab, grouped by viewer legend buckets (not RSS feed ids)."
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
          :counts="indexDocTypeCounts"
        />
      </template>
    </div>
  </div>
</template>
