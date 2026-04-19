<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { fetchCorpusCoverage, type CorpusCoverageResponse } from '../../api/corpusCoverageApi'
import { fetchCorpusDigest, type CorpusDigestResponse } from '../../api/digestApi'
import { fetchCorpusFeeds, type CorpusFeedItem } from '../../api/corpusLibraryApi'
import { fetchCorpusRunsSummary, type CorpusRunSummaryItem } from '../../api/corpusMetricsApi'
import { fetchCorpusTopPersons, type TopPersonItem } from '../../api/corpusPersonsApi'
import { useDashboardNavStore } from '../../stores/dashboardNav'
import { useIndexStatsStore } from '../../stores/indexStats'
import { useShellStore } from '../../stores/shell'
import { lastYmdOfMonth } from '../../utils/dashboardTime'
import { StaleGeneration } from '../../utils/staleGeneration'
import ArtifactActivityChart from './ArtifactActivityChart.vue'
import BriefingCard from './BriefingCard.vue'
import CoverageByMonthChart from './CoverageByMonthChart.vue'
import FeedCoverageTable from './FeedCoverageTable.vue'
import IndexStatusCard from './IndexStatusCard.vue'
import IntelligenceSnapshot from './IntelligenceSnapshot.vue'
import TopicClustersStatusBlock from './TopicClustersStatusBlock.vue'
import PipelineRunHistoryStrip from './PipelineRunHistoryStrip.vue'
import PipelineStageChart from './PipelineStageChart.vue'
import TopicLandscape from './TopicLandscape.vue'
import TopVoices from './TopVoices.vue'
import VerticalBarChart from './VerticalBarChart.vue'

const emit = defineEmits<{
  'go-graph': []
  'open-library': []
  'open-digest': []
}>()

const shell = useShellStore()
const indexStats = useIndexStatsStore()
const dashboardNav = useDashboardNavStore()

const dashTab = ref<'coverage' | 'intelligence' | 'pipeline'>('coverage')
const runs = ref<CorpusRunSummaryItem[]>([])
const coverage = ref<CorpusCoverageResponse | null>(null)
const feeds = ref<CorpusFeedItem[]>([])
const digestIntel = ref<CorpusDigestResponse | null>(null)
const topPersons = ref<TopPersonItem[]>([])
const topPersonsLoading = ref(false)
const topPersonsError = ref<string | null>(null)
const dashLoading = ref(false)
const dashError = ref<string | null>(null)
const dashGate = new StaleGeneration()

function selectTab(tab: 'coverage' | 'intelligence' | 'pipeline'): void {
  dashTab.value = tab
}

async function fetchTopPersons(): Promise<void> {
  const root = shell.corpusPath.trim()
  if (!root || !shell.healthStatus) {
    topPersons.value = []
    return
  }
  topPersonsLoading.value = true
  topPersonsError.value = null
  try {
    const res = await fetchCorpusTopPersons(root, 5)
    topPersons.value = res.persons
  } catch (e) {
    topPersonsError.value = e instanceof Error ? e.message : String(e)
    topPersons.value = []
  } finally {
    topPersonsLoading.value = false
  }
}

async function refreshDashboard(): Promise<void> {
  const seq = dashGate.bump()
  dashLoading.value = true
  dashError.value = null
  const root = shell.corpusPath.trim()
  if (!root || !shell.healthStatus) {
    runs.value = []
    coverage.value = null
    feeds.value = []
    digestIntel.value = null
    if (dashGate.isCurrent(seq)) {
      dashLoading.value = false
    }
    return
  }
  try {
    void indexStats.refreshIndexStats()
    const sum = await fetchCorpusRunsSummary(root)
    if (dashGate.isStale(seq)) {
      return
    }
    runs.value = sum.runs
    coverage.value = await fetchCorpusCoverage(root).catch(() => null)
    try {
      const fd = await fetchCorpusFeeds(root)
      if (dashGate.isStale(seq)) {
        return
      }
      feeds.value = fd.feeds ?? []
    } catch {
      feeds.value = []
    }
    digestIntel.value = await fetchCorpusDigest(root, {
      window: '7d',
      compact: false,
      maxRows: 3,
    }).catch(() => null)
    if (dashGate.isStale(seq)) {
      return
    }
    await fetchTopPersons()
  } catch (e) {
    if (dashGate.isStale(seq)) {
      return
    }
    dashError.value = e instanceof Error ? e.message : String(e)
    runs.value = []
    coverage.value = null
  } finally {
    if (dashGate.isCurrent(seq)) {
      dashLoading.value = false
    }
  }
}

watch(
  () => [shell.corpusPath, shell.healthStatus] as const,
  () => {
    void refreshDashboard()
  },
  { immediate: true },
)

watch(dashTab, (t) => {
  if (t === 'intelligence') {
    void fetchTopPersons()
  }
})

const latestRun = computed(() => runs.value[0] ?? null)

const stagesLatest = computed(() => {
  const r = latestRun.value
  if (!r) {
    return []
  }
  return [
    { label: 'Scraping', seconds: Math.max(0, r.time_scraping_seconds ?? 0) },
    { label: 'Parsing', seconds: Math.max(0, r.time_parsing_seconds ?? 0) },
    { label: 'Normalizing', seconds: Math.max(0, r.time_normalizing_seconds ?? 0) },
    { label: 'I/O & waiting', seconds: Math.max(0, r.time_io_and_waiting_seconds ?? 0) },
  ]
})

const durationFive = computed(() => {
  const dated = runs.value.filter(
    (r) => (r.created_at?.length ?? 0) >= 10 && r.run_duration_seconds != null,
  )
  const chrono = [...dated].sort((a, b) => (a.created_at ?? '').localeCompare(b.created_at ?? ''))
  const last5 = chrono.slice(-5)
  const avg = last5.length
    ? last5.reduce((a, r) => a + (r.run_duration_seconds ?? 0), 0) / last5.length
    : 0
  const lastDur = last5[last5.length - 1]?.run_duration_seconds ?? 0
  let title = 'Run duration stable — within 10% of 5-run average'
  if (last5.length >= 2 && avg > 0) {
    const delta = (lastDur - avg) / avg
    if (delta > 0.1) {
      title = `Latest run ${Math.round(delta * 100)}% slower than 5-run average`
    } else if (delta < -0.1) {
      title = `Latest run ${Math.round(-delta * 100)}% faster than 5-run average`
    }
  }
  return {
    title,
    labels: last5.map((r) => (r.created_at ?? '').slice(0, 10)),
    values: last5.map((r) => r.run_duration_seconds ?? 0),
  }
})

const episodesPerRun = computed(() => {
  const dated = runs.value.filter((r) => (r.created_at?.length ?? 0) >= 10)
  const chrono = [...dated].sort((a, b) => (a.created_at ?? '').localeCompare(b.created_at ?? ''))
  const cap = chrono.slice(-48)
  const eps = cap.map((r) => r.episodes_scraped_total ?? 0)
  const avg = eps.length ? eps.reduce((a, b) => a + b, 0) / eps.length : 0
  const title =
    eps.length >= 3 && eps.slice(-3).every((v) => v < avg * 0.85)
      ? 'Processing volume declining — last 3 runs below average'
      : `Average ${avg.toFixed(0)} episodes per run`
  return {
    title,
    labels: cap.map((r) => (r.created_at ?? '').slice(0, 10)),
    values: eps,
  }
})

const latestOutcomes = computed(() => latestRun.value?.episode_outcomes ?? {})

function onSelectMonth(ym: string): void {
  dashboardNav.setHandoff({
    kind: 'library',
    since: `${ym}-01`,
    until: lastYmdOfMonth(ym),
    missingGiOnly: true,
  })
  emit('open-library')
}

function onSelectFeed(fid: string): void {
  dashboardNav.setHandoff({ kind: 'library', feedId: fid })
  emit('open-library')
}

function openLibraryFailures(): void {
  dashboardNav.setHandoff({ kind: 'library' })
  emit('open-library')
}
</script>

<template>
  <div class="space-y-4 pb-8 text-surface-foreground">
    <p
      v-if="dashLoading"
      class="text-xs text-muted"
    >
      Loading dashboard…
    </p>
    <p
      v-if="dashError"
      class="text-xs text-danger"
    >
      {{ dashError }}
    </p>

    <BriefingCard
      :runs="runs"
      :coverage="coverage"
      :index-env="indexStats.indexEnvelope"
      :catalog-feed-count="feeds.length"
      :corpus-path="shell.corpusPath"
      :api-ready="Boolean(shell.healthStatus)"
      @select-tab="selectTab"
      @rebuild-index="indexStats.requestIndexRebuild(true)"
      @open-library="emit('open-library')"
    />

    <nav
      class="flex flex-wrap gap-1 rounded border border-border bg-elevated p-0.5 text-xs font-medium"
      role="tablist"
      aria-label="Dashboard tabs"
    >
      <button
        type="button"
        role="tab"
        :aria-selected="dashTab === 'coverage'"
        class="rounded px-3 py-1"
        :class="
          dashTab === 'coverage'
            ? 'bg-primary text-primary-foreground'
            : 'text-muted hover:bg-overlay'
        "
        @click="selectTab('coverage')"
      >
        Coverage
      </button>
      <button
        type="button"
        role="tab"
        :aria-selected="dashTab === 'intelligence'"
        class="rounded px-3 py-1"
        :class="
          dashTab === 'intelligence'
            ? 'bg-primary text-primary-foreground'
            : 'text-muted hover:bg-overlay'
        "
        @click="selectTab('intelligence')"
      >
        Intelligence
      </button>
      <button
        type="button"
        role="tab"
        :aria-selected="dashTab === 'pipeline'"
        class="rounded px-3 py-1"
        :class="
          dashTab === 'pipeline'
            ? 'bg-primary text-primary-foreground'
            : 'text-muted hover:bg-overlay'
        "
        @click="selectTab('pipeline')"
      >
        Pipeline
      </button>
    </nav>

    <div
      v-if="dashTab === 'coverage'"
      class="space-y-4"
      role="tabpanel"
    >
      <CoverageByMonthChart
        v-if="coverage"
        :rows="coverage.by_month"
        @select-month="onSelectMonth"
      />
      <FeedCoverageTable
        v-if="coverage"
        :rows="coverage.by_feed"
        :feeds-indexed="indexStats.indexEnvelope?.stats?.feeds_indexed ?? []"
        @select-feed="onSelectFeed"
      />
      <ArtifactActivityChart :artifact-items="shell.artifactList" />
      <IndexStatusCard @rebuild-index="indexStats.requestIndexRebuild(true)" />
    </div>

    <div
      v-else-if="dashTab === 'intelligence'"
      class="space-y-4"
      role="tabpanel"
    >
      <IntelligenceSnapshot
        :digest="digestIntel"
        @open-digest="emit('open-digest')"
      />
      <TopicClustersStatusBlock />
      <TopicLandscape @go-graph="emit('go-graph')" />
      <TopVoices
        :persons="topPersons"
        :loading="topPersonsLoading"
        :error="topPersonsError"
      />
    </div>

    <div
      v-else
      class="space-y-4"
      role="tabpanel"
    >
      <PipelineRunHistoryStrip :runs="runs" />
      <VerticalBarChart
        v-if="durationFive.labels.length"
        :title="durationFive.title"
        :labels="durationFive.labels"
        :values="durationFive.values"
        y-axis-label="Seconds"
        insight-text="Wall clock from metrics.run_duration_seconds (5 most recent runs)."
      />
      <div class="grid gap-4 lg:grid-cols-2">
        <PipelineStageChart
          v-if="stagesLatest.some((s) => s.seconds > 0)"
          :stages="stagesLatest"
        />
        <div
          class="rounded border border-border bg-surface p-3 text-sm"
          data-testid="pipeline-episode-outcomes"
        >
          <h3 class="mb-2 text-[10px] font-semibold uppercase tracking-wider text-muted">
            Outcomes
          </h3>
          <p>
            <span class="text-success">●</span> {{ (latestOutcomes.ok ?? 0).toLocaleString() }} succeeded
          </p>
          <p>
            <span class="text-muted">○</span> {{ (latestOutcomes.skipped ?? 0).toLocaleString() }} skipped
          </p>
          <p>
            <span class="text-danger">✗</span> {{ (latestOutcomes.failed ?? 0).toLocaleString() }} failed
            <button
              v-if="(latestOutcomes.failed ?? 0) > 0"
              type="button"
              class="ml-2 text-xs font-medium text-primary hover:underline"
              @click="openLibraryFailures"
            >
              View failures →
            </button>
          </p>
        </div>
      </div>
      <VerticalBarChart
        v-if="episodesPerRun.labels.length"
        :title="episodesPerRun.title"
        :labels="episodesPerRun.labels"
        :values="episodesPerRun.values"
        y-axis-label="Episodes"
        insight-text="Episodes scraped per run from run.json summaries."
      />
    </div>
  </div>
</template>
