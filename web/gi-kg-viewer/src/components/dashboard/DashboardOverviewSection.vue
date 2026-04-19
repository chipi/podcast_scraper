<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { fetchCorpusStats, type CorpusStatsResponse } from '../../api/corpusMetricsApi'
import { useArtifactsStore } from '../../stores/artifacts'
import { useIndexStatsStore } from '../../stores/indexStats'
import { useShellStore } from '../../stores/shell'
import { computeArtifactMetrics } from '../../utils/metrics'
import { StaleGeneration } from '../../utils/staleGeneration'
import type { MetricRow } from '../../utils/metrics'
import MetricsPanel from './MetricsPanel.vue'
import TopicClustersStatusBlock from './TopicClustersStatusBlock.vue'

const shell = useShellStore()
const artifacts = useArtifactsStore()
const indexStats = useIndexStatsStore()

const artifactMetrics = computed(() => {
  const art = artifacts.displayArtifact
  if (!art) return null
  return computeArtifactMetrics(art)
})

const corpusStats = ref<CorpusStatsResponse | null>(null)
const corpusStatsLoading = ref(false)
const corpusStatsError = ref<string | null>(null)
const corpusCatalogStatsGate = new StaleGeneration()
const graphPanelRefreshGate = new StaleGeneration()

const canFetchCorpusStats = computed(
  () =>
    Boolean(shell.healthStatus) &&
    shell.hasCorpusPath &&
    shell.corpusMetricsApiAvailable !== false,
)

async function refreshCorpusCatalogStats(): Promise<void> {
  const seq = corpusCatalogStatsGate.bump()
  corpusStatsError.value = null
  if (!canFetchCorpusStats.value) {
    corpusStats.value = null
    if (corpusCatalogStatsGate.isCurrent(seq)) {
      corpusStatsLoading.value = false
    }
    return
  }
  corpusStatsLoading.value = true
  try {
    const stats = await fetchCorpusStats(shell.corpusPath.trim())
    if (corpusCatalogStatsGate.isStale(seq)) {
      return
    }
    corpusStats.value = stats
  } catch (e) {
    if (corpusCatalogStatsGate.isStale(seq)) {
      return
    }
    corpusStatsError.value = e instanceof Error ? e.message : String(e)
    corpusStats.value = null
  } finally {
    if (corpusCatalogStatsGate.isCurrent(seq)) {
      corpusStatsLoading.value = false
    }
  }
}

watch(
  () =>
    [shell.corpusPath, shell.healthStatus, shell.corpusMetricsApiAvailable] as const,
  () => {
    void refreshCorpusCatalogStats()
  },
  { immediate: true },
)

function histogramEpisodeSum(h: Record<string, number>): number {
  let s = 0
  for (const v of Object.values(h)) {
    s += v
  }
  return s
}

const artifactListBreakdown = computed(() => {
  const list = shell.artifactList
  if (!Array.isArray(list) || list.length === 0) return null
  let gi = 0
  let kg = 0
  for (const a of list) {
    if (a.kind === 'gi') gi += 1
    else if (a.kind === 'kg') kg += 1
  }
  return { gi, kg, total: list.length }
})

const catalogOverviewRows = computed((): MetricRow[] => {
  const c = corpusStats.value
  if (!c) return []
  const hist = c.publish_month_histogram ?? {}
  const monthSum = histogramEpisodeSum(hist)
  const rows: MetricRow[] = [
    { k: 'Catalog feeds', v: String(c.catalog_feed_count) },
    { k: 'Catalog episodes', v: String(c.catalog_episode_count) },
    { k: 'Digest topic bands', v: String(c.digest_topics_configured) },
    { k: 'Distinct publish months', v: String(Object.keys(hist).length) },
    {
      k: 'Episodes in month histogram',
      v: String(monthSum),
    },
  ]
  const br = artifactListBreakdown.value
  if (br) {
    rows.push({
      k: 'GI/KG paths listed',
      v: `${br.total} (${br.gi} GI, ${br.kg} KG)`,
    })
  }
  return rows
})

const graphPanelRefreshing = ref(false)

async function refreshGraphPanel(): Promise<void> {
  if (!shell.healthStatus || !shell.hasCorpusPath) return
  const seq = graphPanelRefreshGate.bump()
  graphPanelRefreshing.value = true
  try {
    await shell.fetchArtifactList()
    if (graphPanelRefreshGate.isStale(seq)) {
      return
    }
    if (artifacts.selectedRelPaths.length > 0) {
      await artifacts.loadSelected()
    }
  } finally {
    if (graphPanelRefreshGate.isCurrent(seq)) {
      graphPanelRefreshing.value = false
    }
  }
}

const graphRefreshDisabled = computed(
  () =>
    !shell.healthStatus ||
    !shell.hasCorpusPath ||
    graphPanelRefreshing.value ||
    shell.artifactsLoading,
)

const corpusCatalogRefreshDisabled = computed(
  () =>
    !canFetchCorpusStats.value ||
    corpusStatsLoading.value,
)
</script>

<template>
  <div class="space-y-2">
    <h2 class="text-xs font-medium text-surface-foreground">
      Data
    </h2>
    <p class="text-[10px] leading-snug text-muted">
      Corpus root, catalog snapshot from
      <code class="rounded bg-overlay px-0.5 font-mono text-[9px]">GET /api/corpus/stats</code>,
      merged graph metrics from loaded GI/KG, and vector index (same path as Dashboard charts).
    </p>

    <TopicClustersStatusBlock />

    <div class="rounded border border-border bg-elevated p-2 text-[10px]">
      <h3 class="text-xs font-semibold text-surface-foreground">
        Corpus root
      </h3>
      <p
        v-if="shell.corpusPath.trim()"
        class="mt-1.5 break-words leading-snug text-surface-foreground"
      >
        <span class="text-muted">Path:</span>
        {{ shell.corpusPath.trim() }}
      </p>
      <p
        v-else
        class="mt-1.5 text-muted leading-snug"
      >
        Set the output folder on the <strong class="font-medium text-surface-foreground">status bar</strong>
        corpus path field.
      </p>
      <p
        v-if="shell.resolvedCorpusPath && shell.resolvedCorpusPath !== shell.corpusPath.trim()"
        class="mt-1 font-mono text-[9px] leading-tight text-muted break-all"
      >
        Resolved: {{ shell.resolvedCorpusPath }}
      </p>
    </div>

    <div
      class="rounded border border-border bg-elevated p-2 text-[10px]"
      :aria-busy="corpusStatsLoading ? 'true' : undefined"
    >
      <div class="mb-1.5 flex flex-wrap items-center justify-between gap-1">
        <h3 class="text-xs font-semibold text-surface-foreground">
          Corpus catalog
        </h3>
        <button
          type="button"
          class="rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-40"
          :disabled="corpusCatalogRefreshDisabled"
          @click="refreshCorpusCatalogStats()"
        >
          {{ corpusStatsLoading ? 'Loading…' : 'Refresh' }}
        </button>
      </div>
      <p
        v-if="!shell.healthStatus"
        class="text-[10px] text-muted"
      >
        Needs the API.
      </p>
      <p
        v-else-if="!shell.hasCorpusPath"
        class="text-[10px] text-muted"
      >
        Set corpus root on the status bar.
      </p>
      <p
        v-else-if="shell.corpusMetricsApiAvailable === false"
        class="text-[10px] text-muted"
      >
        Corpus metrics not advertised on health — upgrade the API.
      </p>
      <p
        v-else-if="corpusStatsError"
        class="text-[10px] text-danger"
      >
        {{ corpusStatsError }}
      </p>
      <p
        v-else-if="corpusStatsLoading && !corpusStats"
        class="text-[10px] text-muted"
      >
        Loading…
      </p>
      <MetricsPanel
        v-else-if="corpusStats && catalogOverviewRows.length"
        hide-title
        title=""
        unframed
        dense
        class="mt-1 border-t border-border/60 pt-1.5"
        :rows="catalogOverviewRows"
      />
    </div>

    <div class="rounded border border-border bg-elevated p-2 text-[10px]">
      <div class="mb-1.5 flex flex-wrap items-center justify-between gap-1">
        <h3 class="text-xs font-semibold text-surface-foreground">
          Graph
        </h3>
        <button
          type="button"
          class="rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-40"
          :disabled="graphRefreshDisabled"
          :title="
            !shell.healthStatus || !shell.hasCorpusPath
              ? 'Needs API and corpus path'
              : 'Refresh artifact list and reload selected GI/KG into the merged graph'
          "
          @click="refreshGraphPanel()"
        >
          {{
            graphPanelRefreshing || shell.artifactsLoading ? 'Loading…' : 'Refresh'
          }}
        </button>
      </div>
      <MetricsPanel
        v-if="artifactMetrics"
        hide-title
        title=""
        unframed
        dense
        class="mt-1 border-t border-border/60 pt-1.5"
        :rows="artifactMetrics.rows"
      />
      <p
        v-else
        class="text-center text-muted leading-snug"
      >
        Load artifacts on the Graph tab (or Corpus → List / Load into graph) to see graph metrics.
      </p>
    </div>

    <div
      class="rounded border border-border bg-elevated p-2 text-[10px]"
      :aria-busy="indexStats.indexLoading ? 'true' : undefined"
    >
      <div class="mb-1.5 flex flex-wrap items-center justify-between gap-1">
        <h3 class="text-xs font-semibold text-surface-foreground">
          Vector index
        </h3>
        <div class="flex flex-wrap items-center gap-1">
          <button
            type="button"
            class="rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-40"
            :disabled="!shell.healthStatus || indexStats.indexLoading"
            @click="indexStats.refreshIndexStats()"
          >
            {{ indexStats.indexLoading ? 'Loading…' : 'Refresh' }}
          </button>
          <button
            type="button"
            class="rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-40"
            :title="
              indexStats.indexEnvelope?.reason === 'faiss_unavailable'
                ? 'FAISS is not available on the server.'
                : 'Run incremental indexing (podcast index) in the background.'
            "
            :disabled="indexStats.rebuildActionsDisabled"
            @click="indexStats.requestIndexRebuild(false)"
          >
            {{ indexStats.rebuildSubmitting ? 'Queueing…' : 'Update index' }}
          </button>
          <button
            type="button"
            class="rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay disabled:opacity-40"
            :disabled="indexStats.rebuildActionsDisabled"
            @click="indexStats.requestIndexRebuild(true)"
          >
            Full rebuild
          </button>
        </div>
      </div>
      <p
        v-if="indexStats.indexEnvelope?.rebuild_in_progress"
        class="mb-1 text-[10px] text-muted leading-snug"
      >
        Background index job running — stats refresh automatically. You can also use Refresh.
      </p>
      <p
        v-if="indexStats.indexEnvelope?.rebuild_last_error"
        class="mb-1 text-[10px] text-danger leading-snug"
      >
        Last rebuild error: {{ indexStats.indexEnvelope.rebuild_last_error }}
      </p>
      <p
        v-if="!shell.healthStatus"
        class="text-[10px] text-muted"
      >
        Needs the API.
      </p>
      <p
        v-else-if="indexStats.indexError"
        class="text-[10px] text-danger"
      >
        {{ indexStats.indexError }}
      </p>
      <p
        v-else-if="indexStats.indexLoading && !indexStats.indexEnvelope"
        class="text-[10px] text-muted"
      >
        Loading…
      </p>
      <template v-else-if="indexStats.indexEnvelope">
        <div
          v-if="indexStats.indexHealthBanner"
          class="mb-2 rounded border p-1.5 text-[10px] leading-snug"
          :class="
            indexStats.indexHealthBanner.kind === 'warn'
              ? 'border-warning/40 bg-warning/10 text-surface-foreground'
              : 'border-border bg-overlay/30 text-muted'
          "
        >
          <p
            class="font-medium"
            :class="
              indexStats.indexHealthBanner.kind === 'warn'
                ? 'text-warning'
                : 'text-surface-foreground'
            "
          >
            {{
              indexStats.indexHealthBanner.kind === 'warn'
                ? 'Reindex recommended'
                : 'Search / corpus note'
            }}
          </p>
          <ul class="mt-0.5 list-disc pl-3 text-[10px] leading-snug">
            <li
              v-for="(line, i) in indexStats.indexHealthBanner.lines"
              :key="i"
            >
              {{ line }}
            </li>
          </ul>
          <p
            v-if="
              indexStats.indexHealthBanner.kind === 'warn'
                && indexStats.indexEnvelope.artifact_newest_mtime
            "
            class="mt-0.5 font-mono text-[9px] opacity-90 break-all"
          >
            Newest index-related file: {{ indexStats.indexEnvelope.artifact_newest_mtime }}
          </p>
        </div>
        <div
          v-if="!indexStats.indexEnvelope.available"
          class="text-[10px] text-muted leading-snug"
        >
          <p>
            <span class="font-medium text-surface-foreground">No index</span>
            —
            {{ indexStats.indexEnvelope.reason === 'no_index'
              ? 'No FAISS data at the expected path (run corpus indexing).'
              : indexStats.indexEnvelope.reason === 'no_corpus_path'
                ? 'Set corpus root or start the server with --output-dir.'
                : indexStats.indexEnvelope.reason === 'faiss_unavailable'
                  ? 'FAISS is not installed in this Python environment.'
                  : (indexStats.indexEnvelope.reason || 'Unavailable') }}
          </p>
          <p
            v-if="indexStats.indexEnvelope.index_path"
            class="mt-0.5 font-mono text-[9px] break-all"
          >
            Looked in: {{ indexStats.indexEnvelope.index_path }}
          </p>
        </div>
        <MetricsPanel
          v-else
          unframed
          dense
          class="mt-1 border-t border-border/60 pt-1.5"
          title="Index statistics"
          :rows="indexStats.indexRows"
        />
      </template>
    </div>
  </div>
</template>
