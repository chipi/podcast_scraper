<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import type { IndexStatsEnvelope } from '../../api/indexStatsApi'
import { fetchIndexStats } from '../../api/indexStatsApi'
import { useArtifactsStore } from '../../stores/artifacts'
import { useShellStore } from '../../stores/shell'
import { formatBytes } from '../../utils/formatting'
import type { MetricRow } from '../../utils/metrics'
import { computeArtifactMetrics } from '../../utils/metrics'
import CollapsibleSection from '../shared/CollapsibleSection.vue'
import MetricsPanel from './MetricsPanel.vue'
import TypeCountBarChart from './TypeCountBarChart.vue'

const shell = useShellStore()
const artifacts = useArtifactsStore()

const indexEnvelope = ref<IndexStatsEnvelope | null>(null)
const indexLoading = ref(false)
const indexError = ref<string | null>(null)

const artifactMetrics = computed(() => {
  const art = artifacts.displayArtifact
  if (!art) return null
  return computeArtifactMetrics(art)
})

const indexRows = computed((): MetricRow[] => {
  const env = indexEnvelope.value
  if (!env?.available || !env.stats) return []
  const s = env.stats
  const feeds =
    s.feeds_indexed.length > 0 ? s.feeds_indexed.join(', ') : '—'
  return [
    { k: 'Index path', v: env.index_path || '—' },
    { k: 'Total vectors', v: String(s.total_vectors) },
    { k: 'Embedding model', v: s.embedding_model || '—' },
    { k: 'Embedding dim', v: String(s.embedding_dim) },
    { k: 'Last updated', v: s.last_updated || '—' },
    { k: 'On-disk size', v: formatBytes(s.index_size_bytes) },
    { k: 'Feeds indexed', v: feeds },
  ]
})

const indexDocTypeCounts = computed(
  () => indexEnvelope.value?.stats?.doc_type_counts ?? {},
)

async function refreshIndexStats(): Promise<void> {
  indexError.value = null
  if (!shell.healthStatus) {
    indexEnvelope.value = null
    return
  }
  indexLoading.value = true
  try {
    const path = shell.hasCorpusPath ? shell.corpusPath.trim() : undefined
    indexEnvelope.value = await fetchIndexStats(path)
  } catch (e) {
    indexEnvelope.value = null
    indexError.value = e instanceof Error ? e.message : String(e)
  } finally {
    indexLoading.value = false
  }
}

onMounted(() => {
  void refreshIndexStats()
})

watch(
  () => [shell.corpusPath, shell.healthStatus] as const,
  () => {
    void refreshIndexStats()
  },
)

defineExpose({ refreshIndexStats })
</script>

<template>
  <div class="space-y-4">
    <!-- TOP: Corpus overview + Vector index | Loaded graph  (collapsible) -->
    <CollapsibleSection title="Overview" :default-open="true">
      <div class="flex flex-col gap-4 lg:flex-row lg:items-start">
        <!-- Left column: Corpus overview + Vector index -->
        <div class="min-w-0 flex-1 space-y-4">
          <div class="rounded border border-border bg-surface p-3 text-sm text-surface-foreground">
            <h2 class="mb-1 text-sm font-semibold">
              Corpus overview
            </h2>
            <p class="text-xs text-muted">
              Graph metrics from the Graph tab. Index stats use corpus path or server default.
            </p>
            <p
              v-if="shell.corpusPath.trim()"
              class="mt-1.5 text-xs"
            >
              <span class="text-muted">Root:</span>
              {{ shell.corpusPath.trim() }}
            </p>
            <p
              v-if="shell.resolvedCorpusPath && shell.resolvedCorpusPath !== shell.corpusPath.trim()"
              class="mt-0.5 text-[10px] text-muted"
            >
              Resolved: {{ shell.resolvedCorpusPath }}
            </p>
          </div>

          <div
            class="rounded border border-border bg-surface p-3"
            :aria-busy="indexLoading ? 'true' : undefined"
          >
            <div class="mb-2 flex flex-wrap items-center justify-between gap-2">
              <h3 class="text-sm font-semibold text-surface-foreground">
                Vector index
              </h3>
              <button
                type="button"
                class="rounded border border-border px-2 py-1 text-xs hover:bg-overlay disabled:opacity-40"
                :disabled="!shell.healthStatus || indexLoading"
                @click="refreshIndexStats"
              >
                {{ indexLoading ? 'Loading…' : 'Refresh' }}
              </button>
            </div>
            <p
              v-if="!shell.healthStatus"
              class="text-xs text-muted"
            >
              Needs the API.
            </p>
            <p
              v-else-if="indexError"
              class="text-xs text-danger"
            >
              {{ indexError }}
            </p>
            <p
              v-else-if="indexLoading && !indexEnvelope"
              class="text-xs text-muted"
            >
              Loading…
            </p>
            <template v-else-if="indexEnvelope">
              <div
                v-if="!indexEnvelope.available"
                class="text-xs text-muted"
              >
                <p>
                  <span class="font-medium text-surface-foreground">No index</span>
                  —
                  {{ indexEnvelope.reason === 'no_index'
                    ? 'No FAISS data at the expected path (run corpus indexing).'
                    : indexEnvelope.reason === 'no_corpus_path'
                      ? 'Set corpus root or start the server with --output-dir.'
                      : indexEnvelope.reason === 'faiss_unavailable'
                        ? 'FAISS is not installed in this Python environment.'
                        : (indexEnvelope.reason || 'Unavailable') }}
                </p>
                <p
                  v-if="indexEnvelope.index_path"
                  class="mt-1 font-mono text-[10px]"
                >
                  Looked in: {{ indexEnvelope.index_path }}
                </p>
              </div>
              <MetricsPanel
                v-else
                class="mt-2"
                title="Index statistics"
                :rows="indexRows"
              />
            </template>
          </div>
        </div>

        <!-- Right column: Loaded graph -->
        <aside class="w-full shrink-0 lg:w-80">
          <MetricsPanel
            v-if="artifactMetrics"
            title="Loaded graph"
            :rows="artifactMetrics.rows"
          />
          <div
            v-else
            class="rounded border border-dashed border-border bg-surface p-6 text-center text-sm text-muted"
          >
            Load artifacts on the Graph tab to see graph metrics.
          </div>
        </aside>
      </div>
    </CollapsibleSection>

    <!-- BOTTOM: Charts (full width) -->
    <div class="grid gap-4 lg:grid-cols-2">
      <template v-if="artifactMetrics">
        <TypeCountBarChart
          title="Node types (visual groups)"
          :counts="artifactMetrics.visualNodeTypeCounts"
        />
      </template>
      <div
        v-else
        class="rounded border border-dashed border-border bg-surface p-6 text-center text-sm text-muted"
      >
        Load artifacts on the Graph tab to see charts.
      </div>

      <template v-if="indexEnvelope?.available && indexEnvelope.stats">
        <TypeCountBarChart
          title="Indexed document types"
          :counts="indexDocTypeCounts"
        />
      </template>
    </div>
  </div>
</template>
