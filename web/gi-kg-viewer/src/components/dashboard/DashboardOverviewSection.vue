<script setup lang="ts">
import { computed } from 'vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { useIndexStatsStore } from '../../stores/indexStats'
import { useShellStore } from '../../stores/shell'
import { computeArtifactMetrics } from '../../utils/metrics'
import MetricsPanel from './MetricsPanel.vue'

const shell = useShellStore()
const artifacts = useArtifactsStore()
const indexStats = useIndexStatsStore()

const artifactMetrics = computed(() => {
  const art = artifacts.displayArtifact
  if (!art) return null
  return computeArtifactMetrics(art)
})
</script>

<template>
  <div class="space-y-2">
    <h2 class="text-xs font-medium text-surface-foreground">
      Data
    </h2>
    <p class="text-[10px] leading-snug text-muted">
      Graph metrics from the Graph tab. Index stats use corpus path or server default.
    </p>
    <p
      v-if="shell.corpusPath.trim()"
      class="text-[10px] leading-snug text-surface-foreground"
    >
      <span class="text-muted">Corpus root:</span>
      {{ shell.corpusPath.trim() }}
    </p>
    <p
      v-if="shell.resolvedCorpusPath && shell.resolvedCorpusPath !== shell.corpusPath.trim()"
      class="font-mono text-[9px] leading-tight text-muted break-all"
    >
      Resolved: {{ shell.resolvedCorpusPath }}
    </p>

    <div class="rounded border border-border bg-elevated p-2 text-[10px]">
      <MetricsPanel
        v-if="artifactMetrics"
        unframed
        dense
        title="Graph"
        :rows="artifactMetrics.rows"
      />
      <p
        v-else
        class="text-center text-muted leading-snug"
      >
        Load artifacts on the Graph tab to see graph metrics.
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
