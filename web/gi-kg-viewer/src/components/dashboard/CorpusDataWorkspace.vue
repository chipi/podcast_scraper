<script setup lang="ts">
import { useArtifactsStore } from '../../stores/artifacts'
import { useShellStore } from '../../stores/shell'
import HelpTip from '../shared/HelpTip.vue'
import DashboardOverviewSection from './DashboardOverviewSection.vue'

const emit = defineEmits<{
  'go-graph': []
}>()

const shell = useShellStore()
const artifacts = useArtifactsStore()

async function onLoadIntoGraphClick(): Promise<void> {
  await artifacts.loadSelected()
  if (artifacts.displayArtifact) {
    emit('go-graph')
  }
}
</script>

<template>
  <section class="space-y-2" data-testid="corpus-data-workspace">
    <div class="mb-1.5 flex items-center gap-1">
      <h2 class="text-xs font-medium text-kg">
        Corpus artifacts
      </h2>
      <HelpTip>
        Set corpus root on the <strong>status bar</strong>. When the API is healthy, the viewer
        lists artifacts and can load GI/KG into the merged graph.
      </HelpTip>
    </div>
    <div class="mt-1.5 flex flex-wrap gap-1">
      <button
        type="button"
        class="rounded bg-primary px-2 py-0.5 text-[10px] font-medium text-primary-foreground hover:opacity-90 disabled:opacity-40"
        :disabled="!shell.hasCorpusPath || shell.artifactsLoading"
        @click="shell.fetchArtifactList()"
      >
        List
      </button>
    </div>
    <p v-if="shell.artifactsLoading" class="mt-1 text-[10px] text-muted">
      Loading…
    </p>
    <template v-else>
      <div
        v-if="shell.corpusHints.length"
        class="mt-1.5 rounded border border-warning/40 bg-warning/10 px-2 py-1.5 text-[10px] text-surface-foreground"
        role="status"
      >
        <p class="font-medium text-warning">
          Corpus path hint
        </p>
        <ul class="mt-0.5 list-inside list-disc text-muted">
          <li v-for="(h, i) in shell.corpusHints" :key="i">
            {{ h }}
          </li>
        </ul>
      </div>
      <div
        v-if="shell.artifactList.length"
        class="mt-1.5 space-y-1"
      >
        <div class="flex flex-wrap items-center gap-1">
          <button
            type="button"
            class="rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay"
            @click="artifacts.selectAllListed(shell.artifactList.map((a) => a.relative_path))"
          >
            All
          </button>
          <button
            type="button"
            class="rounded border border-border px-1.5 py-0.5 text-[10px] hover:bg-overlay"
            @click="artifacts.deselectAllListed()"
          >
            None
          </button>
          <button
            type="button"
            class="rounded border border-border px-2 py-0.5 text-[10px] font-medium hover:bg-overlay disabled:opacity-40"
            :disabled="artifacts.selectedRelPaths.length === 0 || artifacts.loading"
            @click="onLoadIntoGraphClick()"
          >
            {{ artifacts.loading ? 'Loading…' : 'Load into graph' }}
          </button>
        </div>
        <div class="overflow-y-auto rounded border border-border bg-elevated p-1 text-[11px]">
          <label
            v-for="a in shell.artifactList"
            :key="a.relative_path"
            class="flex cursor-pointer items-start gap-1 py-0.5 hover:bg-overlay"
          >
            <input
              type="checkbox"
              class="mt-0.5 rounded border-border"
              :checked="artifacts.selectedRelPaths.includes(a.relative_path)"
              @change="artifacts.toggleSelection(a.relative_path)"
            >
            <span class="break-all">
              <span :class="a.kind === 'gi' ? 'text-gi' : 'text-kg'">{{ a.kind }}</span>
              {{ a.relative_path }}
            </span>
          </label>
        </div>
      </div>
      <p
        v-else-if="shell.artifactCount !== null && shell.artifactCount === 0"
        class="mt-1 text-[10px] text-muted"
      >
        No artifacts found.
      </p>
    </template>
    <p v-if="shell.artifactsError" class="mt-1 text-[10px] text-danger">
      {{ shell.artifactsError }}
    </p>
    <p v-if="artifacts.loadError" class="mt-1 text-[10px] text-danger">
      {{ artifacts.loadError }}
    </p>
    <p v-if="artifacts.displayArtifact" class="mt-1 text-[10px] text-muted">
      {{ artifacts.displayArtifact.name }} ({{ artifacts.displayArtifact.nodes }} nodes)
    </p>
    <h2 class="mt-3 text-xs font-medium text-surface-foreground">
      API
    </h2>
    <p class="text-[10px] leading-snug text-muted">
      Capability flags from
      <code class="rounded bg-overlay px-0.5 font-mono text-[9px]">GET /api/health</code>
      (graph, search, index routes, catalog). FAISS availability is separate — see the
      <strong>Vector index</strong> card below.
    </p>
    <div class="rounded border border-border bg-elevated p-2 text-[10px]">
      <dl class="space-y-1">
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Health
          </dt>
          <dd
            v-if="shell.healthStatus"
            class="font-medium text-success"
          >
            {{ shell.healthStatusDisplay }}
          </dd>
          <dd
            v-else-if="shell.healthError"
            class="font-medium text-danger"
          >
            {{ shell.healthError }}
          </dd>
          <dd
            v-else
            class="text-muted"
          >
            Checking…
          </dd>
        </div>
      </dl>
      <dl
        v-if="shell.healthStatus"
        class="mt-1.5 space-y-1 border-t border-border/60 pt-1.5"
      >
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Artifacts (graph)
          </dt>
          <dd
            :class="
              shell.artifactsApiAvailable !== false ? 'text-success' : 'text-danger'
            "
          >
            {{ shell.artifactsApiAvailable !== false ? 'Yes' : 'No' }}
          </dd>
        </div>
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Semantic search
          </dt>
          <dd
            :class="shell.searchApiAvailable !== false ? 'text-success' : 'text-danger'"
          >
            {{ shell.searchApiAvailable !== false ? 'Yes' : 'No' }}
          </dd>
        </div>
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Graph explore
          </dt>
          <dd
            :class="shell.exploreApiAvailable !== false ? 'text-success' : 'text-danger'"
          >
            {{ shell.exploreApiAvailable !== false ? 'Yes' : 'No' }}
          </dd>
        </div>
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Index routes
          </dt>
          <dd
            :class="
              shell.indexRoutesApiAvailable !== false ? 'text-success' : 'text-danger'
            "
          >
            {{ shell.indexRoutesApiAvailable !== false ? 'Yes' : 'No' }}
          </dd>
        </div>
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Corpus metrics
          </dt>
          <dd
            :class="
              shell.corpusMetricsApiAvailable !== false ? 'text-success' : 'text-danger'
            "
          >
            {{ shell.corpusMetricsApiAvailable !== false ? 'Yes' : 'No' }}
          </dd>
        </div>
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Library API
          </dt>
          <dd
            :class="
              shell.corpusLibraryApiAvailable ? 'text-success' : 'text-danger'
            "
          >
            {{ shell.corpusLibraryApiAvailable ? 'Yes' : 'No' }}
          </dd>
        </div>
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Digest API
          </dt>
          <dd
            :class="
              shell.corpusDigestApiAvailable ? 'text-success' : 'text-danger'
            "
          >
            {{ shell.corpusDigestApiAvailable ? 'Yes' : 'No' }}
          </dd>
        </div>
        <div class="flex justify-between gap-2">
          <dt class="text-muted">
            Binary (covers)
          </dt>
          <dd
            :class="
              shell.corpusBinaryApiAvailable !== false ? 'text-success' : 'text-danger'
            "
          >
            {{ shell.corpusBinaryApiAvailable !== false ? 'Yes' : 'No' }}
          </dd>
        </div>
      </dl>
      <p
        v-if="shell.healthStatus && !shell.corpusLibraryApiAvailable"
        class="mt-1.5 text-[10px] leading-snug text-danger"
      >
        Corpus Library API not advertised — upgrade/restart the Python server (Library tab).
      </p>
      <button
        type="button"
        class="mt-1.5 rounded border border-border px-2 py-0.5 hover:bg-overlay"
        @click="shell.fetchHealth()"
      >
        Retry health
      </button>
    </div>
    <p
      v-if="shell.healthError"
      class="rounded border border-border bg-overlay p-1.5 text-[10px] text-muted"
    >
      Offline: use <strong>Files</strong> on the status bar to load GI/KG JSON.
    </p>
    <DashboardOverviewSection />
  </section>
</template>
