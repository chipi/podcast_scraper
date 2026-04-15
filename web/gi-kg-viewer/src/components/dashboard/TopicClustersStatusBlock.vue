<script setup lang="ts">
import { computed } from 'vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { useShellStore } from '../../stores/shell'

const shell = useShellStore()
const artifacts = useArtifactsStore()

const showBlock = computed(
  () => Boolean(shell.healthStatus) && shell.hasCorpusPath,
)

const statusLabel = computed((): string => {
  switch (artifacts.topicClustersLoadState) {
    case 'ok':
      return 'Loaded'
    case 'missing':
      return 'Not built'
    case 'error':
      return 'Unavailable'
    case 'local_files':
      return 'Local files only'
    case 'idle':
    default:
      return 'Checking…'
  }
})

const statusClass = computed((): string => {
  switch (artifacts.topicClustersLoadState) {
    case 'ok':
      return 'text-success'
    case 'missing':
      return 'text-muted'
    case 'error':
      return 'text-danger'
    case 'local_files':
      return 'text-warning'
    case 'idle':
    default:
      return 'text-muted'
  }
})
</script>

<template>
  <div
    v-if="showBlock"
    class="rounded border border-border bg-elevated p-2 text-[10px]"
    data-testid="topic-clusters-status-block"
  >
    <h3 class="text-xs font-semibold text-surface-foreground">
      Topic clusters
    </h3>
    <p class="mt-1 leading-snug text-muted">
      From
      <code class="rounded bg-overlay px-0.5 font-mono text-[9px]">GET /api/corpus/topic-clusters</code>
      →
      <code class="rounded bg-overlay px-0.5 font-mono text-[9px]">search/topic_clusters.json</code>
      (graph overlay + optional search join). Not the same as digest topic bands.
    </p>
    <dl class="mt-1.5 space-y-1">
      <div class="flex justify-between gap-2">
        <dt class="text-muted">
          Status
        </dt>
        <dd
          class="max-w-[14rem] text-right font-medium leading-snug"
          :class="statusClass"
        >
          {{ statusLabel }}
        </dd>
      </div>
    </dl>
    <p
      v-if="artifacts.topicClustersLoadState === 'missing'"
      class="mt-1 leading-snug text-muted"
    >
      No
      <code class="rounded bg-overlay px-0.5 font-mono text-[9px]">search/topic_clusters.json</code>
      for this corpus (404). Run
      <code class="rounded bg-overlay px-0.5 font-mono text-[9px]">topic-clusters</code>
      to build it; graph works without it.
    </p>
    <p
      v-if="
        artifacts.topicClustersLoadState === 'error' && artifacts.topicClustersErrorDetail
      "
      class="mt-1 leading-snug text-danger"
    >
      {{ artifacts.topicClustersErrorDetail }}
    </p>
    <p
      v-if="artifacts.topicClustersLoadState === 'local_files'"
      class="mt-1 leading-snug text-muted"
    >
      Topic cluster overlay uses the API when you load via corpus path + API. Choose files from disk
      skips that fetch.
    </p>
    <p
      v-if="artifacts.topicClustersSchemaWarning"
      class="mt-1 leading-snug text-warning"
    >
      {{ artifacts.topicClustersSchemaWarning }}
    </p>
    <p
      v-if="
        artifacts.topicClustersLoadState === 'ok' && artifacts.topicClustersDoc?.schema_version
      "
      class="mt-1 font-mono text-[9px] leading-snug text-muted"
    >
      schema_version: {{ artifacts.topicClustersDoc?.schema_version }}
    </p>
  </div>
</template>
