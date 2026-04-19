<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { ref } from 'vue'
import GraphCanvas from './GraphCanvas.vue'
import { useArtifactsStore } from '../../stores/artifacts'
import { useGraphExpansionStore } from '../../stores/graphExpansion'

const artifacts = useArtifactsStore()
const graphExpansion = useGraphExpansionStore()
const { truncationLine: graphExpansionTruncationLine } = storeToRefs(graphExpansion)

const graphCanvasRef = ref<InstanceType<typeof GraphCanvas> | null>(null)

defineExpose({
  clearInteractionState: () => {
    graphCanvasRef.value?.clearInteractionState()
  },
})
</script>

<template>
  <div
    class="flex min-h-0 min-h-[280px] flex-1 flex-col overflow-hidden"
    data-testid="graph-tab-panel"
  >
    <p
      v-if="artifacts.siblingMergeLine && !artifacts.siblingMergeError"
      class="shrink-0 border-b border-border bg-elevated/40 px-2 py-1 text-[10px] leading-snug text-muted"
      data-testid="graph-sibling-merge-line"
    >
      {{ artifacts.siblingMergeLine }}
    </p>
    <div
      v-if="graphExpansionTruncationLine"
      class="flex shrink-0 flex-wrap items-start justify-between gap-2 border-b border-border bg-elevated/30 px-2 py-1 text-[10px] leading-snug text-muted"
      data-testid="graph-expansion-truncation-line"
    >
      <span class="min-w-0 flex-1">{{ graphExpansionTruncationLine }}</span>
      <button
        type="button"
        class="shrink-0 rounded border border-border px-1.5 py-0.5 text-[9px] font-medium hover:bg-overlay"
        data-testid="graph-expansion-truncation-dismiss"
        @click="graphExpansion.clearTruncationLine()"
      >
        Dismiss
      </button>
    </div>
    <GraphCanvas
      v-if="artifacts.displayArtifact"
      ref="graphCanvasRef"
      class="min-h-0 flex-1"
    />
    <div
      v-if="!artifacts.displayArtifact"
      class="flex min-h-[280px] flex-1 items-center justify-center rounded border border-dashed border-border bg-surface p-8 text-sm text-muted"
    >
      <span class="max-w-md text-center">
        With a healthy API, set <strong>Corpus path</strong> to auto-load all GI/KG; or use
        <strong>List</strong> and <strong>Load into graph</strong> on the <strong>Dashboard</strong>
        corpus workspace. Offline: <strong>Choose files…</strong> on the <strong>status bar</strong>.
      </span>
    </div>
  </div>
</template>
