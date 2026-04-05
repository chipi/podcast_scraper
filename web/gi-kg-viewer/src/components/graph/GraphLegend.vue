<script setup lang="ts">
import { computed } from 'vue'
import { graphNodeLegendLabel, graphNodeTypeStyles, graphNodeTypesOrdered, GRAPH_NODE_UNKNOWN_FILL } from '../../utils/colors'
import { useGraphFilterStore } from '../../stores/graphFilters'

const gf = useGraphFilterStore()

function onRow(key: string): void {
  gf.onLegendClick(key)
}

const activeLegendKey = computed(() => gf.state?.legendSoloVisual ?? null)

const legendSummary = computed(() => {
  const k = activeLegendKey.value
  if (!k) return 'all types'
  return graphNodeLegendLabel(k)
})

defineExpose({ legendSummary })
</script>

<template>
  <div class="text-surface-foreground">
    <div class="flex flex-wrap gap-1.5">
      <button
        v-for="t in graphNodeTypesOrdered"
        :key="t"
        type="button"
        class="flex items-center gap-1 rounded px-1.5 py-0.5 text-xs hover:bg-overlay"
        :class="{ 'ring-1 ring-primary': activeLegendKey === t }"
        @click="onRow(t)"
      >
        <span
          class="inline-block h-3 w-3 shrink-0 rounded border"
          :style="{
            backgroundColor: graphNodeTypeStyles[t].background,
            borderColor: graphNodeTypeStyles[t].border,
          }"
          aria-hidden="true"
        />
        <span>{{ graphNodeLegendLabel(t) }}</span>
      </button>
      <button
        type="button"
        class="flex items-center gap-1 rounded px-1.5 py-0.5 text-xs hover:bg-overlay"
        @click="onRow('__reset__')"
      >
        <span
          class="inline-block h-3 w-3 shrink-0 rounded border"
          :style="{
            backgroundColor: GRAPH_NODE_UNKNOWN_FILL,
            borderColor: '#5c636a',
          }"
          aria-hidden="true"
        />
        <span>Reset</span>
      </button>
    </div>
  </div>
</template>
