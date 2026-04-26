<script setup lang="ts">
/**
 * Unified graph filter chip bar (#658). Replaces the inline Types row
 * and the legacy ``GraphFiltersPopover`` ⚙ button in
 * ``GraphCanvas.vue``. Layout:
 *
 *   [Types] [Feed] [Sources*]  |  [Edges] [Degree]      [× reset all]
 *
 * Sources chip renders only when ``kind === 'both'``.
 */
import { computed } from 'vue'
import { useGraphFilterStore } from '../../stores/graphFilters'
import { useGraphExplorerStore } from '../../stores/graphExplorer'
import GraphTypesChip from './chips/GraphTypesChip.vue'
import GraphFeedChip from './chips/GraphFeedChip.vue'
import GraphSourcesChip from './chips/GraphSourcesChip.vue'
import GraphEdgesChip from './chips/GraphEdgesChip.vue'
import GraphDegreeChip from './chips/GraphDegreeChip.vue'

const props = defineProps<{
  typeHistogramCounts: Record<string, number>
  degreeHistogramCounts: Record<string, number>
}>()

const gf = useGraphFilterStore()
const ge = useGraphExplorerStore()

const anyActive = computed(() => {
  if (gf.filtersAreActive) return true
  if (ge.activeDegreeBucket) return true
  return false
})

function resetAll(): void {
  gf.resetAllFilters()
  ge.clearDegreeBucket()
}
</script>

<template>
  <div
    v-if="gf.state"
    class="flex min-h-7 flex-wrap items-center gap-x-1.5 gap-y-0.5 border-b border-border px-2 py-1 text-surface-foreground"
    data-testid="graph-filter-bar"
  >
    <GraphTypesChip :type-histogram-counts="props.typeHistogramCounts" />
    <GraphFeedChip />
    <GraphSourcesChip />
    <span aria-hidden="true" class="mx-0.5 h-4 w-px bg-border" />
    <GraphEdgesChip />
    <GraphDegreeChip :degree-histogram-counts="props.degreeHistogramCounts" />
    <button
      v-if="anyActive"
      type="button"
      class="ml-auto inline-flex h-6 items-center rounded border border-border px-2 text-[11px] leading-none text-primary hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
      data-testid="graph-chip-reset-all"
      aria-label="Reset all graph filters"
      @click="resetAll"
    >
      × reset all
    </button>
  </div>
</template>
