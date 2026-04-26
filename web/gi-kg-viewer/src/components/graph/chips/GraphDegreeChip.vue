<script setup lang="ts">
/**
 * Graph degree-bucket chip (#658). Wraps the degree bucket grid that
 * previously lived in the ⚙ popover. Active state lives on
 * ``useGraphExplorerStore.activeDegreeBucket`` (unchanged from today).
 */
import { computed, ref } from 'vue'
import { storeToRefs } from 'pinia'
import { useGraphExplorerStore } from '../../../stores/graphExplorer'
import { useFilterChipPopover } from '../../../composables/useFilterChipPopover'
import { DEGREE_BUCKET_ORDER } from '../../../utils/graphDegreeBuckets'

const props = defineProps<{
  degreeHistogramCounts: Record<string, number>
}>()

const ge = useGraphExplorerStore()
const { activeDegreeBucket } = storeToRefs(ge)
const anchorRef = ref<HTMLButtonElement | null>(null)
const panelRef = ref<HTMLDivElement | null>(null)
const { open, toggle } = useFilterChipPopover(anchorRef, panelRef)

const isActive = computed(
  () => activeDegreeBucket.value != null && activeDegreeBucket.value !== '',
)

const chipLabel = computed(() => {
  if (!isActive.value) return 'Degree ▾'
  return `Degree: ${activeDegreeBucket.value} ▾`
})
</script>

<template>
  <div class="relative inline-flex items-center">
    <button
      ref="anchorRef"
      type="button"
      class="inline-flex h-6 items-center rounded border px-2 text-[11px] leading-none transition-colors hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
      :class="
        isActive
          ? 'border-border font-medium text-surface-foreground'
          : 'border-border/70 text-muted'
      "
      data-testid="graph-chip-degree"
      :aria-expanded="open"
      aria-haspopup="dialog"
      aria-label="Degree bucket filter"
      @click="toggle"
    >
      {{ chipLabel }}
    </button>
    <div
      v-show="open"
      ref="panelRef"
      role="dialog"
      aria-label="Degree bucket filter"
      data-testid="graph-popover-degree"
      class="absolute left-0 top-full z-[40] mt-1 w-44 rounded-sm border border-border bg-elevated p-2 shadow-md"
    >
      <p class="mb-1 text-[10px] text-muted">
        Filter to nodes by edge count.
      </p>
      <div class="grid grid-cols-2 gap-0.5">
        <button
          v-for="bid in DEGREE_BUCKET_ORDER"
          :key="bid"
          type="button"
          class="rounded border px-1 py-px text-[10px] leading-tight hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          :class="
            activeDegreeBucket === bid
              ? 'border-primary bg-primary/15 font-medium'
              : 'border-border'
          "
          :aria-pressed="activeDegreeBucket === bid"
          @click="ge.toggleDegreeBucket(bid)"
        >
          {{ bid }}
          <span class="text-muted">({{ props.degreeHistogramCounts[bid] ?? 0 }})</span>
        </button>
      </div>
      <button
        v-if="isActive"
        type="button"
        class="mt-2 w-full rounded border border-border px-1 py-px text-[10px] leading-tight text-primary hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
        aria-label="Clear degree filter"
        @click="ge.clearDegreeBucket()"
      >
        Clear
      </button>
    </div>
  </div>
</template>
