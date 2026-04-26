<script setup lang="ts">
/**
 * Graph node-type chip (#658). Replaces the inline Types row + per-type
 * checkbox grid that previously lived in ``GraphCanvas.vue``. Chip label
 * shows ``Types: K of N ▾`` when filtered.
 */
import { computed } from 'vue'
import { useGraphFilterStore } from '../../../stores/graphFilters'
import { useFilterChipPopover } from '../../../composables/useFilterChipPopover'
import { graphNodeFill, graphNodeLegendLabel } from '../../../utils/colors'

const props = defineProps<{
  /** Per-type histogram counts (computed by GraphCanvas via ``visualNodeTypeCounts``). */
  typeHistogramCounts: Record<string, number>
}>()

const gf = useGraphFilterStore()
const { open, anchorRef, panelRef, toggle } = useFilterChipPopover()

const typeKeys = computed(() => {
  const st = gf.state
  if (!st) return [] as string[]
  return Object.keys(st.allowedTypes).sort()
})

const totalTypes = computed(() => typeKeys.value.length)

const enabledCount = computed(() => {
  const st = gf.state
  if (!st) return 0
  let n = 0
  for (const k of typeKeys.value) {
    if (st.allowedTypes[k] !== false) n += 1
  }
  return n
})

const isActive = computed(() => gf.graphTypesDeviateFromDefaults)

const chipLabel = computed(() => {
  if (!isActive.value) return 'Types ▾'
  return `Types: ${enabledCount.value} of ${totalTypes.value} ▾`
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
      data-testid="graph-chip-types"
      :aria-expanded="open"
      aria-haspopup="dialog"
      aria-label="Node type filter"
      @click="toggle"
    >
      {{ chipLabel }}
    </button>
    <div
      v-show="open && gf.state"
      ref="panelRef"
      role="dialog"
      aria-label="Node type filter"
      data-testid="graph-popover-types"
      class="absolute left-0 top-full z-[40] mt-1 w-60 rounded-sm border border-border bg-elevated p-2 shadow-md"
    >
      <div class="mb-1 flex flex-wrap items-center gap-x-2 gap-y-0.5">
        <button
          type="button"
          class="text-[10px] text-primary underline"
          @click="gf.selectAllTypes()"
        >
          all
        </button>
        <button
          type="button"
          class="text-[10px] text-primary underline"
          @click="gf.deselectAllTypes()"
        >
          none
        </button>
        <button
          v-if="isActive"
          type="button"
          class="text-[10px] text-primary underline"
          data-testid="graph-types-reset"
          @click="gf.resetGraphTypeVisibilityDefaults()"
        >
          reset
        </button>
      </div>
      <ul class="max-h-[18rem] overflow-y-auto">
        <li v-for="t in typeKeys" :key="t">
          <label class="flex cursor-pointer items-center gap-1.5 py-px text-[11px] leading-none text-surface-foreground">
            <input
              type="checkbox"
              class="size-3 shrink-0 rounded border-border"
              :checked="gf.state!.allowedTypes[t]"
              @change="gf.toggleAllowedType(t)"
            >
            <span
              class="h-2.5 w-2.5 shrink-0 rounded-sm ring-1 ring-black/15 dark:ring-white/20"
              :style="{ backgroundColor: graphNodeFill(String(t)) }"
              :title="`${String(t)} (node fill)`"
              aria-hidden="true"
            />
            <span class="min-w-0 flex-1 truncate">{{ graphNodeLegendLabel(t) }}</span>
            <span class="font-medium text-muted">({{ props.typeHistogramCounts[t] ?? 0 }})</span>
          </label>
        </li>
      </ul>
    </div>
  </div>
</template>
