<script setup lang="ts">
/**
 * Graph edge-type chip (#658). Replaces the edges section of the legacy
 * ``GraphFiltersPopover``. Shows ``Edges: K of N ▾`` when filtered.
 */
import { computed, ref } from 'vue'
import { useGraphFilterStore } from '../../../stores/graphFilters'
import { useFilterChipPopover } from '../../../composables/useFilterChipPopover'

const gf = useGraphFilterStore()
const anchorRef = ref<HTMLButtonElement | null>(null)
const panelRef = ref<HTMLDivElement | null>(null)
const { open, toggle } = useFilterChipPopover(anchorRef, panelRef)

const edgeKeys = computed(() => {
  const aet = gf.state?.allowedEdgeTypes
  if (!aet) return [] as string[]
  return Object.keys(aet).sort()
})

const totalEdges = computed(() => edgeKeys.value.length)

const enabledEdges = computed(() => {
  const aet = gf.state?.allowedEdgeTypes
  if (!aet) return 0
  let n = 0
  for (const k of edgeKeys.value) if (aet[k] !== false) n += 1
  return n
})

const isActive = computed(() => totalEdges.value > 0 && enabledEdges.value < totalEdges.value)

const chipLabel = computed(() => {
  if (!isActive.value) return 'Edges ▾'
  return `Edges: ${enabledEdges.value} of ${totalEdges.value} ▾`
})
</script>

<template>
  <div v-if="edgeKeys.length" class="relative inline-flex items-center">
    <button
      ref="anchorRef"
      type="button"
      class="inline-flex h-6 items-center rounded border px-2 text-[11px] leading-none transition-colors hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
      :class="
        isActive
          ? 'border-border font-medium text-surface-foreground'
          : 'border-border/70 text-muted'
      "
      data-testid="graph-chip-edges"
      :aria-expanded="open"
      aria-haspopup="dialog"
      aria-label="Edge type filter"
      @click="toggle"
    >
      {{ chipLabel }}
    </button>
    <div
      v-show="open && gf.state"
      ref="panelRef"
      role="dialog"
      aria-label="Edge type filter"
      data-testid="graph-popover-edges"
      class="absolute left-0 top-full z-[40] mt-1 w-60 rounded-sm border border-border bg-elevated p-2 shadow-md"
    >
      <div class="mb-1 flex items-center gap-x-2">
        <button
          type="button"
          class="text-[10px] text-primary underline"
          @click="gf.selectAllEdgeTypes()"
        >
          all
        </button>
        <button
          type="button"
          class="text-[10px] text-primary underline"
          @click="gf.deselectAllEdgeTypes()"
        >
          none
        </button>
      </div>
      <ul class="max-h-[18rem] overflow-y-auto">
        <li v-for="et in edgeKeys" :key="et">
          <label class="flex cursor-pointer items-center gap-1.5 py-px text-[11px] text-surface-foreground">
            <input
              type="checkbox"
              class="rounded border-border"
              :checked="gf.state!.allowedEdgeTypes[et]"
              @change="gf.toggleAllowedEdgeType(et)"
            >
            <span class="min-w-0 flex-1 truncate" :title="et">{{ et }}</span>
          </label>
        </li>
      </ul>
    </div>
  </div>
</template>
