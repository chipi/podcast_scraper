<script setup lang="ts">
/**
 * Graph "Sources" chip (#658). Combines GI-layer / KG-layer / Hide
 * ungrounded toggles into one popover. Visible only when ``kind === 'both'``
 * (GI+KG); pure-GI / pure-KG views render no Sources chip.
 */
import { computed, ref } from 'vue'
import { useGraphFilterStore } from '../../../stores/graphFilters'
import { useFilterChipPopover } from '../../../composables/useFilterChipPopover'

const gf = useGraphFilterStore()
const anchorRef = ref<HTMLButtonElement | null>(null)
const panelRef = ref<HTMLDivElement | null>(null)
const { open, toggle } = useFilterChipPopover(anchorRef, panelRef)

const kind = computed(() => gf.fullArtifact?.kind)

const isActive = computed(() => {
  const st = gf.state
  if (!st) return false
  if (kind.value === 'both' && (!st.showGiLayer || !st.showKgLayer)) return true
  if ((kind.value === 'gi' || kind.value === 'both') && st.hideUngroundedInsights) return true
  return false
})

const chipLabel = computed(() => {
  const st = gf.state
  if (!st) return 'Sources ▾'
  const parts: string[] = []
  if (kind.value === 'both' && st.showGiLayer && !st.showKgLayer) parts.push('GI only')
  if (kind.value === 'both' && !st.showGiLayer && st.showKgLayer) parts.push('KG only')
  if (kind.value === 'both' && !st.showGiLayer && !st.showKgLayer) parts.push('none')
  if ((kind.value === 'gi' || kind.value === 'both') && st.hideUngroundedInsights) {
    parts.push(parts.length ? '+grounded' : 'grounded')
  }
  return parts.length ? `Sources: ${parts.join(' ')} ▾` : 'Sources ▾'
})
</script>

<template>
  <div v-if="kind === 'both'" class="relative inline-flex items-center">
    <button
      ref="anchorRef"
      type="button"
      class="inline-flex h-6 items-center rounded border px-2 text-[11px] leading-none transition-colors hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
      :class="
        isActive
          ? 'border-border font-medium text-surface-foreground'
          : 'border-border/70 text-muted'
      "
      data-testid="graph-chip-sources"
      :aria-expanded="open"
      aria-haspopup="dialog"
      aria-label="Sources filter"
      @click="toggle"
    >
      {{ chipLabel }}
    </button>
    <div
      v-show="open && gf.state"
      ref="panelRef"
      role="dialog"
      aria-label="Sources filter"
      data-testid="graph-popover-sources"
      class="absolute left-0 top-full z-[40] mt-1 w-56 rounded-sm border border-border bg-elevated p-2 shadow-md"
    >
      <label class="flex cursor-pointer items-center gap-1.5 text-[11px] text-surface-foreground">
        <input
          type="checkbox"
          class="rounded border-border"
          :checked="gf.state!.showGiLayer"
          @change="gf.setShowGiLayer(!gf.state!.showGiLayer)"
        >
        <span>GI</span>
      </label>
      <label class="mt-1 flex cursor-pointer items-center gap-1.5 text-[11px] text-surface-foreground">
        <input
          type="checkbox"
          class="rounded border-border"
          :checked="gf.state!.showKgLayer"
          @change="gf.setShowKgLayer(!gf.state!.showKgLayer)"
        >
        <span>KG</span>
      </label>
      <label class="mt-2 flex cursor-pointer items-center gap-1.5 border-t border-border pt-2 text-[11px] text-surface-foreground">
        <input
          type="checkbox"
          class="rounded border-border"
          :checked="gf.state!.hideUngroundedInsights"
          @change="gf.setHideUngrounded(!gf.state!.hideUngroundedInsights)"
        >
        <span>Hide ungrounded</span>
      </label>
    </div>
  </div>
</template>
