<script setup lang="ts">
/**
 * Min-confidence chip (Search v3 §S1 — Explore merge). Client-side filter over
 * ``results``. Empty = no filter; a numeric string like "0.7" means ≥0.7. Same
 * accuracy caveat as SearchTopicChip — narrows the returned top-K rather than
 * driving retrieval.
 */
import { computed, ref } from 'vue'
import { useSearchStore } from '../../../stores/search'
import { useFilterChipPopover } from '../../../composables/useFilterChipPopover'

const search = useSearchStore()
const anchorRef = ref<HTMLButtonElement | null>(null)
const panelRef = ref<HTMLDivElement | null>(null)
const { open, toggle, close } = useFilterChipPopover(anchorRef, panelRef)

const isActive = computed(() => {
  const raw = search.filters.minConfidence.trim()
  if (!raw) return false
  const n = Number(raw)
  return Number.isFinite(n)
})

const chipLabel = computed(() => {
  if (!isActive.value) return 'Min conf ▾'
  return `Min conf: ${search.filters.minConfidence.trim()} ▾`
})

function clear(): void {
  search.filters.minConfidence = ''
}
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
      data-testid="search-chip-min-confidence"
      :aria-expanded="open"
      aria-haspopup="dialog"
      aria-label="Minimum insight confidence"
      @click="toggle"
    >
      {{ chipLabel }}
    </button>
    <div
      v-show="open"
      ref="panelRef"
      role="dialog"
      aria-label="Minimum insight confidence"
      data-testid="search-popover-min-confidence"
      class="absolute left-0 top-full z-[40] mt-1 w-56 rounded-sm border border-border bg-elevated p-2 shadow-md"
    >
      <label class="block text-[10px] font-semibold uppercase tracking-wider text-muted">
        Min confidence
      </label>
      <input
        v-model="search.filters.minConfidence"
        type="number"
        min="0"
        max="1"
        step="0.05"
        placeholder="0.0 – 1.0"
        class="mt-1 w-full rounded border border-border bg-elevated px-2 py-1 text-xs tabular-nums"
        data-testid="search-popover-min-confidence-input"
        @keydown.enter="close"
      >
      <p class="mt-2 text-[10px] text-muted">
        Client-side over top-K. Hits without a confidence value are dropped when set.
      </p>
      <button
        v-if="isActive"
        type="button"
        class="mt-2 w-full rounded border border-border px-2 py-1 text-[11px] text-primary hover:bg-overlay"
        data-testid="search-popover-min-confidence-clear"
        @click="clear(); close()"
      >
        Clear
      </button>
    </div>
  </div>
</template>
