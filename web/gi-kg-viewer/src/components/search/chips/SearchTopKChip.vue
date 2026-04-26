<script setup lang="ts">
/**
 * Top-k chip (#671) — number popover. Default 10. Active when value
 * differs from the API default.
 */
import { computed } from 'vue'
import { useSearchStore } from '../../../stores/search'
import { useFilterChipPopover } from '../../../composables/useFilterChipPopover'

const TOP_K_DEFAULT = 10

const search = useSearchStore()
const { open, anchorRef, panelRef, toggle, close } = useFilterChipPopover()

const isActive = computed(() => Number(search.filters.topK) !== TOP_K_DEFAULT)

const chipLabel = computed(() => {
  if (!isActive.value) return 'Top‑k ▾'
  return `Top‑k: ${search.filters.topK} ▾`
})

function reset(): void {
  search.filters.topK = TOP_K_DEFAULT
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
      data-testid="search-chip-topk"
      :aria-expanded="open"
      aria-haspopup="dialog"
      aria-label="Top‑k results"
      @click="toggle"
    >
      {{ chipLabel }}
    </button>
    <div
      v-show="open"
      ref="panelRef"
      role="dialog"
      aria-label="Top‑k results"
      data-testid="search-popover-topk"
      class="absolute left-0 top-full z-[40] mt-1 w-44 rounded-sm border border-border bg-elevated p-2 shadow-md"
    >
      <label class="block text-[10px] font-semibold uppercase tracking-wider text-muted">
        Top‑k
      </label>
      <input
        v-model.number="search.filters.topK"
        type="number"
        min="1"
        max="100"
        class="mt-1 w-full rounded border border-border bg-elevated px-2 py-1 text-xs tabular-nums"
        data-testid="search-popover-topk-input"
        @keydown.enter="close"
      >
      <button
        v-if="isActive"
        type="button"
        class="mt-2 w-full rounded border border-border px-2 py-1 text-[11px] text-primary hover:bg-overlay"
        data-testid="search-popover-topk-reset"
        @click="reset(); close()"
      >
        Reset to {{ TOP_K_DEFAULT }}
      </button>
    </div>
  </div>
</template>
