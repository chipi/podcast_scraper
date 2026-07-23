<script setup lang="ts">
/**
 * Speaker-contains chip (Search v3 §S1 — Explore merge). Server-side filter:
 * passes as ``speaker`` query param to /api/search.
 */
import { computed, ref } from 'vue'
import { useSearchStore } from '../../../stores/search'
import { useFilterChipPopover } from '../../../composables/useFilterChipPopover'

const search = useSearchStore()
const anchorRef = ref<HTMLButtonElement | null>(null)
const panelRef = ref<HTMLDivElement | null>(null)
const { open, toggle, close } = useFilterChipPopover(anchorRef, panelRef)

const isActive = computed(() => Boolean(search.filters.speaker.trim()))

const chipLabel = computed(() => {
  const value = search.filters.speaker.trim()
  if (!value) return 'Speaker ▾'
  const truncated = value.length > 20 ? `${value.slice(0, 20)}…` : value
  return `Speaker: ${truncated} ▾`
})

function clear(): void {
  search.filters.speaker = ''
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
      data-testid="search-chip-speaker-contains"
      :aria-expanded="open"
      aria-haspopup="dialog"
      aria-label="Speaker contains"
      @click="toggle"
    >
      {{ chipLabel }}
    </button>
    <div
      v-show="open"
      ref="panelRef"
      role="dialog"
      aria-label="Speaker contains"
      data-testid="search-popover-speaker-contains"
      class="absolute left-0 top-full z-[40] mt-1 w-56 rounded-sm border border-border bg-elevated p-2 shadow-md"
    >
      <label class="block text-[10px] font-semibold uppercase tracking-wider text-muted">
        Speaker contains
      </label>
      <input
        v-model="search.filters.speaker"
        type="text"
        placeholder="e.g. altman"
        class="mt-1 w-full rounded border border-border bg-elevated px-2 py-1 text-xs"
        data-testid="search-popover-speaker-contains-input"
        @keydown.enter="close"
      >
      <p class="mt-2 text-[10px] text-muted">
        Server-side; substring match on speaker names.
      </p>
      <button
        v-if="isActive"
        type="button"
        class="mt-2 w-full rounded border border-border px-2 py-1 text-[11px] text-primary hover:bg-overlay"
        data-testid="search-popover-speaker-contains-clear"
        @click="clear(); close()"
      >
        Clear
      </button>
    </div>
  </div>
</template>
