<script setup lang="ts">
/**
 * Doc types chip (#671). Replaces the multi-checkbox row in the Advanced
 * search dialog. Empty selection = all types.
 */
import { computed, ref } from 'vue'
import { useSearchStore } from '../../../stores/search'
import { useFilterChipPopover } from '../../../composables/useFilterChipPopover'

const TYPE_OPTIONS = [
  { value: 'insight', label: 'Insights' },
  { value: 'quote', label: 'Quotes' },
  { value: 'kg_entity', label: 'KG entities' },
  { value: 'kg_topic', label: 'KG topics' },
  { value: 'summary', label: 'Summary bullets' },
  { value: 'transcript', label: 'Transcript chunks' },
] as const

const search = useSearchStore()
const anchorRef = ref<HTMLButtonElement | null>(null)
const panelRef = ref<HTMLDivElement | null>(null)
const { open, toggle } = useFilterChipPopover(anchorRef, panelRef)

const isActive = computed(() => search.filters.types.length > 0)

const chipLabel = computed(() => {
  if (!isActive.value) return 'Doc types ▾'
  return `Doc types: ${search.filters.types.length} of ${TYPE_OPTIONS.length} ▾`
})

function toggleType(v: string): void {
  const i = search.filters.types.indexOf(v)
  if (i >= 0) search.filters.types.splice(i, 1)
  else search.filters.types.push(v)
}

function clearAll(): void {
  search.filters.types.splice(0, search.filters.types.length)
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
      data-testid="search-chip-doctypes"
      :aria-expanded="open"
      aria-haspopup="dialog"
      aria-label="Doc types filter"
      @click="toggle"
    >
      {{ chipLabel }}
    </button>
    <div
      v-show="open"
      ref="panelRef"
      role="dialog"
      aria-label="Doc types filter"
      data-testid="search-popover-doctypes"
      class="absolute left-0 top-full z-[40] mt-1 w-56 rounded-sm border border-border bg-elevated p-2 shadow-md"
    >
      <div class="mb-1 flex flex-wrap items-center gap-x-2 gap-y-0.5">
        <button
          v-if="isActive"
          type="button"
          class="text-[10px] text-primary underline"
          data-testid="search-popover-doctypes-clear"
          @click="clearAll"
        >
          all (no filter)
        </button>
      </div>
      <ul class="max-h-[18rem] overflow-y-auto">
        <li v-for="opt in TYPE_OPTIONS" :key="opt.value">
          <label class="flex cursor-pointer items-center gap-1.5 py-px text-[11px] leading-none text-surface-foreground">
            <input
              type="checkbox"
              class="size-3 shrink-0 rounded border-border"
              :checked="search.filters.types.includes(opt.value)"
              @change="toggleType(opt.value)"
            >
            <span class="min-w-0 flex-1 truncate">{{ opt.label }}</span>
          </label>
        </li>
      </ul>
    </div>
  </div>
</template>
