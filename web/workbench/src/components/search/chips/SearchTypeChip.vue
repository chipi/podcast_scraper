<script setup lang="ts">
/**
 * Type chip (UXS-019): which content types a search covers — Podcasts (default), News, or both.
 * Shown only when the served corpus holds news (health ``content_types``). Choosing a type re-runs
 * the current query, since it replaces the whole result set.
 */
import { computed, ref } from 'vue'
import { useSearchStore } from '../../../stores/search'
import { useShellStore } from '../../../stores/shell'
import { useFilterChipPopover } from '../../../composables/useFilterChipPopover'

const PODCASTS = ['podcast_episode']
const NEWS = ['news_article']
const BOTH = ['podcast_episode', 'news_article']

const OPTIONS = [
  { key: 'podcasts', label: 'Podcasts', types: PODCASTS },
  { key: 'news', label: 'News', types: NEWS },
  { key: 'both', label: 'Podcasts and news', types: BOTH },
] as const

const search = useSearchStore()
const shell = useShellStore()
const anchorRef = ref<HTMLButtonElement | null>(null)
const panelRef = ref<HTMLDivElement | null>(null)
const { open, toggle, close } = useFilterChipPopover(anchorRef, panelRef)

const current = computed(() => {
  const set = new Set(search.filters.contentTypes)
  return (
    OPTIONS.find((o) => o.types.length === set.size && o.types.every((t) => set.has(t))) ??
    OPTIONS[0]
  )
})
const isActive = computed(() => current.value.key !== 'podcasts')
const chipLabel = computed(() => (isActive.value ? `Type: ${current.value.label} ▾` : 'Type ▾'))

async function choose(key: string): Promise<void> {
  const opt = OPTIONS.find((o) => o.key === key) ?? OPTIONS[0]
  close()
  if (opt.key === current.value.key) return
  search.filters.contentTypes = [...opt.types]
  if (search.query.trim() && shell.corpusPath?.trim()) {
    await search.runSearch(shell.corpusPath)
  }
}
</script>

<template>
  <div v-if="shell.hasNews" class="relative inline-flex items-center">
    <button
      ref="anchorRef"
      type="button"
      class="inline-flex h-6 items-center rounded border px-2 text-[11px] leading-none transition-colors hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
      :class="
        isActive ? 'border-border font-medium text-surface-foreground' : 'border-border/70 text-muted'
      "
      data-testid="search-chip-type"
      :aria-expanded="open"
      aria-haspopup="dialog"
      aria-label="Content type"
      @click="toggle"
    >
      {{ chipLabel }}
    </button>
    <div
      v-show="open"
      ref="panelRef"
      role="dialog"
      aria-label="Content type"
      data-testid="search-popover-type"
      class="absolute left-0 top-full z-[40] mt-1 w-48 rounded-sm border border-border bg-elevated p-1 shadow-md"
    >
      <button
        v-for="opt in OPTIONS"
        :key="opt.key"
        type="button"
        role="menuitemradio"
        :aria-checked="opt.key === current.key"
        class="flex w-full items-center gap-2 rounded px-2 py-1 text-left text-[11px] hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
        :class="opt.key === current.key ? 'font-medium text-surface-foreground' : 'text-muted'"
        :data-testid="`search-type-option-${opt.key}`"
        @click="choose(opt.key)"
      >
        <span aria-hidden="true" class="w-3">{{ opt.key === current.key ? '•' : '' }}</span>
        {{ opt.label }}
      </button>
    </div>
  </div>
</template>
