<script setup lang="ts">
/**
 * Library feed-filter chip (#669). Wraps the shared
 * ``CorpusFeedFilterPanel`` in a popover anchored to the chip button.
 * State is owned by ``LibraryView`` (kept as a local ref there to
 * preserve existing reload semantics) and driven via v-model.
 */
import { computed, ref } from 'vue'
import type { CorpusFeedItem } from '../../../api/corpusLibraryApi'
import { useFilterChipPopover } from '../../../composables/useFilterChipPopover'
import { feedRowVisibleLabel } from '../../../utils/corpusFeedRowDisplay'
import CorpusFeedFilterPanel from '../../shared/CorpusFeedFilterPanel.vue'

const props = withDefaults(
  defineProps<{
    /** Selected feed_id; ``null`` = "All feeds". */
    modelValue: string | null
    /** Feed list from ``GET /api/corpus/feeds``. */
    feeds: ReadonlyArray<CorpusFeedItem>
    corpusPath?: string | null
    loading?: boolean
    error?: string | null
  }>(),
  { corpusPath: null, loading: false, error: null },
)

const emit = defineEmits<{
  (e: 'update:modelValue', v: string | null): void
}>()

const anchorRef = ref<HTMLButtonElement | null>(null)
const panelRef = ref<HTMLDivElement | null>(null)
const { open, toggle } = useFilterChipPopover(anchorRef, panelRef)

const selectedFeed = computed<CorpusFeedItem | null>(() => {
  const id = props.modelValue
  if (id === null) return null
  return props.feeds.find((f) => f.feed_id === id) ?? null
})

const isActive = computed(() => props.modelValue !== null)

const chipLabel = computed(() => {
  const f = selectedFeed.value
  if (!f) return 'Feed ▾'
  return `Feed: ${feedRowVisibleLabel(f)} ▾`
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
      data-testid="library-chip-feed"
      :aria-expanded="open"
      aria-haspopup="dialog"
      aria-label="Feed filter"
      @click="toggle"
    >
      {{ chipLabel }}
    </button>
    <div
      v-show="open"
      ref="panelRef"
      role="dialog"
      aria-label="Feed filter"
      data-testid="library-popover-feed"
      class="absolute left-0 top-full z-[40] mt-1 w-72 rounded-sm border border-border bg-elevated p-2 shadow-md"
    >
      <CorpusFeedFilterPanel
        :model-value="props.modelValue"
        :feeds="props.feeds"
        :corpus-path="props.corpusPath"
        :loading="props.loading"
        :error="props.error"
        data-testid="library-feed-filter-panel"
        search-testid="library-feed-filter-search"
        list-testid="library-feed-filter-list"
        @update:model-value="(v) => emit('update:modelValue', v)"
      />
    </div>
  </div>
</template>
