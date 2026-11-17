<script setup lang="ts">
/**
 * Library unified filter chip bar (#669). Replaces the
 * ``CollapsibleSection`` + grid form in ``LibraryView``. Layout:
 *
 *   [Feed] [Date] [Clustered]                        [× reset]
 *
 * Date chip is the shared ``DateChip`` (from #670) bound to
 * ``corpusLens.sinceYmd`` so Library + Digest + Search all reuse the
 * same component. Feed chip uses the shared ``CorpusFeedFilterPanel``
 * via ``LibraryFeedChip``.
 */
import { computed } from 'vue'
import type { CorpusFeedItem } from '../../api/corpusLibraryApi'
import DateChip from '../shared/DateChip.vue'
import LibraryFeedChip from './chips/LibraryFeedChip.vue'
import LibraryClusteredChip from './chips/LibraryClusteredChip.vue'

const props = withDefaults(
  defineProps<{
    feedFilterId: string | null
    sinceYmd: string
    topicClusterOnly: boolean
    feeds: ReadonlyArray<CorpusFeedItem>
    corpusPath?: string | null
    feedsLoading?: boolean
    feedsError?: string | null
  }>(),
  { corpusPath: null, feedsLoading: false, feedsError: null },
)

const emit = defineEmits<{
  (e: 'update:feedFilterId', v: string | null): void
  (e: 'update:sinceYmd', v: string): void
  (e: 'update:topicClusterOnly', v: boolean): void
  (e: 'reset'): void
}>()

const isAnyActive = computed(
  () =>
    props.feedFilterId !== null
    || props.sinceYmd.trim() !== ''
    || props.topicClusterOnly,
)
</script>

<template>
  <div
    class="flex min-h-7 flex-wrap items-center gap-x-1.5 gap-y-1 px-2 py-1 text-surface-foreground"
    data-testid="library-filter-bar"
  >
    <LibraryFeedChip
      :model-value="props.feedFilterId"
      :feeds="props.feeds"
      :corpus-path="props.corpusPath"
      :loading="props.feedsLoading"
      :error="props.feedsError"
      @update:model-value="(v) => emit('update:feedFilterId', v)"
    />
    <DateChip
      :model-value="props.sinceYmd"
      chip-testid="library-chip-date"
      popover-testid="library-popover-date"
      @update:model-value="(v) => emit('update:sinceYmd', v)"
    />
    <LibraryClusteredChip
      :model-value="props.topicClusterOnly"
      @update:model-value="(v) => emit('update:topicClusterOnly', v)"
    />
    <button
      v-if="isAnyActive"
      type="button"
      class="ml-auto inline-flex h-6 items-center rounded border border-border px-2 text-[11px] leading-none text-primary hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
      data-testid="library-chip-reset"
      aria-label="Reset all library filters"
      @click="emit('reset')"
    >
      × reset
    </button>
  </div>
</template>
