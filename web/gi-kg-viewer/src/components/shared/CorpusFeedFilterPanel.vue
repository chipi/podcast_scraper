<script setup lang="ts">
/**
 * Shared feed-filter panel used by Library (#669), Graph (#658),
 * and Search (#671) chip popovers.
 *
 * Renders an optional search input (above ``CORPUS_FEED_FILTER_SEARCH_THRESHOLD``
 * feeds), an "All feeds" radio at top, and a list of per-feed rows with
 * ``PodcastCover`` + display title + episode count. Single-select; emits the
 * selected ``feed_id`` (or ``null`` for "All feeds") via ``update:modelValue``.
 *
 * Visual contract matches the legacy LibraryView feed picker; the difference
 * is that the panel renders inside a popover anchored to a chip (no fixed
 * max-height — the popover panel handles its own overflow).
 */
import { computed, ref } from 'vue'
import type { CorpusFeedItem } from '../../api/corpusLibraryApi'
import {
  CORPUS_FEED_FILTER_SEARCH_THRESHOLD,
  feedRowAccessibleName,
  feedRowTitleAttr,
  feedRowVisibleLabel,
  filterFeedsByQuery,
} from '../../utils/corpusFeedRowDisplay'
import PodcastCover from './PodcastCover.vue'

const props = withDefaults(
  defineProps<{
    /** Selected feed_id; ``null`` = "All feeds". */
    modelValue: string | null
    /** Feed list from ``GET /api/corpus/feeds``. */
    feeds: ReadonlyArray<CorpusFeedItem>
    /** Corpus root path (forwarded to ``PodcastCover``). */
    corpusPath?: string | null
    /** Whether the loading state should be visible (defaults to false). */
    loading?: boolean
    /** Optional error string to render above the list. */
    error?: string | null
    /** Show the "Clear" affordance when a feed is selected. Default true. */
    showClear?: boolean
    /** Force the search input regardless of feed count (testing aid). */
    forceSearch?: boolean
    /** Custom data-testid on the wrapper for E2E targeting. */
    dataTestid?: string
    /** Custom data-testid on the search input. */
    searchTestid?: string
    /** Custom data-testid on the scroll container. */
    listTestid?: string
  }>(),
  {
    corpusPath: null,
    loading: false,
    error: null,
    showClear: true,
    forceSearch: false,
    dataTestid: 'corpus-feed-filter-panel',
    searchTestid: 'corpus-feed-filter-search',
    listTestid: 'corpus-feed-filter-list',
  },
)

const emit = defineEmits<{
  (e: 'update:modelValue', v: string | null): void
}>()

const search = ref('')

const showSearchInput = computed(
  () => props.forceSearch || props.feeds.length > CORPUS_FEED_FILTER_SEARCH_THRESHOLD,
)

const filteredFeeds = computed(() => filterFeedsByQuery(props.feeds, search.value))

const isAllSelected = computed(() => props.modelValue === null)

function isSelected(f: CorpusFeedItem): boolean {
  return props.modelValue !== null && props.modelValue === f.feed_id
}

function selectAll(): void {
  emit('update:modelValue', null)
}

function selectFeed(f: CorpusFeedItem): void {
  emit('update:modelValue', f.feed_id)
}
</script>

<template>
  <div :data-testid="props.dataTestid" class="flex flex-col gap-1.5">
    <input
      v-if="showSearchInput"
      v-model="search"
      type="search"
      class="w-full rounded border border-border bg-elevated px-2 py-1 text-xs text-surface-foreground"
      placeholder="Filter feeds…"
      aria-label="Filter feeds by display title"
      :data-testid="props.searchTestid"
    >
    <p v-if="props.loading" class="px-1 text-xs text-muted">
      Loading…
    </p>
    <p v-else-if="props.error" class="px-1 text-xs text-danger">
      {{ props.error }}
    </p>
    <div
      v-else
      role="region"
      aria-label="Feeds"
      class="min-w-0 max-w-full overflow-x-hidden overflow-y-auto overscroll-y-contain max-h-[18rem] [scrollbar-width:thin] [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-border"
      :data-testid="props.listTestid"
    >
      <ul class="space-y-0.5 text-sm">
        <li>
          <button
            type="button"
            class="flex w-full min-w-0 items-center gap-2 rounded px-2 py-1 text-left hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            :class="
              isAllSelected ? 'bg-overlay text-surface-foreground font-medium' : 'text-muted'
            "
            :aria-pressed="isAllSelected"
            data-testid="corpus-feed-filter-all"
            @click="selectAll"
          >
            <span
              class="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded bg-overlay text-[10px] font-medium uppercase tracking-wide text-muted"
              aria-hidden="true"
            >
              All
            </span>
            <span class="min-w-0 flex-1 truncate">All feeds</span>
          </button>
        </li>
        <li v-for="f in filteredFeeds" :key="f.feed_id || '__empty__'">
          <button
            type="button"
            class="flex w-full min-w-0 items-center gap-2 rounded px-2 py-1 text-left hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            :class="
              isSelected(f) ? 'bg-overlay text-surface-foreground font-medium' : 'text-muted'
            "
            :title="feedRowTitleAttr(f)"
            :aria-label="feedRowAccessibleName(f)"
            :aria-pressed="isSelected(f)"
            @click="selectFeed(f)"
          >
            <PodcastCover
              :corpus-path="props.corpusPath"
              :feed-image-local-relpath="f.image_local_relpath"
              :feed-image-url="f.image_url"
              :alt="`Cover for ${feedRowVisibleLabel(f)}`"
              size-class="h-8 w-8"
            />
            <span class="min-w-0 flex-1 truncate text-sm">{{
              feedRowVisibleLabel(f)
            }}</span>
            <span class="shrink-0 text-[10px] text-muted">({{ f.episode_count }})</span>
          </button>
        </li>
      </ul>
    </div>
    <button
      v-if="props.showClear && !isAllSelected"
      type="button"
      class="self-end rounded px-2 py-0.5 text-[10px] text-primary underline hover:bg-overlay focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
      data-testid="corpus-feed-filter-clear"
      @click="selectAll"
    >
      Clear
    </button>
  </div>
</template>
