<script setup lang="ts">
/**
 * Graph feed-filter chip (#658). Wraps the shared
 * ``CorpusFeedFilterPanel`` inside a popover anchored to a chip button
 * in the graph's filter bar. State lives on
 * ``useGraphFilterStore.state.graphFeedFilterId``.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useShellStore } from '../../../stores/shell'
import { useGraphFilterStore } from '../../../stores/graphFilters'
import { useFilterChipPopover } from '../../../composables/useFilterChipPopover'
import {
  fetchCorpusFeeds,
  type CorpusFeedItem,
} from '../../../api/corpusLibraryApi'
import { feedRowVisibleLabel } from '../../../utils/corpusFeedRowDisplay'
import CorpusFeedFilterPanel from '../../shared/CorpusFeedFilterPanel.vue'

const shell = useShellStore()
const gf = useGraphFilterStore()
const anchorRef = ref<HTMLButtonElement | null>(null)
const panelRef = ref<HTMLDivElement | null>(null)
const { open, toggle } = useFilterChipPopover(anchorRef, panelRef)

const feeds = ref<CorpusFeedItem[]>([])
const loading = ref(false)
const error = ref<string | null>(null)

async function loadFeeds(): Promise<void> {
  const path = shell.corpusPath.trim()
  if (!path) {
    feeds.value = []
    return
  }
  loading.value = true
  error.value = null
  try {
    const resp = await fetchCorpusFeeds(path)
    feeds.value = resp.feeds
  }
  catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
    feeds.value = []
  }
  finally {
    loading.value = false
  }
}

onMounted(() => {
  void loadFeeds()
})

watch(() => shell.corpusPath, () => {
  void loadFeeds()
})

const selectedId = computed(() => gf.state?.graphFeedFilterId ?? null)
const selectedFeed = computed<CorpusFeedItem | null>(() => {
  const id = selectedId.value
  if (id === null) return null
  return feeds.value.find((f) => f.feed_id === id) ?? null
})

const chipLabel = computed(() => {
  const f = selectedFeed.value
  if (!f) return 'Feed ▾'
  return `Feed: ${feedRowVisibleLabel(f)} ▾`
})

const isActive = computed(() => selectedId.value !== null)

function onSelect(v: string | null): void {
  if (v === null) {
    gf.clearFeedFilter()
  }
  else {
    gf.setFeedFilter(v)
  }
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
      data-testid="graph-chip-feed"
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
      data-testid="graph-popover-feed"
      class="absolute left-0 top-full z-[40] mt-1 w-72 rounded-sm border border-border bg-elevated p-2 shadow-md"
    >
      <CorpusFeedFilterPanel
        :model-value="selectedId"
        :feeds="feeds"
        :corpus-path="shell.corpusPath"
        :loading="loading"
        :error="error"
        data-testid="graph-feed-filter-panel"
        search-testid="graph-feed-filter-search"
        list-testid="graph-feed-filter-list"
        @update:model-value="onSelect"
      />
    </div>
  </div>
</template>
