<script setup lang="ts">
/**
 * ShowsView (UXS-015 / RFC-104) — shows-first grid for the operator Library tab.
 *
 * Lists the corpus's shows (``GET /api/corpus/feeds``) as cover cards, sorted by
 * episode count. Selecting a card emits ``select`` with the feed; ``ShowsBrowse``
 * swaps the grid for ``ShowDetailView``. Pure read over an existing endpoint —
 * no new server surface. Cover art reuses the shared ``PodcastCover``.
 */
import { onMounted, ref, watch } from 'vue'
import { useShellStore } from '../../stores/shell'
import { fetchCorpusFeeds, type CorpusFeedItem } from '../../api/corpusLibraryApi'
import PodcastCover from '../shared/PodcastCover.vue'

const emit = defineEmits<{ (e: 'select', feed: CorpusFeedItem): void }>()

const shell = useShellStore()
const feeds = ref<CorpusFeedItem[]>([])
const loading = ref(false)
const error = ref<string | null>(null)

function showLabel(f: CorpusFeedItem): string {
  return f.display_title?.trim() || f.feed_id
}

async function load(): Promise<void> {
  const path = shell.corpusPath.trim()
  if (!path) {
    feeds.value = []
    return
  }
  loading.value = true
  error.value = null
  try {
    const body = await fetchCorpusFeeds(path)
    feeds.value = [...body.feeds].sort(
      (a, b) => b.episode_count - a.episode_count || showLabel(a).localeCompare(showLabel(b)),
    )
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load shows'
    feeds.value = []
  } finally {
    loading.value = false
  }
}

onMounted(load)
watch(
  () => shell.corpusPath,
  () => void load(),
)

defineExpose({ reload: load })
</script>

<template>
  <div data-testid="shows-grid" class="h-full min-h-0 overflow-y-auto p-3">
    <p v-if="loading && feeds.length === 0" class="p-2 text-xs text-muted">Loading shows…</p>
    <p
      v-else-if="error"
      class="p-2 text-xs text-danger"
      data-testid="shows-grid-error"
    >
      {{ error }}
    </p>
    <p
      v-else-if="feeds.length === 0"
      class="p-6 text-center text-sm text-muted"
      data-testid="shows-grid-empty"
    >
      No shows in this corpus.
    </p>
    <ul v-else class="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
      <li v-for="f in feeds" :key="f.feed_id">
        <div
          role="button"
          tabindex="0"
          :data-testid="`shows-card-${f.feed_id}`"
          data-shows-card
          class="group flex w-full flex-col gap-1.5 rounded-lg border border-default bg-overlay p-2 text-left outline-none hover:bg-overlay-2 focus-visible:ring-2 focus-visible:ring-primary"
          :aria-label="`${showLabel(f)}, ${f.episode_count} episodes`"
          @click="emit('select', f)"
          @keydown.enter.prevent="emit('select', f)"
          @keydown.space.prevent="emit('select', f)"
        >
          <PodcastCover
            :corpus-path="shell.corpusPath"
            :feed-image-local-relpath="f.image_local_relpath"
            :feed-image-url="f.image_url"
            :alt="`Cover for ${showLabel(f)}`"
            size-class="aspect-square w-full rounded-lg"
          />
          <p class="line-clamp-2 text-sm font-semibold leading-snug text-surface-foreground">
            {{ showLabel(f) }}
          </p>
          <p class="text-[11px] text-muted">
            {{ f.episode_count }} {{ f.episode_count === 1 ? 'episode' : 'episodes' }}
          </p>
          <p v-if="f.description" class="line-clamp-2 text-[11px] text-muted">
            {{ f.description }}
          </p>
        </div>
      </li>
    </ul>
  </div>
</template>
