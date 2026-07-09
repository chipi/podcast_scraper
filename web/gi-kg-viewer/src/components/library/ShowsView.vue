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
        <!--
          Editorial cover-forward card (operator feedback 2026-07-09): no border frame —
          a full-bleed square cover with the title over a bottom gradient, meta + a roomier
          summary below. rounded-xl + ring/shadow on hover; the fixed aspect-square keeps every
          card identical height regardless of art or summary length.
        -->
        <div
          role="button"
          tabindex="0"
          :data-testid="`shows-card-${f.feed_id}`"
          data-shows-card
          class="group flex w-full flex-col overflow-hidden rounded-xl bg-overlay/40 text-left shadow-sm outline-none ring-1 ring-border/60 transition hover:shadow-md hover:ring-primary/60 focus-visible:ring-2 focus-visible:ring-primary"
          :aria-label="`${showLabel(f)}, ${f.episode_count} episodes`"
          @click="emit('select', f)"
          @keydown.enter.prevent="emit('select', f)"
          @keydown.space.prevent="emit('select', f)"
        >
          <div class="relative">
            <PodcastCover
              frameless
              :corpus-path="shell.corpusPath"
              :feed-image-local-relpath="f.image_local_relpath"
              :feed-image-url="f.image_url"
              :alt="`Cover for ${showLabel(f)}`"
              size-class="aspect-square w-full"
            />
            <div
              class="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/85 via-black/45 to-transparent px-2.5 pb-2 pt-10"
            >
              <p class="line-clamp-2 text-sm font-semibold leading-snug text-white drop-shadow-sm">
                {{ showLabel(f) }}
              </p>
            </div>
          </div>
          <div class="flex flex-col gap-1 px-2.5 pb-2.5 pt-2">
            <p class="text-[11px] font-medium text-muted">
              {{ f.episode_count }} {{ f.episode_count === 1 ? 'episode' : 'episodes' }}
            </p>
            <p v-if="f.description" class="line-clamp-4 text-xs leading-snug text-muted">
              {{ f.description }}
            </p>
          </div>
        </div>
      </li>
    </ul>
  </div>
</template>
