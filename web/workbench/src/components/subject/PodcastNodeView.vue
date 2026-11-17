<script setup lang="ts">
/**
 * FB14 — Podcast/Show node view. Renders a show's basics (description, episode
 * count) + a paged, date-sorted episode list, mirroring the player. Reached by
 * focusing a ``podcast:<slug>`` node (real graph Podcast node when in-slice, or
 * synthetic via the id when not). The show id is matched to a feed by title
 * (punctuation-insensitive), and the cover is emitted up so NodeDetail shows it
 * in the rail header instead of the generic letter avatar.
 */
import { computed, ref, watch } from 'vue'
import {
  fetchCorpusFeeds,
  fetchCorpusEpisodes,
  type CorpusFeedItem,
  type CorpusEpisodeListItem,
} from '../../api/corpusLibraryApi'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import { StaleGeneration } from '../../utils/staleGeneration'
import PodcastCover from '../shared/PodcastCover.vue'

const props = withDefaults(defineProps<{ subjectIdOverride?: string }>(), {
  subjectIdOverride: '',
})
const emit = defineEmits<{
  cover: [{ imageUrl: string | null; imageLocalRelpath: string | null } | null]
}>()

const shell = useShellStore()
const subject = useSubjectStore()

const podcastId = computed(
  () => props.subjectIdOverride?.trim() || subject.graphNodeCyId?.trim() || '',
)

/** ``g:podcast:planet-money`` → ``planet money`` (display fallback). */
function showSlugLabel(id: string): string {
  return id
    .replace(/^(?:g:|k:|kg:)+/, '')
    .replace(/^podcast:/, '')
    .replace(/[-_]+/g, ' ')
    .trim()
}
/** Alphanumeric-only key so slug `oshaughnessy` matches title `O'Shaughnessy`
 *  (P1: the apostrophe-vs-space mismatch left some shows blank). */
function matchKey(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]/g, '')
}

const feed = ref<CorpusFeedItem | null>(null)
const episodes = ref<CorpusEpisodeListItem[]>([])
const loading = ref(false)
const error = ref<string | null>(null)
const gate = new StaleGeneration()

async function load(id: string): Promise<void> {
  const root = shell.corpusPath?.trim()
  const want = matchKey(showSlugLabel(id))
  if (!id || !root || !shell.healthStatus || !want) {
    feed.value = null
    episodes.value = []
    error.value = null
    emit('cover', null)
    return
  }
  const seq = gate.bump()
  loading.value = true
  error.value = null
  feed.value = null
  episodes.value = []
  emit('cover', null)
  try {
    const feeds = await fetchCorpusFeeds(root)
    if (gate.isStale(seq)) return
    const f = (feeds.feeds ?? []).find((x) => matchKey(x.display_title ?? '') === want) ?? null
    feed.value = f
    emit('cover', f ? { imageUrl: f.image_url ?? null, imageLocalRelpath: f.image_local_relpath ?? null } : null)
    if (f?.feed_id) {
      const eps = await fetchCorpusEpisodes(root, { feedId: f.feed_id, limit: 500 })
      if (gate.isStale(seq)) return
      episodes.value = eps.items ?? []
    }
  } catch (e) {
    if (gate.isStale(seq)) return
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    if (gate.isCurrent(seq)) loading.value = false
  }
}
watch(podcastId, (id) => void load(id ?? ''), { immediate: true })

// P3 — date sort (default newest) + pagination (10/page), like the topic timeline.
const EPISODES_PAGE_SIZE = 10
const sortOrder = ref<'asc' | 'desc'>('desc')
const page = ref(1)
watch([podcastId, sortOrder], () => {
  page.value = 1
})
const sortedEpisodes = computed(() => {
  const rows = episodes.value.slice()
  rows.sort((a, b) => {
    const av = a.publish_date ?? ''
    const bv = b.publish_date ?? ''
    if (av === bv) return 0
    const cmp = av < bv ? -1 : 1
    return sortOrder.value === 'asc' ? cmp : -cmp
  })
  return rows
})
const totalPages = computed(() => Math.max(1, Math.ceil(sortedEpisodes.value.length / EPISODES_PAGE_SIZE)))
const pagedEpisodes = computed(() => {
  const start = (page.value - 1) * EPISODES_PAGE_SIZE
  return sortedEpisodes.value.slice(start, start + EPISODES_PAGE_SIZE)
})

function openEpisode(ep: CorpusEpisodeListItem): void {
  if (!ep.metadata_relative_path) return
  subject.focusEpisode(ep.metadata_relative_path, {
    uiTitle: ep.episode_title,
    episodeId: ep.episode_id ?? undefined,
  })
}
</script>

<template>
  <div class="min-w-0" data-testid="podcast-node-view">
    <p v-if="feed?.episode_count" class="text-[10px] text-muted" data-testid="podcast-node-count">
      {{ feed.episode_count }} episode{{ feed.episode_count === 1 ? '' : 's' }}
    </p>
    <!-- P2 — summary up: the description leads the panel (cover + title live in
         the rail header now). -->
    <p
      v-if="feed?.description"
      class="mt-1 line-clamp-5 break-words text-[11px] leading-snug text-muted"
      data-testid="podcast-node-description"
    >
      {{ feed.description }}
    </p>

    <div class="mb-1 mt-3 flex items-center justify-between gap-2">
      <h3 class="text-[10px] font-semibold uppercase tracking-wider text-muted">Episodes</h3>
      <div v-if="episodes.length" class="flex items-center gap-1" data-testid="podcast-node-sort">
        <span class="text-[10px] text-muted">Date</span>
        <button
          type="button"
          class="rounded border px-1.5 py-0.5 text-[10px]"
          :class="sortOrder === 'asc' ? 'border-gi/60 bg-gi/15' : 'border-border text-muted'"
          @click="sortOrder = 'asc'"
        >
          Oldest
        </button>
        <button
          type="button"
          class="rounded border px-1.5 py-0.5 text-[10px]"
          :class="sortOrder === 'desc' ? 'border-gi/60 bg-gi/15' : 'border-border text-muted'"
          @click="sortOrder = 'desc'"
        >
          Newest
        </button>
      </div>
    </div>

    <p v-if="loading" class="text-[11px] text-muted" data-testid="podcast-node-loading">Loading…</p>
    <p v-else-if="error" class="text-[11px] text-warning">{{ error }}</p>
    <ul v-else-if="episodes.length" class="space-y-2" data-testid="podcast-node-episodes">
      <li v-for="ep in pagedEpisodes" :key="ep.metadata_relative_path" class="min-w-0">
        <button
          type="button"
          class="flex w-full min-w-0 items-start gap-2 rounded text-left hover:bg-overlay/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          data-testid="podcast-node-episode-open"
          :title="`Open ${ep.episode_title}`"
          @click="openEpisode(ep)"
        >
          <PodcastCover
            :corpus-path="shell.corpusPath"
            :episode-image-url="ep.episode_image_url ?? null"
            :feed-image-url="ep.feed_image_url ?? null"
            :episode-image-local-relpath="ep.episode_image_local_relpath ?? null"
            :feed-image-local-relpath="ep.feed_image_local_relpath ?? null"
            :alt="`Cover for ${ep.episode_title}`"
            size-class="h-9 w-9"
          />
          <div class="min-w-0 flex-1">
            <p class="text-[10px] font-medium text-gi/90">{{ ep.publish_date || 'Date unknown' }}</p>
            <p class="break-words text-[11px] font-semibold leading-snug text-surface-foreground">
              {{ ep.episode_title }}
            </p>
            <p
              v-if="ep.summary_title"
              class="break-words text-[10px] leading-snug text-muted line-clamp-2"
            >
              {{ ep.summary_title }}
            </p>
          </div>
        </button>
      </li>
    </ul>
    <p v-else class="text-[11px] text-muted" data-testid="podcast-node-empty">
      No episodes found for this show.
    </p>

    <nav
      v-if="totalPages > 1"
      class="mt-2 flex items-center justify-center gap-1"
      aria-label="Episode pages"
      data-testid="podcast-node-pager"
    >
      <button
        type="button"
        class="rounded border border-border px-1.5 py-0.5 text-[10px] text-muted hover:bg-overlay disabled:opacity-40"
        :disabled="page === 1"
        aria-label="Previous page"
        @click="page = Math.max(1, page - 1)"
      >
        ‹
      </button>
      <span class="px-1.5 text-[10px] tabular-nums text-muted">{{ page }} / {{ totalPages }}</span>
      <button
        type="button"
        class="rounded border border-border px-1.5 py-0.5 text-[10px] text-muted hover:bg-overlay disabled:opacity-40"
        :disabled="page >= totalPages"
        aria-label="Next page"
        @click="page = Math.min(totalPages, page + 1)"
      >
        ›
      </button>
    </nav>
  </div>
</template>
