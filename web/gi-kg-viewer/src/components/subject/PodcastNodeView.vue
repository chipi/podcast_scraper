<script setup lang="ts">
/**
 * FB14 — Podcast/Show node view. Renders a show's basics (cover, title,
 * description, episode count) + its episode list, mirroring the player. Reached
 * by focusing a ``podcast:<slug>`` node (real graph Podcast node when in-slice,
 * or synthetic via the id when not). Data comes from the corpus feeds + the
 * feed-scoped episode list; the show id is matched to a feed by title.
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
function norm(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim()
}

const feed = ref<CorpusFeedItem | null>(null)
const episodes = ref<CorpusEpisodeListItem[]>([])
const loading = ref(false)
const error = ref<string | null>(null)
const gate = new StaleGeneration()

async function load(id: string): Promise<void> {
  const root = shell.corpusPath?.trim()
  const want = norm(showSlugLabel(id))
  if (!id || !root || !shell.healthStatus || !want) {
    feed.value = null
    episodes.value = []
    error.value = null
    return
  }
  const seq = gate.bump()
  loading.value = true
  error.value = null
  feed.value = null
  episodes.value = []
  try {
    const feeds = await fetchCorpusFeeds(root)
    if (gate.isStale(seq)) return
    const f = (feeds.feeds ?? []).find((x) => norm(x.display_title ?? '') === want) ?? null
    feed.value = f
    if (f?.feed_id) {
      const eps = await fetchCorpusEpisodes(root, { feedId: f.feed_id, limit: 200 })
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

const title = computed(() => feed.value?.display_title || showSlugLabel(podcastId.value))

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
    <div class="flex items-start gap-2">
      <PodcastCover
        :corpus-path="shell.corpusPath"
        :feed-image-url="feed?.image_url ?? null"
        :feed-image-local-relpath="feed?.image_local_relpath ?? null"
        :alt="`Cover for ${title}`"
        size-class="h-12 w-12"
      />
      <div class="min-w-0 flex-1">
        <p class="break-words text-sm font-semibold text-surface-foreground" data-testid="podcast-node-title">
          {{ title }}
        </p>
        <p v-if="feed?.episode_count" class="text-[10px] text-muted">
          {{ feed.episode_count }} episode{{ feed.episode_count === 1 ? '' : 's' }}
        </p>
      </div>
    </div>
    <p
      v-if="feed?.description"
      class="mt-2 line-clamp-4 break-words text-[11px] leading-snug text-muted"
      data-testid="podcast-node-description"
    >
      {{ feed.description }}
    </p>

    <h3 class="mb-1 mt-3 text-[10px] font-semibold uppercase tracking-wider text-muted">
      Episodes
    </h3>
    <p v-if="loading" class="text-[11px] text-muted" data-testid="podcast-node-loading">Loading…</p>
    <p v-else-if="error" class="text-[11px] text-warning">{{ error }}</p>
    <ul v-else-if="episodes.length" class="space-y-2" data-testid="podcast-node-episodes">
      <li v-for="ep in episodes" :key="ep.metadata_relative_path" class="min-w-0">
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
  </div>
</template>
