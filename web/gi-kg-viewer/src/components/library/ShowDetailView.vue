<script setup lang="ts">
/**
 * ShowDetailView (UXS-015 / RFC-104) — one show's header + episode list.
 *
 * Header derives from the ``feed`` prop (title, count, RSS, clamped description +
 * large cover). Episodes load from ``GET /api/corpus/episodes?feed_id=…`` newest-first,
 * paginated via the existing cursor. An episode row opens the episode in the subject
 * rail via ``subject.focusEpisode`` — the same path a flat-Library row uses (no new
 * cross-link policy / load-source band-aid; see E2E_SURFACE_MAP automation contract).
 */
import { computed, ref, watch } from 'vue'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import {
  fetchCorpusEpisodes,
  type CorpusEpisodeListItem,
  type CorpusFeedItem,
} from '../../api/corpusLibraryApi'
import PodcastCover from '../shared/PodcastCover.vue'

const props = defineProps<{ feed: CorpusFeedItem }>()
const emit = defineEmits<{ (e: 'back'): void }>()

const shell = useShellStore()
const subject = useSubjectStore()

const episodes = ref<CorpusEpisodeListItem[]>([])
const loading = ref(false)
const error = ref<string | null>(null)
const nextCursor = ref<string | null>(null)
const descExpanded = ref(false)

const PAGE = 50
const DESC_CLAMP = 180

const title = computed(() => props.feed.display_title?.trim() || props.feed.feed_id)
const description = computed(() => props.feed.description?.trim() || '')
const descIsLong = computed(() => description.value.length > DESC_CLAMP)

function fmtDate(iso: string | null | undefined): string {
  const s = (iso ?? '').trim()
  return s ? s.slice(0, 10) : ''
}

function episodeSummary(e: CorpusEpisodeListItem): string {
  return (e.summary_preview?.trim() || e.summary_title?.trim() || '').trim()
}

function episodeTopics(e: CorpusEpisodeListItem): string[] {
  if (e.topics?.length) return e.topics.slice(0, 4)
  return (e.summary_bullets_preview ?? []).slice(0, 4)
}

async function load(reset: boolean): Promise<void> {
  const path = shell.corpusPath.trim()
  if (!path) {
    episodes.value = []
    return
  }
  loading.value = true
  error.value = null
  try {
    const body = await fetchCorpusEpisodes(path, {
      feedId: props.feed.feed_id,
      limit: PAGE,
      cursor: reset ? null : nextCursor.value,
    })
    episodes.value = reset ? body.items : [...episodes.value, ...body.items]
    nextCursor.value = body.next_cursor
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load episodes'
    if (reset) episodes.value = []
  } finally {
    loading.value = false
  }
}

function selectEpisode(e: CorpusEpisodeListItem): void {
  subject.focusEpisode(e.metadata_relative_path, {
    uiTitle: e.episode_title?.trim() || null,
    episodeId: e.episode_id ?? null,
  })
}

watch(
  () => props.feed.feed_id,
  () => {
    nextCursor.value = null
    descExpanded.value = false
    void load(true)
  },
  { immediate: true },
)
</script>

<template>
  <div data-testid="show-detail" class="flex h-full min-h-0 flex-col">
    <div class="shrink-0 border-b border-default p-3">
      <button
        type="button"
        data-testid="show-detail-back"
        class="mb-2 text-xs text-muted outline-none hover:text-surface-foreground focus-visible:ring-2 focus-visible:ring-primary"
        @click="emit('back')"
      >
        ‹ Shows
      </button>
      <div class="flex gap-3">
        <PodcastCover
          :corpus-path="shell.corpusPath"
          :feed-image-local-relpath="feed.image_local_relpath"
          :feed-image-url="feed.image_url"
          :alt="`Cover for ${title}`"
          size-class="h-20 w-20 rounded-xl sm:h-24 sm:w-24"
        />
        <div class="min-w-0 flex-1">
          <h1 class="text-xl font-extrabold leading-tight tracking-tight text-surface-foreground">
            {{ title }}
          </h1>
          <p class="mt-0.5 text-xs text-muted">
            {{ feed.episode_count }} {{ feed.episode_count === 1 ? 'episode' : 'episodes' }}
            <template v-if="feed.rss_url">
              ·
              <a
                :href="feed.rss_url"
                target="_blank"
                rel="noopener"
                class="text-primary hover:underline"
                >RSS</a
              >
            </template>
          </p>
          <p
            v-if="description"
            class="mt-1.5 text-sm leading-relaxed text-muted"
            :class="descExpanded ? '' : 'line-clamp-3'"
          >
            {{ description }}
          </p>
          <button
            v-if="descIsLong"
            type="button"
            class="mt-0.5 text-xs text-primary hover:underline"
            @click="descExpanded = !descExpanded"
          >
            {{ descExpanded ? 'Show less' : 'Show more' }}
          </button>
        </div>
      </div>
    </div>

    <div class="min-h-0 flex-1 overflow-y-auto p-2">
      <p v-if="loading && episodes.length === 0" class="p-2 text-xs text-muted">Loading episodes…</p>
      <p v-else-if="error" class="p-2 text-xs text-danger" data-testid="show-detail-error">
        {{ error }}
      </p>
      <p
        v-else-if="episodes.length === 0"
        class="p-4 text-xs text-muted"
        data-testid="show-detail-empty"
      >
        No episodes.
      </p>
      <ul v-else class="space-y-0.5 text-sm">
        <li v-for="(e, i) in episodes" :key="e.metadata_relative_path">
          <div
            role="button"
            tabindex="0"
            data-library-episode-row
            :data-testid="`show-detail-episode-${i}`"
            class="group flex w-full gap-2 rounded px-2 py-1.5 text-left outline-none hover:bg-overlay/35 focus-visible:ring-2 focus-visible:ring-primary"
            :aria-label="`${e.episode_title}, ${title}`"
            @click="selectEpisode(e)"
            @keydown.enter.prevent="selectEpisode(e)"
            @keydown.space.prevent="selectEpisode(e)"
          >
            <PodcastCover
              :corpus-path="shell.corpusPath"
              :episode-image-local-relpath="e.episode_image_local_relpath"
              :feed-image-local-relpath="e.feed_image_local_relpath"
              :episode-image-url="e.episode_image_url"
              :feed-image-url="e.feed_image_url"
              :alt="`Cover for ${e.episode_title}`"
              size-class="h-9 w-9"
            />
            <div class="min-w-0 flex-1">
              <div class="flex items-baseline justify-between gap-2">
                <span class="min-w-0 flex-1 truncate font-medium text-surface-foreground">{{
                  e.episode_title
                }}</span>
                <span v-if="fmtDate(e.publish_date)" class="shrink-0 text-[11px] text-muted">{{
                  fmtDate(e.publish_date)
                }}</span>
              </div>
              <p v-if="episodeSummary(e)" class="line-clamp-1 text-[11px] text-muted">
                {{ episodeSummary(e) }}
              </p>
              <div v-if="episodeTopics(e).length || e.has_gi || e.has_kg" class="mt-0.5 flex flex-wrap items-center gap-1">
                <span
                  v-for="tp in episodeTopics(e)"
                  :key="tp"
                  class="rounded bg-overlay px-1.5 py-0.5 text-[10px] text-muted"
                  >{{ tp }}</span
                >
                <span
                  v-if="e.has_gi"
                  class="rounded bg-emerald-700/25 px-1.5 py-0.5 text-[10px] text-emerald-300"
                  title="Has grounded insights"
                  >GI</span
                >
                <span
                  v-if="e.has_kg"
                  class="rounded bg-sky-700/25 px-1.5 py-0.5 text-[10px] text-sky-300"
                  title="Has knowledge graph"
                  >KG</span
                >
              </div>
            </div>
          </div>
        </li>
      </ul>

      <button
        v-if="nextCursor"
        type="button"
        data-testid="show-detail-load-more"
        class="mt-2 w-full rounded border border-default py-1.5 text-xs text-muted outline-none hover:bg-overlay-2 focus-visible:ring-2 focus-visible:ring-primary"
        :disabled="loading"
        @click="load(false)"
      >
        {{ loading ? 'Loading…' : 'Load more' }}
      </button>
    </div>
  </div>
</template>
