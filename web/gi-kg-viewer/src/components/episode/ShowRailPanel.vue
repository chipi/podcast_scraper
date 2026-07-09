<script setup lang="ts">
/**
 * ShowRailPanel (UXS-015 / RFC-104) — a Show opened in the right subject rail.
 *
 * Mirrors EpisodeDetailPanel's header (compact cover + title + meta line), then a
 * show-level Signals band (top topics + key people, from `GET /corpus/feed-signals`
 * — Topic/Person nodes counted across the show's episode KGs), then the episode
 * list. Reads `subject.feedId` (set by `subject.focusShow` from the Shows grid) and
 * re-fetches feed (header) + episodes + signals. Clicking an episode calls
 * `subject.focusEpisode`, a topic/person chip calls `subject.focusTopic`/`focusPerson`
 * — each opens in this same rail and pushes the show onto the Back stack, so the
 * child rail's Back returns here.
 */
import { computed, ref, watch } from 'vue'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import {
  fetchCorpusEpisodes,
  fetchCorpusFeeds,
  fetchFeedSignals,
  type CorpusEpisodeListItem,
  type CorpusFeedItem,
  type CorpusFeedSignalsResponse,
} from '../../api/corpusLibraryApi'
import PodcastCover from '../shared/PodcastCover.vue'

const shell = useShellStore()
const subject = useSubjectStore()

const feed = ref<CorpusFeedItem | null>(null)
const episodes = ref<CorpusEpisodeListItem[]>([])
const signals = ref<CorpusFeedSignalsResponse | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)
const nextCursor = ref<string | null>(null)
const descExpanded = ref(false)

const PAGE = 50
const DESC_CLAMP = 180
const SIGNALS_TOP_K = 8

const topTopics = computed(() => signals.value?.top_topics ?? [])
const keyPeople = computed(() => signals.value?.key_people ?? [])
const hasSignals = computed(() => topTopics.value.length > 0 || keyPeople.value.length > 0)

const title = computed(
  () => feed.value?.display_title?.trim() || subject.feedUiLabel || subject.feedId || 'Show',
)
const episodeCount = computed(() => feed.value?.episode_count ?? episodes.value.length)
const description = computed(() => feed.value?.description?.trim() || '')
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

async function loadFeed(): Promise<void> {
  const path = shell.corpusPath.trim()
  const id = subject.feedId?.trim()
  if (!path || !id) {
    feed.value = null
    return
  }
  try {
    const body = await fetchCorpusFeeds(path)
    feed.value = body.feeds.find((f) => f.feed_id === id) ?? null
  } catch {
    feed.value = null
  }
}

async function loadEpisodes(reset: boolean): Promise<void> {
  const path = shell.corpusPath.trim()
  const id = subject.feedId?.trim()
  if (!path || !id) {
    episodes.value = []
    return
  }
  loading.value = true
  error.value = null
  try {
    const body = await fetchCorpusEpisodes(path, {
      feedId: id,
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

async function loadSignals(): Promise<void> {
  const path = shell.corpusPath.trim()
  const id = subject.feedId?.trim()
  if (!path || !id) {
    signals.value = null
    return
  }
  try {
    signals.value = await fetchFeedSignals(path, id, SIGNALS_TOP_K)
  } catch {
    signals.value = null
  }
}

function selectEpisode(e: CorpusEpisodeListItem): void {
  subject.focusEpisode(e.metadata_relative_path, {
    uiTitle: e.episode_title?.trim() || null,
    episodeId: e.episode_id ?? null,
  })
}

// Topic / person chips open the unified node view in this same rail; Back returns
// here (focusGraphNode pushes the show onto the history stack).
function openTopic(id: string): void {
  subject.focusTopic(id)
}
function openPerson(id: string): void {
  subject.focusPerson(id)
}

watch(
  () => subject.feedId,
  () => {
    nextCursor.value = null
    descExpanded.value = false
    signals.value = null
    void loadFeed()
    void loadEpisodes(true)
    void loadSignals()
  },
  { immediate: true },
)
</script>

<template>
  <div
    class="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden"
    data-testid="show-rail-panel"
  >
    <!-- Header — mirrors EpisodeDetailPanel: compact cover (4.5rem) left + title + meta -->
    <div class="shrink-0 border-b border-border px-2 py-2">
      <div class="flex min-w-0 items-start gap-3">
        <PodcastCover
          class="shrink-0"
          :corpus-path="shell.corpusPath"
          :feed-image-local-relpath="feed?.image_local_relpath"
          :feed-image-url="feed?.image_url"
          :alt="`Cover for ${title}`"
          size-class="h-[4.5rem] w-[4.5rem]"
        />
        <div class="min-h-0 min-w-0 flex-1">
          <h3
            class="min-w-0 select-text text-base font-semibold leading-snug text-surface-foreground"
          >
            {{ title }}
          </h3>
          <p class="mt-0.5 text-xs text-muted">
            {{ episodeCount }} {{ episodeCount === 1 ? 'episode' : 'episodes' }}
            <template v-if="feed?.rss_url">
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
            class="mt-1 text-xs leading-snug text-muted"
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

    <!-- Show-level signals (UXS-015 / RFC-104): most-covered topics + key people,
         counted across the show's episode KGs. Chips open the node view in-rail. -->
    <div
      v-if="hasSignals"
      class="shrink-0 border-b border-border px-3 py-2"
      data-testid="show-rail-signals"
    >
      <div v-if="topTopics.length" class="mb-2">
        <h4 class="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted">
          Top topics
        </h4>
        <div class="flex flex-wrap gap-1">
          <button
            v-for="tp in topTopics"
            :key="tp.topic_id"
            type="button"
            data-testid="show-rail-topic"
            class="rounded-full bg-overlay px-2 py-0.5 text-[11px] text-topic outline-none transition hover:bg-overlay-2 focus-visible:ring-2 focus-visible:ring-primary"
            @click="openTopic(tp.topic_id)"
          >
            {{ tp.label }} <span class="text-muted">· {{ tp.episode_count }}</span>
          </button>
        </div>
      </div>
      <div v-if="keyPeople.length">
        <h4 class="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted">
          Key people
        </h4>
        <div class="flex flex-wrap gap-1">
          <button
            v-for="p in keyPeople"
            :key="p.person_id"
            type="button"
            data-testid="show-rail-person"
            class="rounded-full bg-overlay px-2 py-0.5 text-[11px] text-person outline-none transition hover:bg-overlay-2 focus-visible:ring-2 focus-visible:ring-primary"
            @click="openPerson(p.person_id)"
          >
            {{ p.name }} <span class="text-muted">· {{ p.episode_count }}</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Episode list (clicking an episode opens it in this rail, with Back-to-show) -->
    <div class="min-h-0 flex-1 overflow-y-auto p-2">
      <p v-if="loading && episodes.length === 0" class="p-2 text-xs text-muted">
        Loading episodes…
      </p>
      <p v-else-if="error" class="p-2 text-xs text-danger" data-testid="show-rail-error">
        {{ error }}
      </p>
      <p v-else-if="episodes.length === 0" class="p-4 text-xs text-muted" data-testid="show-rail-empty">
        No episodes.
      </p>
      <ul v-else class="space-y-0.5 text-sm">
        <li v-for="(e, i) in episodes" :key="e.metadata_relative_path">
          <div
            role="button"
            tabindex="0"
            data-library-episode-row
            :data-testid="`show-rail-episode-${i}`"
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
              <div
                v-if="episodeTopics(e).length || e.has_gi || e.has_kg"
                class="mt-0.5 flex flex-wrap items-center gap-1"
              >
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
        data-testid="show-rail-load-more"
        class="mt-2 w-full rounded border border-default py-1.5 text-xs text-muted outline-none hover:bg-overlay-2 focus-visible:ring-2 focus-visible:ring-primary"
        :disabled="loading"
        @click="loadEpisodes(false)"
      >
        {{ loading ? 'Loading…' : 'Load more' }}
      </button>
    </div>
  </div>
</template>
