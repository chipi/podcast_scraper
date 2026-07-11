<script setup lang="ts">
/**
 * ShowRailPanel (UXS-015 / RFC-104) — a Show opened in the right subject rail.
 *
 * Header mirrors EpisodeDetailPanel (compact cover + title + meta), with a roomier
 * summary and an "open in graph" action that loads the show's episode KGs onto the
 * graph. Below the header a **Signals** band surfaces show-level aggregates from
 * `GET /corpus/feed-signals` — top topics, key people, recurring guests, dominant
 * themes, trending topics and a pooled grounding score. The episode list (sortable
 * newest/oldest) shows each episode's full summary + digest-parity topic pills
 * (cluster-coloured), fetched with `with_cil_topics`. Chips open the node view in
 * this same rail via focusTopic/focusPerson; each pushes the show onto the Back stack.
 */
import { computed, ref, watch } from 'vue'
import { useArtifactsStore } from '../../stores/artifacts'
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
import CilTopicPillsRow from '../shared/CilTopicPillsRow.vue'
import PodcastCover from '../shared/PodcastCover.vue'

const emit = defineEmits<{
  (e: 'switch-main-tab', tab: 'digest' | 'library' | 'graph' | 'dashboard'): void
}>()

const shell = useShellStore()
const subject = useSubjectStore()
const artifacts = useArtifactsStore()

const feed = ref<CorpusFeedItem | null>(null)
const episodes = ref<CorpusEpisodeListItem[]>([])
const signals = ref<CorpusFeedSignalsResponse | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)
const nextCursor = ref<string | null>(null)
const descExpanded = ref(false)
const sortOrder = ref<'newest' | 'oldest'>('newest')
const graphError = ref<string | null>(null)

const PAGE = 50
const DESC_CLAMP = 220
const SIGNALS_TOP_K = 8

const title = computed(
  () => feed.value?.display_title?.trim() || subject.feedUiLabel || subject.feedId || 'Show',
)
const episodeCount = computed(() => feed.value?.episode_count ?? episodes.value.length)
const description = computed(() => feed.value?.description?.trim() || '')
const descIsLong = computed(() => description.value.length > DESC_CLAMP)

const topTopics = computed(() => signals.value?.top_topics ?? [])
const keyPeople = computed(() => signals.value?.key_people ?? [])
const recurringGuests = computed(() => signals.value?.recurring_guests ?? [])
const dominantThemes = computed(() => signals.value?.dominant_themes ?? [])
const trendingTopics = computed(() => signals.value?.trending_topics ?? [])
const grounding = computed(() => signals.value?.grounding ?? null)
const groundingPct = computed(() =>
  grounding.value ? Math.round(grounding.value.rate * 100) : null,
)
const hasSignals = computed(
  () =>
    topTopics.value.length > 0 ||
    keyPeople.value.length > 0 ||
    recurringGuests.value.length > 0 ||
    dominantThemes.value.length > 0 ||
    trendingTopics.value.length > 0 ||
    grounding.value != null,
)

function fmtDate(iso: string | null | undefined): string {
  const s = (iso ?? '').trim()
  return s ? s.slice(0, 10) : ''
}
function episodeSummary(e: CorpusEpisodeListItem): string {
  return (e.summary_preview?.trim() || e.summary_title?.trim() || '').trim()
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
      sort: sortOrder.value,
      withCilTopics: true,
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

function setSort(order: 'newest' | 'oldest'): void {
  if (sortOrder.value === order) return
  sortOrder.value = order
  nextCursor.value = null
  void loadEpisodes(true)
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
function onEpisodePillClick(e: CorpusEpisodeListItem, index: number): void {
  const pill = (e.cil_digest_topics ?? [])[index]
  if (pill?.topic_id) subject.focusTopic(pill.topic_id)
}

// Open the whole show on the graph: append its (loaded) episode KGs onto the graph
// canvas and switch to the Graph tab. There is no single "feed node" — the show's
// graph is the union of its episodes' knowledge graphs.
async function openShowInGraph(): Promise<void> {
  graphError.value = null
  const paths = episodes.value
    .filter((e) => e.has_kg && e.kg_relative_path)
    .map((e) => e.kg_relative_path as string)
  if (!paths.length) {
    graphError.value = 'No knowledge graphs on disk for this show yet.'
    return
  }
  emit('switch-main-tab', 'graph')
  try {
    await artifacts.appendRelativeArtifacts(paths)
  } catch (e) {
    graphError.value = e instanceof Error ? e.message : 'Could not load the show graph.'
  }
}

watch(
  () => subject.feedId,
  () => {
    nextCursor.value = null
    descExpanded.value = false
    signals.value = null
    graphError.value = null
    sortOrder.value = 'newest'
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
    <!-- Header — compact cover + title + meta; roomier summary + open-in-graph. -->
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
          <div class="flex items-start justify-between gap-2">
            <h3
              class="min-w-0 select-text text-base font-semibold leading-snug text-surface-foreground"
            >
              {{ title }}
            </h3>
            <button
              type="button"
              data-testid="show-rail-open-graph"
              class="shrink-0 rounded p-1 text-muted outline-none transition hover:bg-overlay hover:text-primary focus-visible:ring-2 focus-visible:ring-primary"
              title="Open this show on the graph"
              aria-label="Open this show on the graph"
              @click="openShowInGraph"
            >
              <svg
                viewBox="0 0 24 24"
                class="h-4 w-4"
                fill="none"
                stroke="currentColor"
                stroke-width="1.8"
                aria-hidden="true"
              >
                <circle cx="6" cy="6" r="2.4" />
                <circle cx="18" cy="7" r="2.4" />
                <circle cx="12" cy="17" r="2.4" />
                <path d="M8 7.5 15.6 8M7.4 8 11 14.8M16.6 9 13 15" />
              </svg>
            </button>
          </div>
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
            class="mt-1 text-xs leading-relaxed text-muted"
            :class="descExpanded ? '' : 'line-clamp-4'"
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
          <p v-if="graphError" class="mt-1 text-[11px] text-danger">{{ graphError }}</p>
        </div>
      </div>
    </div>

    <!-- Show-level signals (UXS-015 Phase 2 + operator feedback #9): topics, people,
         recurring guests, dominant themes, trending, grounding. Chips open in-rail. -->
    <div
      v-if="hasSignals"
      class="max-h-[42%] shrink-0 space-y-2 overflow-y-auto border-b border-border px-3 py-2"
      data-testid="show-rail-signals"
    >
      <div v-if="grounding" class="flex items-center gap-2" data-testid="show-rail-grounding">
        <span class="text-[11px] font-semibold uppercase tracking-wide text-muted">Grounding</span>
        <span
          class="rounded-full bg-emerald-700/25 px-2 py-0.5 text-[11px] font-semibold text-emerald-300"
        >
          {{ groundingPct }}% quote-backed
        </span>
        <span class="text-[11px] text-muted">
          {{ grounding.grounded_insights }}/{{ grounding.total_insights }} ·
          {{ grounding.people_count }} people
        </span>
      </div>

      <div v-if="dominantThemes.length">
        <h4 class="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted">Themes</h4>
        <div class="flex flex-wrap gap-1">
          <button
            v-for="th in dominantThemes"
            :key="th.theme_id"
            type="button"
            data-testid="show-rail-theme"
            class="rounded-full border px-2 py-0.5 text-[11px] font-medium outline-none transition hover:opacity-90 focus-visible:ring-2 focus-visible:ring-primary"
            style="border-color: rgba(125, 211, 192, 0.5); background-color: rgba(125, 211, 192, 0.18)"
            @click="openTopic(th.theme_id)"
          >
            {{ th.label }} <span class="text-muted">· {{ th.topic_count }}</span>
          </button>
        </div>
      </div>

      <div v-if="trendingTopics.length">
        <h4 class="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted">
          Trending
        </h4>
        <div class="flex flex-wrap gap-1">
          <button
            v-for="tr in trendingTopics"
            :key="tr.topic_id"
            type="button"
            data-testid="show-rail-trending"
            class="rounded-full bg-accent/20 px-2 py-0.5 text-[11px] font-medium text-accent outline-none transition hover:bg-accent/30 focus-visible:ring-2 focus-visible:ring-primary"
            @click="openTopic(tr.topic_id)"
          >
            {{ tr.label }} <span class="opacity-80">· {{ tr.velocity }}×</span>
          </button>
        </div>
      </div>

      <div v-if="topTopics.length">
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

      <div v-if="recurringGuests.length">
        <h4 class="mb-1 text-[11px] font-semibold uppercase tracking-wide text-muted">
          Recurring guests
        </h4>
        <div class="flex flex-wrap gap-1">
          <button
            v-for="p in recurringGuests"
            :key="p.person_id"
            type="button"
            data-testid="show-rail-recurring"
            class="rounded-full bg-overlay px-2 py-0.5 text-[11px] text-person outline-none transition hover:bg-overlay-2 focus-visible:ring-2 focus-visible:ring-primary"
            @click="openPerson(p.person_id)"
          >
            {{ p.name }} <span class="text-muted">· {{ p.episode_count }}</span>
          </button>
        </div>
      </div>
    </div>

    <!-- Episode list — sortable; each row: full summary + digest-parity topic pills. -->
    <div class="min-h-0 flex-1 overflow-y-auto p-2">
      <div class="mb-1 flex items-center justify-end gap-1 px-1">
        <span class="mr-auto text-[11px] text-muted">Sort</span>
        <button
          type="button"
          data-testid="show-rail-sort-newest"
          class="rounded px-1.5 py-0.5 text-[11px] outline-none focus-visible:ring-2 focus-visible:ring-primary"
          :class="sortOrder === 'newest' ? 'bg-overlay-2 text-primary' : 'text-muted hover:bg-overlay'"
          :aria-pressed="sortOrder === 'newest'"
          @click="setSort('newest')"
        >
          Newest
        </button>
        <button
          type="button"
          data-testid="show-rail-sort-oldest"
          class="rounded px-1.5 py-0.5 text-[11px] outline-none focus-visible:ring-2 focus-visible:ring-primary"
          :class="sortOrder === 'oldest' ? 'bg-overlay-2 text-primary' : 'text-muted hover:bg-overlay'"
          :aria-pressed="sortOrder === 'oldest'"
          @click="setSort('oldest')"
        >
          Oldest
        </button>
      </div>

      <p v-if="loading && episodes.length === 0" class="p-2 text-xs text-muted">
        Loading episodes…
      </p>
      <p v-else-if="error" class="p-2 text-xs text-danger" data-testid="show-rail-error">
        {{ error }}
      </p>
      <p
        v-else-if="episodes.length === 0"
        class="p-4 text-xs text-muted"
        data-testid="show-rail-empty"
      >
        No episodes.
      </p>
      <ul v-else class="space-y-0.5 text-sm">
        <li v-for="(e, i) in episodes" :key="e.metadata_relative_path">
          <div class="group flex w-full gap-2 rounded px-2 py-1.5 text-left">
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
              <div
                role="button"
                tabindex="0"
                data-library-episode-row
                :data-testid="`show-rail-episode-${i}`"
                class="cursor-pointer rounded outline-none focus-visible:ring-2 focus-visible:ring-primary"
                :aria-label="`${e.episode_title}, ${title}`"
                @click="selectEpisode(e)"
                @keydown.enter.prevent="selectEpisode(e)"
                @keydown.space.prevent="selectEpisode(e)"
              >
                <div class="flex items-baseline justify-between gap-2">
                  <span
                    class="min-w-0 flex-1 truncate font-medium text-surface-foreground group-hover:text-primary"
                    >{{ e.episode_title }}</span
                  >
                  <span v-if="fmtDate(e.publish_date)" class="shrink-0 text-[11px] text-muted">{{
                    fmtDate(e.publish_date)
                  }}</span>
                </div>
                <p v-if="episodeSummary(e)" class="mt-0.5 text-[11px] leading-snug text-muted">
                  {{ episodeSummary(e) }}
                </p>
              </div>
              <CilTopicPillsRow
                v-if="e.cil_digest_topics && e.cil_digest_topics.length"
                class="mt-1"
                :pills="e.cil_digest_topics"
                cluster-member-appearance="kg"
                truncation="none"
                max-width-class="auto"
                :data-testid="`show-rail-episode-pills-${i}`"
                @pill-click="(idx) => onEpisodePillClick(e, idx)"
              />
              <div v-if="e.has_gi || e.has_kg" class="mt-0.5 flex flex-wrap items-center gap-1">
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
