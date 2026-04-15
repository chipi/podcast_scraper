<script setup lang="ts">
import { computed, ref } from 'vue'
import {
  fetchTopicTimeline,
  fetchTopicTimelineMerged,
  type CilArcEpisodeBlock,
  type CilTopicTimelineMergedResponse,
  type CilTopicTimelineResponse,
} from '../../api/cilApi'
import { useShellStore } from '../../stores/shell'
import HelpTip from './HelpTip.vue'
import PodcastCover from './PodcastCover.vue'
import { formatCalendarDateForDisplay } from '../../utils/formatting'
import { primaryTextFromLooseGiNode } from '../../utils/parsing'
import { StaleGeneration } from '../../utils/staleGeneration'

const shell = useShellStore()

const dialogRef = ref<HTMLDialogElement | null>(null)
const timelineMode = ref<'single' | 'cluster'>('single')
const topicIdOpen = ref('')
const clusterTopicIdsOpen = ref<string[]>([])
const loading = ref(false)
const errorText = ref<string | null>(null)
const payload = ref<CilTopicTimelineResponse | CilTopicTimelineMergedResponse | null>(null)

/** API returns oldest-first; default UI order is newest first (desc). */
const dateSortOrder = ref<'asc' | 'desc'>('desc')

const isClusterTimeline = computed(() => timelineMode.value === 'cluster')

const dialogTitle = computed(() =>
  isClusterTimeline.value ? 'Cluster timeline' : 'Topic timeline',
)

const topicIdsForA11y = computed((): string => {
  if (isClusterTimeline.value) {
    return clusterTopicIdsOpen.value.join(', ')
  }
  return topicIdOpen.value
})

const timelineLoadGate = new StaleGeneration()

const corpusPathForCovers = computed(() =>
  (shell.resolvedCorpusPath ?? shell.corpusPath ?? '').trim(),
)

function onBackdropClick(e: MouseEvent): void {
  const el = dialogRef.value
  if (el && e.target === el) {
    el.close()
  }
}

function close(): void {
  dialogRef.value?.close()
}

function resetForOpen(): void {
  loading.value = false
  errorText.value = null
  payload.value = null
}

function dedupeTopicIds(ids: string[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const raw of ids) {
    const t = String(raw).trim()
    if (!t || seen.has(t)) {
      continue
    }
    seen.add(t)
    out.push(t)
  }
  return out
}

function insightLine(ins: Record<string, unknown>): string {
  return primaryTextFromLooseGiNode(ins).trim() || '(no text)'
}

function episodePrimaryHeading(ep: CilArcEpisodeBlock): string {
  const t = ep.episode_title?.trim()
  if (t) {
    return t
  }
  const n = ep.episode_number
  if (n != null && Number.isFinite(Number(n))) {
    return `Episode ${n}`
  }
  const f = ep.feed_title?.trim()
  if (f) {
    return f
  }
  return 'Unnamed episode'
}

/** Human subtitle under the heading (not the internal episode_id). */
function episodeContextLine(ep: CilArcEpisodeBlock): string | null {
  const t = ep.episode_title?.trim()
  const f = ep.feed_title?.trim()
  const n = ep.episode_number
  const hasNum = n != null && Number.isFinite(Number(n))
  if (t && f) {
    return `Podcast: ${f}`
  }
  if (t && !f && hasNum) {
    return `Episode ${n}`
  }
  if (!t && f && hasNum) {
    return `Podcast: ${f}`
  }
  if (!t && f && !hasNum) {
    return 'No episode title in corpus metadata'
  }
  if (!t && !f && !hasNum) {
    return 'Corpus episode'
  }
  return null
}

function formatEpisodeDate(raw: string | null | undefined): string {
  if (!raw?.trim()) {
    return ''
  }
  return formatCalendarDateForDisplay(raw)
}

function internalEpisodeTooltip(ep: CilArcEpisodeBlock): string {
  return `Internal episode id: ${ep.episode_id}`
}

function episodeSortKey(ep: CilArcEpisodeBlock): string {
  const d = (ep.publish_date ?? '').trim()
  const id = String(ep.episode_id ?? '')
  return `${d}\0${id}`
}

const sortedEpisodes = computed((): CilArcEpisodeBlock[] => {
  const eps = payload.value?.episodes
  if (!eps?.length) {
    return []
  }
  const arr = [...eps]
  arr.sort((a, b) => {
    const cmp = episodeSortKey(a).localeCompare(episodeSortKey(b))
    return dateSortOrder.value === 'asc' ? cmp : -cmp
  })
  return arr
})

async function load(): Promise<void> {
  const path = (shell.resolvedCorpusPath ?? shell.corpusPath).trim()
  if (!path) {
    errorText.value =
      'Set a corpus path and load artifacts so the server can resolve the corpus root.'
    return
  }
  const seq = timelineLoadGate.bump()
  loading.value = true
  errorText.value = null
  payload.value = null
  try {
    let body: CilTopicTimelineResponse | CilTopicTimelineMergedResponse
    if (timelineMode.value === 'cluster') {
      const ids = clusterTopicIdsOpen.value
      if (ids.length === 0) {
        errorText.value = 'Missing cluster topic ids.'
        return
      }
      body = await fetchTopicTimelineMerged(path, ids)
    } else {
      const tid = topicIdOpen.value
      if (!tid) {
        errorText.value = 'Missing topic id.'
        return
      }
      body = await fetchTopicTimeline(path, tid)
    }
    if (timelineLoadGate.isStale(seq)) {
      return
    }
    payload.value = body
  } catch (e) {
    if (timelineLoadGate.isStale(seq)) {
      return
    }
    errorText.value = e instanceof Error ? e.message : String(e)
  } finally {
    if (timelineLoadGate.isCurrent(seq)) {
      loading.value = false
    }
  }
}

async function open(topicId: string): Promise<void> {
  timelineMode.value = 'single'
  clusterTopicIdsOpen.value = []
  topicIdOpen.value = topicId.trim()
  resetForOpen()
  dialogRef.value?.showModal()
  await load()
}

async function openCluster(topicIds: string[]): Promise<void> {
  const ids = dedupeTopicIds(topicIds)
  timelineMode.value = 'cluster'
  topicIdOpen.value = ''
  clusterTopicIdsOpen.value = ids
  resetForOpen()
  dialogRef.value?.showModal()
  await load()
}

defineExpose({ open, openCluster, close })
</script>

<template>
  <dialog
    ref="dialogRef"
    data-testid="topic-timeline-dialog"
    class="w-[min(100%,42rem)] max-h-[min(92vh,48rem)] overflow-hidden rounded-lg border border-border bg-surface text-surface-foreground shadow-xl [&::backdrop]:bg-black/40"
    aria-labelledby="topic-timeline-title"
    @click="onBackdropClick"
  >
    <div class="flex max-h-[min(92vh,48rem)] flex-col">
      <div class="flex shrink-0 items-start justify-between gap-3 border-b border-border px-4 py-3">
        <div class="min-w-0 flex-1 pr-2">
          <div class="flex items-center gap-1.5">
            <h2
              id="topic-timeline-title"
              class="text-sm font-semibold text-surface-foreground"
            >
              {{ dialogTitle }}
            </h2>
            <HelpTip
              class="shrink-0 self-center"
              :pref-width="400"
              :button-aria-label="
                isClusterTimeline
                  ? 'About cluster timeline and topic ids'
                  : 'About topic timeline and topic id'
              "
            >
              <p
                v-if="isClusterTimeline"
                class="font-sans text-[11px] leading-snug text-elevated-foreground"
              >
                <strong class="font-medium text-elevated-foreground">Corpus-wide</strong> (CIL /
                RFC-072): the server runs <strong class="font-medium text-elevated-foreground">one</strong>
                scan for <strong class="font-medium text-elevated-foreground">all cluster topic
                ids</strong> and merges episodes: each episode appears once, with insights about
                <strong class="font-medium text-elevated-foreground">any</strong> of those topics. The
                list can be
                <strong class="font-medium text-elevated-foreground">empty, one episode, or many</strong>
                — that is how many episodes match, not a bug. It is
                <strong class="font-medium text-elevated-foreground">not limited</strong> to nodes
                visible in the current graph. The How to read line explains the dots.
              </p>
              <p
                v-else
                class="font-sans text-[11px] leading-snug text-elevated-foreground"
              >
                <strong class="font-medium text-elevated-foreground">Corpus-wide</strong> (CIL /
                RFC-072): the server scans <strong class="font-medium text-elevated-foreground">every
                episode</strong> under your corpus path that has bridge + GI on disk, and lists those
                with at least one insight linked to this topic. The list can be
                <strong class="font-medium text-elevated-foreground">empty, one episode, or many</strong>
                — that is how many episodes match, not a bug. It is <strong
                  class="font-medium text-elevated-foreground"
                >not limited</strong> to nodes visible in the current graph. The How to read line
                explains the dots.
              </p>
              <p
                v-if="topicIdsForA11y"
                class="mt-2.5 border-t border-border pt-2.5 text-[11px] leading-snug text-muted"
              >
                <span class="mb-1 block font-medium text-elevated-foreground">{{
                  isClusterTimeline ? 'Topic ids (cluster)' : 'Topic id'
                }}</span>
                <span class="block break-all font-mono text-[11px] text-elevated-foreground">
                  {{ topicIdsForA11y }}
                </span>
              </p>
            </HelpTip>
          </div>
          <span
            v-if="topicIdsForA11y"
            class="sr-only"
            data-testid="topic-timeline-topic-id"
          >
            {{ isClusterTimeline ? 'Topic ids' : 'Topic id' }}: {{ topicIdsForA11y }}
          </span>
        </div>
        <button
          type="button"
          class="shrink-0 rounded border border-border px-2 py-1 text-xs hover:bg-overlay"
          data-testid="topic-timeline-close"
          @click="close"
        >
          Close
        </button>
      </div>

      <div
        class="min-h-0 flex-1 overflow-y-auto px-4 py-3 text-xs leading-relaxed"
        data-testid="topic-timeline-body"
      >
        <p
          v-if="loading"
          class="text-muted"
          data-testid="topic-timeline-loading"
        >
          Loading…
        </p>
        <p
          v-else-if="errorText"
          class="text-destructive"
          data-testid="topic-timeline-error"
        >
          {{ errorText }}
        </p>
        <div
          v-else-if="payload && payload.episodes.length === 0"
          class="space-y-2 text-muted"
          data-testid="topic-timeline-empty"
        >
          <p>
            No matching episodes in this corpus for
            {{ isClusterTimeline ? 'these cluster topics' : 'this topic' }}.
          </p>
          <p class="text-[11px] leading-snug">
            The server scanned your whole corpus path; <strong
              class="font-medium text-surface-foreground"
            >zero</strong> means no episode had bridge + GI <strong
              class="font-medium text-surface-foreground"
            >and</strong> at least one <strong class="font-medium text-surface-foreground">ABOUT</strong>
            edge from an insight to
            {{ isClusterTimeline ? 'a topic id that matched' : 'this topic' }}.
            <span class="font-mono">g:topic:…</span> ids are normalized to match bridge/GI.
            {{ isClusterTimeline ? 'Cluster timelines prefer ids from loaded graph Topic nodes when present; ' : '' }}Common causes:
            <span class="font-mono">topic_clusters.json</span> slugs differ from GI topic nodes, topics listed in bridge but never linked via ABOUT in GI, <span class="font-mono">k:topic:…</span> without GI, or missing <span class="font-mono">*.bridge.json</span>.
          </p>
        </div>
        <div
          v-else-if="payload"
          class="space-y-0"
          data-testid="topic-timeline-episodes"
        >
          <div
            class="mb-3 space-y-2 border-b border-border pb-3"
            data-testid="topic-timeline-legend"
          >
            <p class="text-[10px] leading-snug text-muted">
              <span class="font-medium text-surface-foreground">Corpus scan:</span>
              <span data-testid="topic-timeline-episode-count">
                {{
                  payload.episodes.length === 1
                    ? '1 episode in this corpus'
                    : `${payload.episodes.length} episodes in this corpus`
                }}
              </span>
              with insights about
              {{ isClusterTimeline ? 'any member topic in this cluster' : 'this topic' }}
              (whole path, not just the graph).
              <span class="font-medium text-surface-foreground">How to read:</span>
              each block is one episode; lines below are separate insights.
            </p>
            <div
              class="flex flex-wrap items-center gap-2"
              role="group"
              aria-label="Sort episodes by date"
            >
              <span class="text-[10px] font-medium text-muted">Date</span>
              <button
                type="button"
                class="rounded border px-2 py-0.5 text-[10px] transition-colors"
                :class="
                  dateSortOrder === 'asc'
                    ? 'border-gi/60 bg-gi/15 font-medium text-surface-foreground ring-1 ring-gi/35'
                    : 'border-border text-muted hover:bg-overlay'
                "
                data-testid="topic-timeline-sort-asc"
                @click="dateSortOrder = 'asc'"
              >
                Oldest first
              </button>
              <button
                type="button"
                class="rounded border px-2 py-0.5 text-[10px] transition-colors"
                :class="
                  dateSortOrder === 'desc'
                    ? 'border-gi/60 bg-gi/15 font-medium text-surface-foreground ring-1 ring-gi/35'
                    : 'border-border text-muted hover:bg-overlay'
                "
                data-testid="topic-timeline-sort-desc"
                @click="dateSortOrder = 'desc'"
              >
                Newest first
                <span
                  v-if="dateSortOrder === 'desc'"
                  class="ml-1 text-[9px] text-gi"
                >(default)</span>
              </button>
            </div>
          </div>
          <div
            v-for="(ep, ei) in sortedEpisodes"
            :key="`${ep.episode_id}-${ei}`"
            class="flex gap-3 sm:gap-4"
            :class="ei < sortedEpisodes.length - 1 ? 'mb-6 border-b border-border/50 pb-6' : ''"
            :data-testid="'topic-timeline-episode-' + ei"
          >
            <div
              class="shrink-0 self-start"
              :data-testid="'topic-timeline-episode-cover-' + ei"
            >
              <PodcastCover
                :corpus-path="corpusPathForCovers"
                :episode-image-local-relpath="ep.episode_image_local_relpath"
                :feed-image-local-relpath="ep.feed_image_local_relpath"
                :episode-image-url="ep.episode_image_url"
                :feed-image-url="ep.feed_image_url"
                :alt="`Cover for ${episodePrimaryHeading(ep)}`"
                size-class="h-14 w-14 sm:h-16 sm:w-16"
              />
            </div>
            <section class="min-w-0 flex-1">
              <time
                v-if="ep.publish_date && formatEpisodeDate(ep.publish_date)"
                class="block text-[11px] font-medium text-gi/90"
                :datetime="ep.publish_date"
              >
                {{ formatEpisodeDate(ep.publish_date) }}
              </time>
              <p
                v-else-if="ep.publish_date"
                class="text-[11px] font-medium text-gi/90"
              >
                {{ ep.publish_date }}
              </p>
              <p
                v-else
                class="text-[11px] text-muted"
              >
                Date unknown
              </p>
              <h3
                class="mt-0.5 text-[13px] font-semibold leading-snug text-surface-foreground"
                :data-testid="'topic-timeline-episode-title-' + ei"
              >
                {{ episodePrimaryHeading(ep) }}
              </h3>
              <p
                v-if="episodeContextLine(ep)"
                class="mt-0.5 text-[11px] leading-snug text-muted"
                :title="internalEpisodeTooltip(ep)"
                :data-testid="'topic-timeline-episode-context-' + ei"
              >
                {{ episodeContextLine(ep) }}
              </p>
              <ul
                class="mt-2 list-none space-y-1.5 pl-0"
                role="list"
              >
                <li
                  v-for="(ins, ii) in ep.insights"
                  :key="ii"
                  class="text-[11px] leading-snug text-surface-foreground"
                  :data-testid="'topic-timeline-insight-' + ei + '-' + ii"
                >
                  <span class="text-muted">· </span>{{ insightLine(ins) }}
                </li>
              </ul>
            </section>
          </div>
        </div>
      </div>
    </div>
  </dialog>
</template>
