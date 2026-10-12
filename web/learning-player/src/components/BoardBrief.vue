<script setup lang="ts">
/**
 * A board's Brief (RFC-119 amendment 2026-10-12, #2340), in a sheet opened from the "Brief" button
 * on the board's header (operator 2026-10-12: not shown by default; "same as the episode").
 *
 * Laid out as the episode's Brief (`KnowledgePanel`) is, so the two read as one thing: the grab
 * handle, "Brief" with ✕, then what it is about (the board's name, what it holds), the play button
 * into its moments, and folding sections with their counts — Key points as accent-ruled rows, and
 * Topics & People as one row of pills, the storylines and themes first, every pill naming its kind.
 *
 * Only what is saved on the board: a saved show, topic or person is not expanded into its episodes
 * (operator 2026-10-12). A board with no episodes says it has nothing to brief yet.
 *
 * A native <dialog> + `showModal()`, as the episode description sheet uses: focus trap, Escape, an
 * inert page behind it. Anything that navigates closes the sheet first, so it never sits over the
 * page it opened.
 */
import { computed, nextTick, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import CloseIcon from './CloseIcon.vue'
import CollapsibleSection from './CollapsibleSection.vue'
import TopVoices from './TopVoices.vue'
import { useSheetDrag } from '../composables/useSheetDrag'
import { BOARD_REEL_PREFIX, getBoardBrief, getTopicReel } from '../services/api'
import { track } from '../services/analytics'
import { minutesValue } from '../services/moments'
import type { BoardBrief, Entity } from '../services/types'
import { kindPill } from '../utils/interests'

const props = defineProps<{ open: boolean; boardId: string; boardName: string; version?: number }>()
const emit = defineEmits<{ (e: 'close'): void }>()

const { t } = useI18n()
const router = useRouter()
const el = ref<HTMLDialogElement | null>(null)
const panelEl = ref<HTMLElement | null>(null)
const handleDrag = useSheetDrag(panelEl, () => emit('close'))

const brief = ref<BoardBrief | null>(null)
const failed = ref(false)
const clips = ref(0)
const seconds = ref(0)
const KEY_POINTS_COLLAPSED = 3
const keyPointsExpanded = ref(false)
const TOPICS_COLLAPSED = 12
const topicsExpanded = ref(false)

let latest = 0
async function load(): Promise<void> {
  const run = ++latest
  failed.value = false
  keyPointsExpanded.value = false
  topicsExpanded.value = false
  const subject = BOARD_REEL_PREFIX + props.boardId
  const [b, reel] = await Promise.all([
    getBoardBrief(props.boardId).catch(() => null),
    getTopicReel(subject, 15).catch(() => null),
  ])
  if (run !== latest) return
  brief.value = b
  failed.value = b === null
  clips.value = reel?.clips.length ?? 0
  seconds.value = reel?.total_seconds ?? 0
}
// Fetched when it opens, and again when the board changes while it is open.
watch(
  () => [props.open, props.boardId, props.version] as const,
  ([open]) => {
    if (open) void load()
  },
  { immediate: true },
)

watch(
  [() => props.open, el],
  async ([open]) => {
    await nextTick()
    const d = el.value
    if (!d) return
    if (open && !d.open) d.showModal()
    else if (!open && d.open) d.close()
  },
  { immediate: true },
)

function onBackdropClick(e: MouseEvent): void {
  if (e.target === el.value) emit('close')
}
/** Every link in the sheet leaves the board: close first, so the sheet is not left over the page. */
function onBodyClick(e: MouseEvent): void {
  if ((e.target as HTMLElement | null)?.closest('a[href]')) emit('close')
}

const holdsLine = computed(() => {
  const h = brief.value?.holds
  if (!h) return ''
  const parts: [number, string][] = [
    [h.episodes, 'holdsEpisodes'],
    [h.moments, 'holdsMoments'],
    [h.shows, 'holdsShows'],
    [h.subjects, 'holdsSubjects'],
    [h.searches, 'holdsSearches'],
    [h.links, 'holdsLinks'],
  ]
  return parts
    .filter(([n]) => n > 0)
    .map(([n, key]) => t(`boardBrief.${key}`, { count: n }, n))
    .join(' · ')
})

const points = computed(() => brief.value?.points ?? [])
const visiblePoints = computed(() =>
  keyPointsExpanded.value ? points.value : points.value.slice(0, KEY_POINTS_COLLAPSED),
)

/** "What this board's about", as the show page's band: a row per kind — storylines, themes,
 * topics — most of the board's episodes first, no numbers on the pills. */
const storylines = computed(() => (brief.value?.threads ?? []).filter((x) => x.kind === 'storyline'))
const themes = computed(() => (brief.value?.threads ?? []).filter((x) => x.kind === 'theme'))
const topics = computed(() => brief.value?.about ?? [])
const shownTopics = computed(() => (topicsExpanded.value ? topics.value : topics.value.slice(0, TOPICS_COLLAPSED)))
const hasAbout = computed(() => storylines.value.length + themes.value.length + topics.value.length > 0)
/** Top voices: who speaks in most of the board's episodes, in the topic page's grid. */
const voices = computed<Entity[]>(() =>
  (brief.value?.voices ?? []).map((v) => ({ id: v.id, name: v.name, kind: 'person', image_url: v.image_url })),
)

function pointTo(p: BoardBrief['points'][number]) {
  return {
    name: 'player',
    params: { slug: p.episode.slug },
    ...(p.start_ms != null ? { query: { t: String(Math.floor(p.start_ms / 1000)) } } : {}),
  }
}

function playMoments(): void {
  track('reel_open', { source: 'board', kind: 'board' })
  emit('close')
  void router.push({
    name: 'topic-reel',
    params: { id: BOARD_REEL_PREFIX + props.boardId },
    query: { minutes: '15' },
  })
}
</script>

<template>
  <dialog
    ref="el"
    data-testid="board-brief"
    :aria-label="t('boardBrief.title')"
    class="fixed inset-x-0 bottom-0 top-[var(--lp-sheet-top)] m-0 h-[calc(100dvh-var(--lp-sheet-top))] max-h-none w-full max-w-none border-0 bg-transparent p-0 text-canvas-foreground backdrop:bg-black/50 lg:inset-0 lg:m-auto lg:h-fit lg:max-h-[80dvh] lg:max-w-2xl"
    @close="emit('close')"
    @click="onBackdropClick"
  >
    <div
      ref="panelEl"
      class="flex h-full flex-col overflow-hidden rounded-t-2xl border-t border-border bg-surface pb-[env(safe-area-inset-bottom)] lg:max-h-[80dvh] lg:rounded-2xl lg:border lg:pb-0"
    >
      <div
        class="flex h-6 shrink-0 touch-none items-center justify-center lg:hidden"
        aria-hidden="true"
        data-testid="board-brief-handle"
        v-bind="handleDrag"
      >
        <span class="h-1.5 w-10 rounded-full bg-border"></span>
      </div>
      <header class="flex items-center justify-between border-b border-border px-4 py-3">
        <span class="font-display text-lg font-bold">{{ t('boardBrief.title') }}</span>
        <button
          type="button"
          class="lp-nav shrink-0"
          data-testid="board-brief-close"
          :aria-label="t('boardBrief.close')"
          @click="emit('close')"
        >
          <CloseIcon />
        </button>
      </header>

      <div class="min-h-0 flex-1 overflow-y-auto px-4 py-4" @click="onBodyClick">
        <!-- What this is about, as the episode's Brief opens: a kicker, the name, then the room. -->
        <section class="mb-5" data-testid="board-brief-dossier">
          <p class="lp-kicker mb-0.5 text-muted">{{ t('ec.board') }}</p>
          <h2 class="font-display text-xl font-bold leading-tight text-canvas-foreground">{{ boardName }}</h2>
          <p v-if="holdsLine" class="mt-1 text-sm text-muted" data-testid="board-brief-holds">{{ holdsLine }}</p>
        </section>

        <p v-if="failed" class="text-sm text-muted" data-testid="board-brief-failed">{{ t('boardBrief.unavailable') }}</p>
        <p v-else-if="brief && brief.holds.episodes === 0" class="text-sm text-muted" data-testid="board-brief-empty">
          {{ t('boardBrief.empty') }}
        </p>

        <template v-else-if="brief">
          <!-- The board's own moments, the episode Brief's play button: an action, amber outline. -->
          <button
            v-if="clips > 0"
            type="button"
            class="mb-5 flex w-full items-center gap-3 rounded border border-accent px-3 py-2.5 text-left"
            data-testid="board-brief-play"
            @click="playMoments"
          >
            <span
              class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent text-accent-foreground"
              aria-hidden="true"
            >▶</span>
            <span class="min-w-0">
              <span class="block text-sm font-bold">{{ t('moments.playCount', { count: clips }, clips) }}</span>
              <span class="block text-xs text-muted">{{
                t('boardBrief.playHint', { length: t('moments.minutes', { n: minutesValue(seconds) }) })
              }}</span>
            </span>
          </button>

          <!-- Top voices, as on a topic: a board has no host or guest of its own, it has the
               people who speak across its episodes (operator 2026-10-12). -->
          <TopVoices
            v-if="voices.length"
            class="mb-5"
            :people="voices"
            :limit="8"
            :route-for="(id: string) => ({ name: 'person', params: { id } })"
          />

          <CollapsibleSection
            v-if="points.length"
            :title="t('kp.keyPoints')"
            :count="points.length"
            section-key="board-key-points"
            class="mb-5"
          >
            <ul class="flex flex-col gap-2.5" data-testid="board-brief-points">
              <li v-for="p in visiblePoints" :key="p.insight_id + p.episode.slug">
                <RouterLink
                  :to="pointTo(p)"
                  class="block border-l-2 border-accent/50 pl-3 no-underline"
                  data-testid="board-brief-point"
                >
                  <span class="block text-sm leading-relaxed text-surface-foreground">{{ p.text }}</span>
                  <span class="mt-0.5 block truncate text-xs text-muted">{{
                    p.speaker
                      ? t('boardBrief.pointBy', { speaker: p.speaker, episode: p.episode.title })
                      : p.episode.title
                  }}</span>
                </RouterLink>
              </li>
            </ul>
            <button
              v-if="!keyPointsExpanded && points.length > KEY_POINTS_COLLAPSED"
              type="button"
              data-testid="board-brief-points-more"
              class="mt-3 text-sm font-bold text-accent"
              @click="keyPointsExpanded = true"
            >
              {{ t('home.showMore', { count: points.length - KEY_POINTS_COLLAPSED }) }}
            </button>
          </CollapsibleSection>

          <CollapsibleSection
            v-if="hasAbout"
            :title="t('boardBrief.about')"
            section-key="board-about"
            class="mb-5"
          >
            <div v-if="storylines.length" class="mb-3">
              <h3 class="lp-kicker mb-1.5">{{ t('podcast.sigStorylines') }}</h3>
              <div class="flex flex-wrap gap-1.5">
                <RouterLink
                  v-for="th in storylines"
                  :key="th.id"
                  :to="{ name: 'storyline', params: { id: th.id } }"
                  class="inline-flex items-center rounded-full px-2.5 py-1 text-xs no-underline transition"
                  :class="kindPill('storyline')"
                  data-testid="board-brief-thread"
                >{{ th.label }}</RouterLink>
              </div>
            </div>
            <div v-if="themes.length" class="mb-3">
              <h3 class="lp-kicker mb-1.5">{{ t('podcast.sigThemes') }}</h3>
              <div class="flex flex-wrap gap-1.5">
                <RouterLink
                  v-for="th in themes"
                  :key="th.id"
                  :to="{ name: 'theme', params: { id: th.id } }"
                  class="inline-flex items-center rounded-full px-2.5 py-1 text-xs no-underline transition hover:opacity-80"
                  :class="kindPill('theme')"
                  data-testid="board-brief-thread"
                >{{ th.label }}</RouterLink>
              </div>
            </div>
            <div v-if="topics.length">
              <h3 class="lp-kicker mb-1.5">{{ t('podcast.sigTopics') }}</h3>
              <div class="flex flex-wrap gap-1.5" data-testid="board-brief-topics">
                <RouterLink
                  v-for="tp in shownTopics"
                  :key="tp.id"
                  :to="{ name: 'topic', params: { id: tp.id } }"
                  class="inline-flex items-center rounded-full px-2.5 py-1 text-xs no-underline transition hover:opacity-80"
                  :class="[kindPill('topic'), tp.saved ? 'font-semibold ring-1 ring-inset ring-accent' : '']"
                  :title="tp.saved ? t('boardBrief.savedHere') : undefined"
                  data-testid="board-brief-subject"
                >{{ tp.label }}</RouterLink>
                <button
                  v-if="!topicsExpanded && topics.length > TOPICS_COLLAPSED"
                  type="button"
                  data-testid="board-brief-topics-more"
                  class="rounded-full px-2.5 py-1 text-xs font-bold text-accent hover:bg-overlay"
                  @click="topicsExpanded = true"
                >
                  {{ t('kp.moreTags', { count: topics.length - TOPICS_COLLAPSED }) }}
                </button>
              </div>
            </div>
          </CollapsibleSection>
        </template>
      </div>
    </div>
  </dialog>
</template>
