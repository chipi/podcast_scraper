<script setup lang="ts">
/**
 * A board's Brief, at the top of the open board (RFC-119 amendment 2026-10-12, #2340): what the
 * board holds, what its episodes share, each episode's key point to hear where it is said, the
 * storylines and themes that link them, and the door into the board's own short listen.
 *
 * Only what is saved on the board: a saved show, topic or person is not expanded into its episodes
 * (operator 2026-10-12). A board with no episodes (shows, searches and links only) has nothing to
 * brief, and nothing renders. `version` changes when the board's items do, so the Brief follows an
 * add or a remove instead of describing the board as it was when it opened.
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import ReelDoor from './ReelDoor.vue'
import { BOARD_REEL_PREFIX, getBoardBrief } from '../services/api'
import type { BoardBrief } from '../services/types'

const props = defineProps<{ boardId: string; boardName: string; version?: number }>()
const { t } = useI18n()
const brief = ref<BoardBrief | null>(null)
const failed = ref(false)

let latest = 0
async function load(): Promise<void> {
  const run = ++latest
  failed.value = false
  try {
    const b = await getBoardBrief(props.boardId)
    if (run === latest) brief.value = b
  } catch {
    if (run === latest) {
      brief.value = null
      failed.value = true
    }
  }
}
watch(() => [props.boardId, props.version] as const, load, { immediate: true })

/** "3 episodes · 1 saved moment · 1 show": only the kinds the board has. */
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

const shown = computed(() => (brief.value && brief.value.holds.episodes > 0 ? brief.value : null))

function pointTo(p: BoardBrief['points'][number]) {
  return {
    name: 'player',
    params: { slug: p.episode.slug },
    ...(p.start_ms != null ? { query: { t: String(Math.floor(p.start_ms / 1000)) } } : {}),
  }
}
</script>

<template>
  <section
    v-if="shown"
    class="mb-3 space-y-3 rounded-xl border border-border bg-elevated/40 p-3"
    data-testid="board-brief"
  >
    <div class="flex flex-wrap items-center justify-between gap-2">
      <div class="min-w-0">
        <h3 class="lp-kicker text-muted">{{ t('boardBrief.title') }}</h3>
        <p class="text-xs text-muted" data-testid="board-brief-holds">{{ holdsLine }}</p>
      </div>
      <ReelDoor :subject-id="BOARD_REEL_PREFIX + boardId" :label="boardName" source="board" />
    </div>

    <section v-if="shown.about.length" data-testid="board-brief-about">
      <h4 class="lp-kicker mb-1.5 text-muted">{{ t('boardBrief.about') }}</h4>
      <ul class="flex flex-wrap gap-2">
        <li v-for="s in shown.about" :key="s.id">
          <RouterLink
            :to="{ name: s.kind, params: { id: s.id } }"
            class="inline-flex items-center gap-1 rounded-full border border-border px-3 py-1 text-sm text-canvas-foreground no-underline transition hover:bg-overlay"
            :class="s.saved ? 'border-accent' : ''"
            :title="s.saved ? t('boardBrief.savedHere') : undefined"
            data-testid="board-brief-subject"
          >
            {{ s.label }}
            <span class="text-xs text-muted">{{ t('boardBrief.inEpisodes', { count: s.episodes }, s.episodes) }}</span>
          </RouterLink>
        </li>
      </ul>
    </section>

    <section v-if="shown.points.length" data-testid="board-brief-points">
      <h4 class="lp-kicker mb-1.5 text-muted">{{ t('boardBrief.points') }}</h4>
      <ul class="space-y-2">
        <li v-for="p in shown.points" :key="p.insight_id + p.episode.slug" class="flex gap-2">
          <span class="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" aria-hidden="true" />
          <RouterLink :to="pointTo(p)" class="min-w-0 no-underline" data-testid="board-brief-point">
            <span class="block text-sm leading-relaxed text-canvas-foreground">{{ p.text }}</span>
            <span class="block truncate text-xs text-muted">{{
              p.speaker
                ? t('boardBrief.pointBy', { speaker: p.speaker, episode: p.episode.title })
                : p.episode.title
            }}</span>
          </RouterLink>
        </li>
      </ul>
    </section>

    <section v-if="shown.threads.length" data-testid="board-brief-threads">
      <h4 class="lp-kicker mb-1.5 text-muted">{{ t('boardBrief.threads') }}</h4>
      <ul class="space-y-1.5">
        <li v-for="th in shown.threads" :key="th.kind + th.id">
          <RouterLink
            :to="{ name: th.kind, params: { id: th.id } }"
            class="inline-flex items-center gap-1.5 text-sm font-semibold text-accent no-underline"
            data-testid="board-brief-thread"
          >
            <span class="lp-kicker text-muted">{{ t(`ec.${th.kind}`) }}</span>
            {{ th.label }}
            <span class="text-xs font-normal text-muted">{{ t('boardBrief.inEpisodes', { count: th.episodes }, th.episodes) }}</span>
          </RouterLink>
        </li>
      </ul>
    </section>
  </section>
  <p v-else-if="failed" class="mb-3 text-xs text-muted" data-testid="board-brief-failed">
    {{ t('boardBrief.unavailable') }}
  </p>
</template>
