<script setup lang="ts">
/**
 * Home's "In progress" (operator 2026-10-11: "what's a door into things I started?"): every Learn
 * path the listener is on (up to three) and the last short listen left unfinished, above Continue
 * listening. A path's card plays the day to do now (the earliest open day not done), or — every
 * open day done — opens the next one now, or says the next opens tomorrow; the card itself opens
 * the week. The short listen resumes at its clip. Signed in only; hidden with nothing in progress.
 */
import { computed, onActivated, onDeactivated, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { clearReelProgress, getInProgress, learnOpenNext, type LearningPath, type ReelProgress } from '../services/api'
import { useGoneReferencesStore } from '../stores/goneReferences'
import GoneBadge from './GoneBadge.vue'
import { track } from '../services/analytics'
import { learnDayRoute } from '../services/learnPlay'
import { episodeArtwork } from '../utils/episode'
import CalendarMark from './CalendarMark.vue'
import EpisodeActions from './EpisodeActions.vue'
import { formatTime } from '../player/transcriptSync'
import type { EpisodeDetail } from '../services/types'

/** The episodes the listener is partway through, beyond Continue listening's hero, most recent
 *  first (operator 2026-10-14: one "In progress" for everything started; "Jump back in" folded in).
 *  Home owns them — they are its continue list — and a "Mark as played" takes one out at once. */
const props = withDefaults(defineProps<{ episodes?: { detail: EpisodeDetail; position: number }[] }>(), {
  episodes: () => [],
})
const { t } = useI18n()
const paths = ref<LearningPath[]>([])
const reel = ref<ReelProgress | null>(null)
const busy = ref(false)
const gone = useGoneReferencesStore()

let latest = 0
async function load(): Promise<void> {
  const run = ++latest
  const got = await getInProgress().catch(() => null)
  if (run !== latest || !got) return
  paths.value = got.paths
  reel.value = got.reel
}
onMounted(() => {
  void gone.ensureLoaded()
  void load()
})

/** A short listen whose subject the corpus no longer has cannot resume (#2345): remove it. */
async function removeReel(): Promise<void> {
  if (!reel.value) return
  const id = reel.value.subject_id
  await clearReelProgress(id).catch(() => undefined)
  if (reel.value?.subject_id === id) reel.value = null
}
// Home is kept alive; a day done or a reel finished elsewhere must show on return.
let leftOnce = false
onDeactivated(() => {
  leftOnce = true
})
onActivated(() => {
  if (leftOnce) void load()
})

// A finished path is not in progress: off Home at once (operator 2026-10-14); its week and
// Profile → Stats keep it. The server leaves it out; this covers a day done while Home was open.
const cards = computed(() =>
  paths.value.filter((p) => !p.finished_at && !(p.steps.length > 0 && p.steps.every((s) => s.done))).map((p) => {
    const now = p.steps.find((s) => s.unlocked && !s.done) ?? null
    // The day to do, else the newest open one: its episode's artwork is the card's.
    const face = now ?? [...p.steps].reverse().find((s) => s.unlocked) ?? p.steps[0] ?? null
    const done = p.steps.filter((s) => s.done).length
    return { path: p, now, done, art: face ? episodeArtwork(face.episode) : null }
  }),
)
const reelArt = computed(() => (reel.value?.episode ? episodeArtwork(reel.value.episode) : null))
const episodeCards = computed(() =>
  props.episodes.map(({ detail, position }) => {
    const total = detail.duration_seconds || 0
    return {
      slug: detail.slug,
      title: detail.title,
      show: detail.podcast_title ?? '',
      moments: !!detail.has_moments,
      art: episodeArtwork(detail),
      pct: total > 0 ? Math.min(100, Math.round((position / total) * 100)) : 0,
      minutesLeft: total > 0 ? Math.max(1, Math.round((total - position) / 60)) : 0,
    }
  }),
)
const reelPct = computed(() =>
  reel.value ? Math.round((100 * reel.value.index) / Math.max(1, reel.value.clips)) : 0,
)
/** "Day 3 opens in 14 h": days count 24 h from the start, not calendar days. */
function opensIn(path: LearningPath): string {
  const at = path.next_opens_at
  if (!at) return ''
  const hours = Math.ceil((at * 1000 - Date.now()) / 3_600_000)
  return hours <= 1
    ? t('inProgress.opensSoon', { n: path.day + 1 })
    : t('inProgress.opensIn', { n: path.day + 1, h: hours })
}

async function openNext(path: LearningPath): Promise<void> {
  if (busy.value) return
  busy.value = true
  try {
    const updated = await learnOpenNext(path.subject_id)
    paths.value = paths.value.map((p) => (p.subject_id === updated.subject_id ? updated : p))
    track('learn_path', { action: 'open_next', kind: path.subject_kind, day: updated.day, source: 'in_progress' })
  } catch {
    // Not opened: the button stays.
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <section v-if="cards.length || reel || episodes.length" class="mt-7" data-testid="home-in-progress">
    <div class="mb-3 flex items-baseline justify-between gap-3">
      <h2 class="lp-section">{{ t('inProgress.title') }}</h2>
      <span class="font-mono text-xs tabular-nums text-muted">{{ cards.length + (reel ? 1 : 0) + episodes.length }}</span>
    </div>
    <!-- A swipe of cards on a phone, a grid from md: each card is the door back into one thing. -->
    <ul class="lp-rail flex snap-x snap-mandatory gap-3 overflow-x-auto pb-2 md:grid md:grid-cols-2 md:overflow-visible lg:grid-cols-3">
      <li v-for="c in cards" :key="c.path.subject_id" class="w-[84%] shrink-0 snap-start md:w-auto">
        <article
          class="relative flex h-full flex-col overflow-hidden rounded-2xl border border-learn/30 bg-surface"
          :data-subject="c.path.subject_id"
          data-testid="in-progress-learn"
        >
          <img v-if="c.art" :src="c.art" alt="" class="absolute inset-0 h-full w-full object-cover opacity-30" />
          <div class="absolute inset-0 bg-gradient-to-t from-canvas via-canvas/80 to-canvas/30" />
          <div class="relative flex flex-1 flex-col p-4">
            <p class="inline-flex items-center gap-1.5 font-mono text-[11px] font-bold uppercase tracking-[0.12em] text-learn">
              <CalendarMark />{{ t('inProgress.learnKicker', { n: c.path.day, total: c.path.steps.length }) }}
            </p>
            <RouterLink
              :to="{ name: 'learn', query: { subject: c.path.subject_id } }"
              class="mt-1 line-clamp-2 font-display text-xl font-extrabold leading-tight tracking-tight text-canvas-foreground no-underline after:absolute after:inset-0"
              data-testid="in-progress-learn-open"
              @click="track('learn_path', { action: 'open', kind: c.path.subject_kind, source: 'in_progress' })"
            >{{ c.path.subject_label }}</RouterLink>
            <!-- The path is frozen and still plays; only its subject page is gone (#2345). -->
            <GoneBadge v-if="gone.isGone(c.path.subject_id)" class="mt-0.5" />
            <p v-if="c.now" class="mt-1 line-clamp-1 text-sm text-muted">{{ c.now.episode.title }}</p>
            <p v-else class="mt-1 text-sm text-muted">{{ t('inProgress.doneToday', { n: c.path.day }) }}</p>
            <!-- The week, a segment a day: done, open, not yet. -->
            <div class="mt-3 flex gap-1" role="img" :aria-label="t('learn.progress', { done: c.done, total: c.path.steps.length })">
              <span
                v-for="s in c.path.steps"
                :key="s.day"
                class="h-1.5 flex-1 rounded-full"
                :class="s.done ? 'bg-learn' : s.unlocked ? 'bg-learn/40' : 'bg-overlay'"
              />
            </div>
            <div class="relative z-30 mt-auto flex items-center gap-3 pt-3">
              <template v-if="c.now">
                <!-- The Resume hero's oval (operator 2026-10-11: one play control across Home). -->
                <RouterLink
                  :to="learnDayRoute(c.path.subject_id, c.now)"
                  class="inline-flex h-11 items-center gap-2 rounded-full bg-learn px-5 font-bold text-canvas no-underline"
                  data-testid="in-progress-learn-play"
                  @click="track('learn_path', { action: 'step_play', kind: c.path.subject_kind, day: c.now.day, source: 'in_progress' })"
                >▶ {{ t('inProgress.playDay', { n: c.now.day }) }} · {{ formatTime(c.now.start_ms / 1000) }}</RouterLink>
              </template>
              <template v-else>
                <button
                  v-if="c.path.can_open_next"
                  type="button"
                  class="inline-flex h-11 items-center gap-2 rounded-full bg-learn px-5 font-bold text-canvas"
                  :disabled="busy"
                  data-testid="in-progress-learn-next"
                  @click="openNext(c.path)"
                >{{ t('learn.openNext') }} ›</button>
                <span v-else class="text-sm text-muted" data-testid="in-progress-learn-later">{{ opensIn(c.path) }}</span>
              </template>
            </div>
          </div>
        </article>
      </li>
      <li v-if="reel" class="w-[84%] shrink-0 snap-start md:w-auto">
        <article class="relative flex h-full flex-col overflow-hidden rounded-2xl border border-listen/30 bg-surface" data-testid="in-progress-reel">
          <img v-if="reelArt" :src="reelArt" alt="" class="absolute inset-0 h-full w-full object-cover opacity-30" />
          <div class="absolute inset-0 bg-gradient-to-t from-canvas via-canvas/80 to-canvas/30" />
          <div class="relative flex flex-1 flex-col p-4">
            <p class="font-mono text-[11px] font-bold uppercase tracking-[0.12em] text-listen">
              ▶ {{ t('inProgress.reelKicker', { n: reel.index + 1, total: reel.clips }) }}
            </p>
            <p class="mt-1 line-clamp-2 font-display text-xl font-extrabold leading-tight tracking-tight text-canvas-foreground">{{ reel.subject_label }}</p>
            <GoneBadge v-if="gone.isGone(reel.subject_id)" class="mt-0.5" />
            <p v-if="reel.quote_text" class="mt-1 line-clamp-2 text-sm text-muted">“{{ reel.quote_text }}”</p>
            <p v-if="reel.speaker" class="text-xs text-muted">— {{ reel.speaker }}</p>
            <div class="mt-3 h-1.5 overflow-hidden rounded-full bg-overlay" role="img" :aria-label="t('inProgress.reelKicker', { n: reel.index + 1, total: reel.clips })">
              <div class="h-full rounded-full bg-listen" :style="{ width: `${reelPct}%` }" />
            </div>
            <div class="relative z-30 mt-auto flex items-center gap-3 pt-3">
              <button
                v-if="gone.isGone(reel.subject_id)"
                type="button"
                class="relative z-30 inline-flex h-11 items-center rounded-full border border-border px-5 font-bold text-canvas-foreground"
                data-testid="in-progress-reel-remove"
                @click="removeReel"
              >{{ t('inProgress.remove') }}</button>
              <RouterLink
                v-else
                :to="{ name: 'topic-reel', params: { id: reel.subject_id }, query: { minutes: String(reel.minutes ?? 15), at: String(reel.index + 1) } }"
                class="inline-flex h-11 items-center gap-2 rounded-full bg-listen px-5 font-bold text-canvas no-underline after:absolute after:inset-0"
                :aria-label="t('inProgress.resumeLabel', { subject: reel.subject_label, n: reel.index + 1 })"
                data-testid="in-progress-reel-resume"
                @click="track('reel_open', { source: 'in_progress', kind: reel.subject_kind })"
              >▶ {{ t('inProgress.resume') }} · {{ t('inProgress.clipN', { n: reel.index + 1 }) }}</RouterLink>
            </div>
          </div>
        </article>
      </li>
      <!-- Episodes partway through, after the paths and the short listen (operator 2026-10-14): the
           same card, its show over its title, how far, Resume, and ⋯ with Mark as played. -->
      <li v-for="e in episodeCards" :key="e.slug" class="w-[84%] shrink-0 snap-start md:w-auto">
        <article
          class="relative flex h-full flex-col overflow-hidden rounded-2xl border border-accent/30 bg-surface"
          :data-slug="e.slug"
          data-testid="in-progress-episode"
        >
          <img v-if="e.art" :src="e.art" alt="" class="absolute inset-0 h-full w-full object-cover opacity-30" />
          <div class="absolute inset-0 bg-gradient-to-t from-canvas via-canvas/80 to-canvas/30" />
          <div class="relative flex flex-1 flex-col p-4">
            <div class="flex items-start justify-between gap-2">
              <p class="min-w-0 truncate font-mono text-[11px] font-bold uppercase tracking-[0.12em] text-accent">
                {{ e.show }}
              </p>
              <EpisodeActions :slug="e.slug" :episode-title="e.title" :moments="e.moments" mark-played class="relative z-30 -mt-2 -mr-2" />
            </div>
            <RouterLink
              :to="{ name: 'player', params: { slug: e.slug } }"
              class="mt-1 line-clamp-2 font-display text-xl font-extrabold leading-tight tracking-tight text-canvas-foreground no-underline after:absolute after:inset-0"
              data-testid="in-progress-episode-open"
            >{{ e.title }}</RouterLink>
            <p v-if="e.minutesLeft" class="mt-1 text-sm text-muted">{{ t('inProgress.minutesLeft', { n: e.minutesLeft }) }}</p>
            <div class="mt-3 h-1.5 overflow-hidden rounded-full bg-overlay" aria-hidden="true">
              <div class="h-full rounded-full bg-accent" :style="{ width: `${e.pct}%` }" />
            </div>
            <div class="relative z-30 mt-auto flex items-center gap-3 pt-3">
              <RouterLink
                :to="{ name: 'player', params: { slug: e.slug }, query: { play: '1' } }"
                class="inline-flex h-11 items-center gap-2 rounded-full bg-accent px-5 font-bold text-accent-foreground no-underline"
                data-testid="in-progress-episode-resume"
              >▶ {{ t('inProgress.resume') }}</RouterLink>
            </div>
          </div>
        </article>
      </li>
    </ul>
  </section>
</template>
