<script setup lang="ts">
/**
 * N minutes on a topic, theme or storyline — goal-shaped listening (ADR-163 §8 amendment).
 *
 * The strongest clips ABOUT one subject (a theme or storyline: any of its member topics) from
 * different episodes, back to back (`GET /discover/reel`,
 * the Moments picker's clip windows). The player store plays them: each clip loads its own episode
 * (`startTopicReel` with the shell's `resolveNextUpForSlug`), and like any reel it never moves a
 * resume point or counts as a listen until "Listen to the full episode" (on from the clip, or from
 * the start) makes it one.
 *
 * - The title is the subject; under it what the reel actually runs ("2:24 · 12 clips · 9 episodes"),
 *   never the budget it was asked for (operator 2026-10-11: "real length everywhere").
 * - No start page (operator 2026-10-11): it starts on clip 1 at once. Opened by a tap the sound
 *   plays; where the browser blocks it (a link or reload, no tap behind it) clip 1 waits with ▶.
 * - One bar for the whole reel (time heard of time total), not a dot per clip; under the clip, the
 *   clips still to come (speaker · episode), each a way to jump there.
 * - Leaving the page keeps it playing; the mini-player brings you back here.
 * - After the last clip, the end card lists the episodes heard, each a way to listen in full.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import BackIcon from '../components/BackIcon.vue'
import { getTopicReel, type TopicReel } from '../services/api'
import { resolveNextUpForSlug } from '../services/nextUp'
import { formatTime } from '../player/transcriptSync'
import EpisodeListenChoice from '../components/EpisodeListenChoice.vue'
import { usePlayerStore, type ReelMoment } from '../stores/player'
import { toCountBucket, toRankBucket, track } from '../services/analytics'
import { artworkThumb, episodeArtwork } from '../utils/episode'

const props = defineProps<{ id: string; minutes?: number; at?: number }>()
const { t } = useI18n()
const router = useRouter()
const player = usePlayerStore()

const data = ref<TopicReel | null>(null)
const failed = ref(false)
const budget = computed(() => props.minutes ?? 15)

onMounted(async () => {
  try {
    data.value = await getTopicReel(props.id, budget.value)
  } catch {
    failed.value = true
    return
  }
  // No start page: the reel starts at once. Opened by a tap (a pill, a door) the audio plays; where
  // the browser blocks it (a link, a reload: no tap behind it) the reel waits on clip 1 and the
  // card's ▶ resumes it. A reel already playing (back from the mini-player) is left as it is.
  // `at` (1-based) resumes a short listen at that clip (Home's "In progress").
  const at = Math.min(Math.max(1, props.at ?? 1), data.value.clips.length) - 1
  if (data.value.clips.length && !live.value) void start(at)
})

const moments = computed<ReelMoment[]>(() =>
  (data.value?.clips ?? []).map((c) => ({
    insightId: c.insight_id,
    text: c.quote_text || c.text,
    speaker: c.speaker,
    startMs: c.play_start_ms ?? c.start_ms,
    endMs: c.play_end_ms ?? c.end_ms,
    slug: c.episode.slug,
    episodeTitle: c.episode.title,
    showTitle: c.episode.podcast_title,
    artwork: episodeArtwork(c.episode),
  })),
)
/** This page's reel is the one playing (the listener may have started another since). */
const live = computed(() => player.reel?.topic?.id === props.id ? player.reel : null)
const current = computed(() => (live.value ? live.value.moments[live.value.index] : null))
const episodesHeard = computed(() => {
  const seen = new Map<string, ReelMoment>()
  for (const m of moments.value) if (m.slug && !seen.has(m.slug)) seen.set(m.slug, m)
  return [...seen.values()]
})

/** The clip on the card: the playing one, or the first while the reel waits for a tap. */
const shown = computed(() => current.value ?? moments.value[0] ?? null)
const shownIndex = computed(() => (live.value ? live.value.index : 0))
const lengthMs = (m: ReelMoment) => Math.max(0, m.endMs - m.startMs)
const totalMs = computed(() => moments.value.reduce((n, m) => n + lengthMs(m), 0))
/** Time heard: the clips before this one, and how far into this one the audio is. */
const heardMs = computed(() => {
  const r = live.value
  if (!r) return 0
  if (r.done) return totalMs.value
  const before = r.moments.slice(0, r.index).reduce((n, m) => n + lengthMs(m), 0)
  const clip = r.moments[r.index]
  const into = Math.min(lengthMs(clip), Math.max(0, player.currentTime * 1000 - clip.startMs))
  return before + into
})
const heardPct = computed(() => (totalMs.value ? Math.min(100, (100 * heardMs.value) / totalMs.value) : 0))
const upNext = computed(() =>
  moments.value.map((m, i) => ({ m, i })).filter(({ i }) => i > shownIndex.value),
)
const episodeCount = computed(() => episodesHeard.value.length)
const playingNow = computed(() => !!live.value && !live.value.done && player.playing)

/** One reel event (operator 2026-10-11: real telemetry for everything new): which action, on which
 *  kind of subject, on which clip — never which subject or episode. */
function report(action: 'start' | 'next' | 'prev' | 'jump' | 'keep' | 'keep_start' | 'done'): void {
  if (!data.value) return
  const at = live.value ? live.value.index + 1 : undefined
  track('reel_action', {
    action,
    kind: data.value.subject_kind,
    clips: toCountBucket(moments.value.length),
    ...(at ? { clip: toRankBucket(at) } : {}),
  })
}
watch(
  () => live.value?.done,
  (done, was) => {
    if (done && !was) report('done')
  },
)
function next(): void {
  report('next')
  player.reelNext()
}
function prev(): void {
  report('prev')
  player.reelPrev()
}

async function start(at = 0): Promise<void> {
  if (!data.value) return
  report('start')
  await player.startTopicReel(
    { id: props.id, label: data.value.subject_label, minutes: budget.value, kind: data.value.subject_kind },
    moments.value,
    resolveNextUpForSlug,
    at,
  )
}
/** ▶ / ❚❚ on the card: start the reel, or pause and resume it (never a `play_start`: a reel is
 *  not a listen). */
function togglePlay(): void {
  if (!live.value || live.value.done) {
    void start()
    return
  }
  if (player.playing) player.el?.pause()
  else player.resumeAfterInterruption()
}
function jump(i: number): void {
  if (live.value && !live.value.done) {
    report('jump')
    void player.reelGo(i)
  }
  else void start(i)
}
/** Out of the clips into the clip's episode: on from here, or the whole episode from 0:00. */
function listenToEpisode(fromStart: boolean): void {
  const slug = current.value?.slug
  if (!slug) return
  report(fromStart ? 'keep_start' : 'keep')
  if (fromStart) player.seek(0)
  player.exitReel(true)
  void router.push({ name: 'player', params: { slug } })
}
function goBack(): void {
  if (window.history.length > 1) router.back()
  else void router.push({ name: 'browse' })
}
</script>

<template>
  <section class="lp-page pb-8 pt-4" data-testid="topic-reel-view">
    <button type="button" class="lp-nav" :aria-label="t('nav.back')" @click="goBack">
      <BackIcon />
      <span>{{ t('nav.back') }}</span>
    </button>

    <p v-if="failed" class="mt-6 text-sm text-muted" data-testid="topic-reel-unavailable">{{ t('reel.unavailable') }}</p>
    <template v-else-if="data">
      <header class="mt-3">
        <span class="lp-kicker text-listen">{{ t('reel.kicker') }} · {{ t(`ec.${data.subject_kind}`) }}</span>
        <h1 class="font-display text-3xl font-extrabold tracking-tight" data-testid="topic-reel-title">
          {{ data.subject_label }}
        </h1>
        <p class="mt-1 text-sm text-muted" data-testid="topic-reel-length">
          {{ formatTime(data.total_seconds) }} · {{ t('reel.clips', { n: data.clips.length }, data.clips.length) }} ·
          {{ t('reel.episodes', { n: episodeCount }, episodeCount) }}
        </p>
      </header>

      <p v-if="!data.clips.length" class="mt-6 text-sm text-muted" data-testid="topic-reel-empty">
        {{ t('reel.empty', { topic: data.subject_label }) }}
      </p>

      <template v-else>
        <div class="mt-5" data-testid="topic-reel-progress">
          <div
            class="h-1 overflow-hidden rounded-full bg-overlay"
            role="progressbar"
            :aria-valuenow="Math.round(heardPct)"
            aria-valuemin="0"
            aria-valuemax="100"
            :aria-label="t('reel.elapsed', { at: formatTime(heardMs / 1000), total: formatTime(totalMs / 1000) })"
          >
            <div class="h-full rounded-full bg-listen transition-[width] duration-300" :style="{ width: `${heardPct}%` }" />
          </div>
          <p class="mt-1 flex justify-between font-mono text-[11px] tabular-nums text-muted">
            <span>{{ formatTime(heardMs / 1000) }}</span><span>{{ formatTime(totalMs / 1000) }}</span>
          </p>
        </div>

        <section v-if="live?.done" class="mt-6" data-testid="topic-reel-done">
          <h2 class="lp-section mb-3">{{ t('reel.done', { topic: data.subject_label }) }}</h2>
          <ul class="grid gap-2">
            <li v-for="m in episodesHeard" :key="m.slug">
              <RouterLink :to="{ name: 'player', params: { slug: m.slug } }" class="lp-tap block rounded-xl border border-border p-3" data-testid="topic-reel-episode">
                <span class="line-clamp-2 text-sm font-bold">{{ m.episodeTitle }}</span>
                <span class="block break-words text-xs text-muted">{{ m.showTitle }}</span>
              </RouterLink>
            </li>
          </ul>
          <button type="button" class="lp-tap mt-4 font-bold text-accent" data-testid="topic-reel-again" @click="start()">{{ t('reel.again') }}</button>
        </section>

        <template v-else-if="shown">
          <article class="mt-4 rounded-2xl border border-border p-5" data-testid="topic-reel-current">
            <p class="font-mono text-xs uppercase tracking-wide text-muted">
              {{ t('reel.ofCount', { n: shownIndex + 1, total: moments.length }) }}
            </p>
            <div class="mt-3 flex min-w-0 items-center gap-3">
              <img v-if="shown.artwork" :src="shown.artwork" alt="" class="h-12 w-12 shrink-0 rounded-md object-cover" />
              <span class="min-w-0">
                <span class="line-clamp-2 text-sm font-bold">{{ shown.episodeTitle }}</span>
                <span class="block break-words text-xs text-muted">{{ shown.showTitle }}</span>
              </span>
            </div>
            <blockquote class="mt-4 text-lg leading-snug" data-testid="topic-reel-quote">“{{ shown.text }}”</blockquote>
            <p v-if="shown.speaker" class="mt-1 text-sm text-muted">— {{ shown.speaker }}</p>
            <!-- Centred in the card (operator 2026-10-11, on the phone): the transport is the card's
                 control, and "Listen to the full episode" sits centred under it. -->
            <div class="mt-5 flex items-center justify-center gap-3" data-testid="topic-reel-transport">
              <button type="button" class="lp-tap h-11 w-11 rounded-full border border-border" :disabled="!live" :aria-label="t('reel.prev')" data-testid="topic-reel-prev" @click="prev">‹</button>
              <button
                type="button"
                class="lp-tap inline-flex h-12 w-12 items-center justify-center rounded-full bg-listen text-lg text-canvas"
                :aria-label="playingNow ? t('reel.pause') : t('reel.play')"
                :data-testid="playingNow ? 'topic-reel-pause' : 'topic-reel-play'"
                @click="togglePlay"
              >{{ playingNow ? '❚❚' : '▶' }}</button>
              <button type="button" class="lp-tap h-11 w-11 rounded-full border border-border" :disabled="!live" :aria-label="t('reel.next')" data-testid="topic-reel-next" @click="next">›</button>
            </div>
            <EpisodeListenChoice
              v-if="live"
              class="mt-3"
              centered
              :time="player.currentTime"
              testid="topic-reel-listen"
              keep-testid="topic-reel-keep"
              start-testid="topic-reel-play-start"
              @play-on="listenToEpisode(false)"
              @from-start="listenToEpisode(true)"
            />
          </article>

          <section v-if="upNext.length" class="mt-6">
            <h2 class="lp-section mb-2">{{ t('reel.upNext') }}</h2>
            <ol class="divide-y divide-border">
              <li v-for="{ m, i } in upNext" :key="`${m.slug}:${m.insightId}`">
                <button
                  type="button"
                  class="lp-tap flex w-full min-w-0 items-center gap-3 py-2.5 text-left"
                  :aria-label="t('reel.jump', { n: i + 1, speaker: m.speaker ?? '', episode: m.episodeTitle ?? '' })"
                  data-testid="topic-reel-up-next"
                  @click="jump(i)"
                >
                  <span class="w-6 shrink-0 font-mono text-xs tabular-nums text-muted">{{ i + 1 }}</span>
                  <!-- The clip's episode artwork (operator 2026-10-15): the clips come from different
                       episodes, and the cover says which at a glance. -->
                  <img
                    v-if="artworkThumb(m.artwork)"
                    :src="artworkThumb(m.artwork)!"
                    alt=""
                    loading="lazy"
                    class="h-9 w-9 shrink-0 rounded bg-overlay object-cover"
                    data-testid="topic-reel-up-next-art"
                  />
                  <span v-else class="h-9 w-9 shrink-0 rounded bg-overlay" aria-hidden="true" />
                  <span class="min-w-0 text-sm">
                    <span class="font-semibold">{{ m.speaker }}</span>
                    <span v-if="m.speaker" class="text-muted"> · </span>
                    <span class="text-muted">{{ m.episodeTitle }}</span>
                  </span>
                </button>
              </li>
            </ol>
          </section>
        </template>
      </template>
    </template>
  </section>
</template>
