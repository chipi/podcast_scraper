<script setup lang="ts">
/**
 * N minutes on a topic, theme or storyline — goal-shaped listening (ADR-163 §8 amendment).
 *
 * The strongest clips ABOUT one subject (a theme or storyline: any of its member topics) from
 * different episodes, back to back (`GET /discover/reel`,
 * the Moments picker's clip windows). The player store plays them: each clip loads its own episode
 * (`startTopicReel` with the shell's `resolveNextUpForSlug`), and like any reel it never moves a
 * resume point or counts as a listen until "Keep listening to this episode" makes it one.
 *
 * - The header says what the reel actually runs ("6 clips · 1:12"), never the budget it was asked
 *   for: a small corpus gives less than fifteen minutes, and the page does not pretend otherwise.
 * - It starts on Play — a tap, so the browser lets the audio start.
 * - Leaving the page keeps it playing; the mini-player brings you back here.
 * - After the last clip, the end card lists the episodes heard, each a way to listen in full.
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import BackIcon from '../components/BackIcon.vue'
import { getTopicReel, type TopicReel } from '../services/api'
import { resolveNextUpForSlug } from '../services/nextUp'
import { formatTime } from '../player/transcriptSync'
import { usePlayerStore, type ReelMoment } from '../stores/player'
import { episodeArtwork } from '../utils/episode'

const props = defineProps<{ id: string; minutes?: number }>()
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
  }
})

const moments = computed<ReelMoment[]>(() =>
  (data.value?.clips ?? []).map((c) => ({
    insightId: c.insight_id,
    text: c.quote_text || c.text,
    speaker: c.speaker,
    startMs: c.start_ms,
    endMs: c.end_ms,
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

async function start(): Promise<void> {
  if (!data.value) return
  await player.startTopicReel(
    { id: props.id, label: data.value.subject_label, minutes: budget.value },
    moments.value,
    resolveNextUpForSlug,
  )
}
function keepListening(): void {
  const slug = current.value?.slug
  if (!slug) return
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
        <span class="lp-kicker text-topic">{{ t('reel.kicker') }}</span>
        <h1 class="font-display text-3xl font-extrabold tracking-tight" data-testid="topic-reel-title">
          {{ t('reel.title', { minutes: budget, topic: data.subject_label }) }}
        </h1>
        <p class="mt-1 text-sm text-muted" data-testid="topic-reel-length">
          {{ t('reel.length', { clips: data.clips.length, time: formatTime(data.total_seconds) }) }}
        </p>
      </header>

      <p v-if="!data.clips.length" class="mt-6 text-sm text-muted" data-testid="topic-reel-empty">
        {{ t('reel.empty', { topic: data.subject_label }) }}
      </p>

      <template v-else>
        <ol class="mt-5 flex gap-1" :aria-label="t('reel.progress')">
          <li
            v-for="(m, i) in moments"
            :key="`${m.slug}:${m.insightId}`"
            class="h-1 flex-1 rounded-full"
            :class="live && i <= live.index ? 'bg-accent' : 'bg-overlay'"
            data-testid="topic-reel-segment"
          />
        </ol>

        <button
          v-if="!live"
          type="button"
          class="lp-tap mt-6 inline-flex items-center gap-2 rounded-full bg-accent px-5 py-3 font-bold text-accent-foreground"
          data-testid="topic-reel-play"
          @click="start"
        >▶ {{ t('reel.play', { clips: data.clips.length }) }}</button>

        <article
          v-else-if="current && !live.done"
          class="mt-6 rounded-2xl border border-border p-5"
          data-testid="topic-reel-current"
        >
          <p class="font-mono text-xs uppercase tracking-wide text-muted">
            {{ t('reel.ofCount', { n: live.index + 1, total: live.moments.length }) }}
          </p>
          <div class="mt-3 flex min-w-0 items-center gap-3">
            <img v-if="current.artwork" :src="current.artwork" alt="" class="h-12 w-12 shrink-0 rounded-md object-cover" />
            <span class="min-w-0">
              <span class="line-clamp-2 text-sm font-bold">{{ current.episodeTitle }}</span>
              <span class="block break-words text-xs text-muted">{{ current.showTitle }}</span>
            </span>
          </div>
          <blockquote class="mt-4 text-lg leading-snug" data-testid="topic-reel-quote">“{{ current.text }}”</blockquote>
          <p v-if="current.speaker" class="mt-1 text-sm text-muted">— {{ current.speaker }}</p>
          <div class="mt-5 flex flex-wrap items-center gap-3">
            <button type="button" class="lp-tap h-11 w-11 rounded-full border border-border" :aria-label="t('reel.prev')" data-testid="topic-reel-prev" @click="player.reelPrev()">‹</button>
            <button type="button" class="lp-tap h-11 w-11 rounded-full border border-border" :aria-label="t('reel.next')" data-testid="topic-reel-next" @click="player.reelNext()">›</button>
            <button type="button" class="lp-tap font-bold text-accent" data-testid="topic-reel-keep" @click="keepListening">
              {{ t('reel.keep') }}
            </button>
          </div>
        </article>

        <section v-else-if="live?.done" class="mt-6" data-testid="topic-reel-done">
          <h2 class="lp-section mb-3">{{ t('reel.done', { topic: data.subject_label }) }}</h2>
          <ul class="grid gap-2">
            <li v-for="m in episodesHeard" :key="m.slug">
              <RouterLink :to="{ name: 'player', params: { slug: m.slug } }" class="lp-tap block rounded-xl border border-border p-3" data-testid="topic-reel-episode">
                <span class="line-clamp-2 text-sm font-bold">{{ m.episodeTitle }}</span>
                <span class="block break-words text-xs text-muted">{{ m.showTitle }}</span>
              </RouterLink>
            </li>
          </ul>
          <button type="button" class="lp-tap mt-4 font-bold text-accent" data-testid="topic-reel-again" @click="start">{{ t('reel.again') }}</button>
        </section>
      </template>
    </template>
  </section>
</template>
