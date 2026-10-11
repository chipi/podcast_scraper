<script setup lang="ts">
/**
 * One day of a "Learn X this week" path: the episode, its clip on the subject (what is said, by
 * whom), "▶ Play from mm:ss" — the episode opened playing at the clip — and "Done". A day not yet
 * come says when it opens and offers nothing to do.
 *
 * Done, with a later day left, it offers "Next day now" (operator 2026-10-11: a day a day is the
 * rhythm, not a wall).
 *
 * Only today's day is open (operator 2026-10-11: "focus on where we are"). An earlier day is a
 * closed card that says whether it was done; tapping it opens it, to replay that day, and its ⌃
 * closes it again. A later day stays closed.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { formatTime } from '../player/transcriptSync'
import type { PathStep } from '../services/api'
import { track } from '../services/analytics'

const props = defineProps<{
  step: PathStep
  busy?: boolean
  kind?: 'topic' | 'theme' | 'storyline'
  /** Show the clip, Play and Done (today's day, or an earlier one the listener opened). */
  open?: boolean
  /** Done, and the next day may be opened now: offer "Next day now". */
  canOpenNext?: boolean
}>()
const emit = defineEmits<{ done: []; toggle: []; next: [] }>()
const { t } = useI18n()
const time = computed(() => formatTime(props.step.start_ms / 1000))
const playTo = computed(() => ({
  name: 'player',
  params: { slug: props.step.episode.slug },
  query: { t: String(Math.floor(props.step.start_ms / 1000)), play: '1' },
}))
</script>

<template>
  <!-- Open: the whole card opens the episode, paused (operator 2026-10-11): the title's link stretches
       over it; "Play from", "Done" and ⌃ sit above it. Closed and already open to the listener (an
       earlier day): the whole card is the button that opens it. -->
  <article
    class="relative rounded-xl border border-border p-4"
    :class="step.unlocked ? '' : 'opacity-60'"
    :data-day="step.day"
    :data-open="open ? 'true' : 'false'"
    data-testid="learn-step"
  >
    <p class="flex items-center font-mono text-[11px] font-bold uppercase tracking-[0.12em] text-muted">
      <span>{{ t('learn.dayN', { n: step.day }) }}</span>
      <span v-if="step.done" class="ml-2 text-accent" data-testid="learn-step-done-mark">✓ {{ t('learn.done') }}</span>
      <span v-else-if="step.unlocked && !open" class="ml-2" data-testid="learn-step-not-done">· {{ t('learn.notDone') }}</span>
      <button
        v-if="step.unlocked && open"
        type="button"
        class="lp-tap relative z-30 -my-2 ml-auto inline-flex h-9 w-9 items-center justify-center rounded-full text-base"
        :aria-label="t('learn.collapse', { n: step.day })"
        aria-expanded="true"
        data-testid="learn-step-toggle"
        @click="emit('toggle')"
      >⌃</button>
    </p>
    <template v-if="open && step.unlocked">
      <RouterLink
        :to="{ name: 'player', params: { slug: step.episode.slug } }"
        class="mt-1 block font-semibold text-canvas-foreground no-underline after:absolute after:inset-0"
        data-testid="learn-step-episode"
      >{{ step.episode.title }}</RouterLink>
      <p class="line-clamp-2 text-xs text-muted">{{ step.episode.podcast_title }}</p>
      <blockquote class="mt-3 text-sm leading-snug text-canvas-foreground" data-testid="learn-step-quote">“{{ step.quote_text || step.text }}”</blockquote>
      <p v-if="step.speaker" class="mt-1 text-xs text-muted">{{ step.speaker }}</p>
      <div class="relative z-30 mt-3 flex flex-wrap items-center gap-x-4 gap-y-1">
        <RouterLink
          :to="playTo"
          class="lp-tap inline-flex min-h-[44px] items-center text-sm font-bold text-accent no-underline"
          data-testid="learn-step-play"
          @click="kind && track('learn_path', { action: 'step_play', kind, day: step.day })"
        >▶ {{ t('why.playFrom', { time }) }}</RouterLink>
        <button
          v-if="!step.done"
          type="button"
          class="lp-tap min-h-[44px] text-sm font-bold text-canvas-foreground"
          :disabled="busy"
          data-testid="learn-step-mark-done"
          @click="emit('done')"
        >{{ t('learn.markDone') }}</button>
        <button
          v-else-if="canOpenNext"
          type="button"
          class="lp-tap min-h-[44px] text-sm font-bold text-learn"
          :disabled="busy"
          data-testid="learn-step-next"
          @click="emit('next')"
        >{{ t('learn.openNext') }} ›</button>
      </div>
    </template>
    <template v-else>
      <button
        v-if="step.unlocked"
        type="button"
        class="mt-1 block w-full text-left font-semibold text-canvas-foreground after:absolute after:inset-0"
        :aria-label="t('learn.expand', { n: step.day, title: step.episode.title })"
        aria-expanded="false"
        data-testid="learn-step-open"
        @click="emit('toggle')"
      >{{ step.episode.title }}</button>
      <p v-else class="mt-1 font-semibold text-canvas-foreground" data-testid="learn-step-episode">{{ step.episode.title }}</p>
      <p class="line-clamp-1 text-xs text-muted">{{ step.episode.podcast_title }}</p>
      <p v-if="!step.unlocked" class="mt-2 text-xs text-muted" data-testid="learn-step-locked">{{ t('learn.opensOn', { n: step.day }) }}</p>
    </template>
  </article>
</template>
