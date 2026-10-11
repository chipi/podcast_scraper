<script setup lang="ts">
/**
 * One day of a "Learn X this week" path: the episode, its clip on the subject (what is said, by
 * whom), "▶ Play from mm:ss" — the episode opened playing at the clip — and "Done". A day not yet
 * come says when it opens and offers nothing to do.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { formatTime } from '../player/transcriptSync'
import type { PathStep } from '../services/api'
import { track } from '../services/analytics'

const props = defineProps<{ step: PathStep; busy?: boolean; kind?: 'topic' | 'theme' | 'storyline' }>()
const emit = defineEmits<{ done: [] }>()
const { t } = useI18n()
const time = computed(() => formatTime(props.step.start_ms / 1000))
const playTo = computed(() => ({
  name: 'player',
  params: { slug: props.step.episode.slug },
  query: { t: String(Math.floor(props.step.start_ms / 1000)), play: '1' },
}))
</script>

<template>
  <!-- The whole card opens the episode, paused (operator 2026-10-11): the title's link stretches over
       it; "Play from" and "Done" sit above it. -->
  <article
    class="relative rounded-xl border border-border p-4"
    :class="step.unlocked ? '' : 'opacity-60'"
    :data-day="step.day"
    data-testid="learn-step"
  >
    <p class="font-mono text-[11px] font-bold uppercase tracking-[0.12em] text-muted">
      {{ t('learn.dayN', { n: step.day }) }}
      <span v-if="step.done" class="ml-2 text-accent" data-testid="learn-step-done-mark">✓ {{ t('learn.done') }}</span>
    </p>
    <RouterLink
      :to="{ name: 'player', params: { slug: step.episode.slug } }"
      class="mt-1 block font-semibold text-canvas-foreground no-underline after:absolute after:inset-0"
      data-testid="learn-step-episode"
    >{{ step.episode.title }}</RouterLink>
    <p class="line-clamp-2 text-xs text-muted">{{ step.episode.podcast_title }}</p>
    <template v-if="step.unlocked">
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
      </div>
    </template>
    <p v-else class="mt-2 text-xs text-muted" data-testid="learn-step-locked">{{ t('learn.opensOn', { n: step.day }) }}</p>
  </article>
</template>
