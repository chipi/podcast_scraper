<script setup lang="ts">
/**
 * Moments, inside the artwork (operator 2026-10-10: "the key area stays as on the played episode
 * and we just change inside the artwork").
 *
 * The page stays the episode page — masthead, artwork, obi, transport. While a reel plays, the
 * artwork shows it: segments across the top (how far through the reel), and at the bottom, where
 * the live insight card sits otherwise, the current moment — which of how many and when, the point,
 * who says it and their role, a thin bar for the clip, ‹ › to the moment before or after, and
 * **Listen to the full episode** (play on from here, or from the start). After the last moment the
 * same place shows the end card.
 *
 * Positioned by its host: a fragment of two absolutely placed blocks inside the artwork box.
 * Presentational: the player store owns the reel; this emits.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { ReelMoment } from '../stores/player'
import { formatTime } from '../player/transcriptSync'
import EpisodeListenChoice from './EpisodeListenChoice.vue'
import { minutesValue, reelSeconds } from '../services/moments'
import TranslatedFrom from './TranslatedFrom.vue'

const props = defineProps<{
  moments: ReelMoment[]
  index: number
  done: boolean
  /** Seconds on the episode's audio timeline. */
  currentTime: number
  /** Where "Back to …" returns to, in seconds. */
  returnTo: number
  /** The episode's length in seconds, for the end card. */
  episodeSeconds: number
  /** The next queued episode's title: the end card offers its moments. */
  nextTitle?: string | null
  /** Speaker name → their role in this episode ("host" / "guest"): "Nora, host". */
  speakerRoles?: Record<string, string>
  /** A learning day's clip (operator 2026-10-14): its 1-based day; the card speaks as that day. */
  learnDay?: number
  /** What became of the day once its clip ended (`LearnEnd.status`). */
  learnStatus?: 'pending' | 'done' | 'not_heard' | 'failed'
}>()

const emit = defineEmits<{
  (e: 'prev'): void
  (e: 'next'): void
  (e: 'keep'): void
  (e: 'back'): void
  (e: 'from-start'): void
  (e: 'next-episode'): void
  (e: 'learn-back'): void
  (e: 'learn-mark-done'): void
}>()

const { t } = useI18n()

const current = computed(() => props.moments[props.index] ?? null)
const clipProgress = computed(() => {
  const m = current.value
  if (!m || m.endMs <= m.startMs) return 0
  return Math.max(0, Math.min(1, (props.currentTime * 1000 - m.startMs) / (m.endMs - m.startMs)))
})
function minutes(seconds: number): string {
  return t('moments.minutes', { n: minutesValue(seconds) })
}
function segState(i: number): 'done' | 'now' | 'todo' {
  if (props.done || i < props.index) return 'done'
  return i === props.index ? 'now' : 'todo'
}
</script>

<template>
  <!-- How far through the reel, across the top of the artwork. -->
  <ol class="absolute inset-x-0 top-0 z-10 flex gap-1 p-3" aria-hidden="true" data-testid="moments-segments">
    <li
      v-for="(m, i) in moments"
      :key="m.insightId"
      class="h-[3px] flex-1 rounded-sm"
      :class="
        segState(i) === 'done'
          ? 'bg-canvas-foreground'
          : segState(i) === 'now'
            ? 'bg-accent'
            : 'bg-canvas-foreground/30'
      "
      :data-state="segState(i)"
      data-testid="moments-segment"
    />
  </ol>

  <div class="absolute inset-x-0 bottom-0 z-10" data-testid="moments-card">
    <div class="relative h-16"><div class="moments-scrim absolute inset-0" /></div>
    <div class="moments-body px-4 pb-4 pt-1">
      <!-- A learning day's end (operator 2026-10-14): the day done, back to the week, or the rest of
           the episode. Not done by the clip (skipped through, or the save failed): the listener's own. -->
      <div v-if="done && learnDay" data-testid="learn-clip-done" :data-status="learnStatus ?? 'pending'">
        <p class="lp-kicker" :class="learnStatus === 'done' ? 'text-learn' : ''" data-testid="learn-clip-kicker">
          {{ learnStatus === 'done' ? t('learn.clipDone', { n: learnDay }) : t('learn.clipEnd', { n: learnDay }) }}
        </p>
        <p class="mt-1 font-display text-base font-bold leading-snug text-canvas-foreground">
          {{
            learnStatus === 'done'
              ? t('learn.clipDoneTitle')
              : learnStatus === 'not_heard'
                ? t('learn.clipNotHeard')
                : learnStatus === 'failed'
                  ? t('learn.clipFailed')
                  : t('learn.clipSaving')
          }}
        </p>
        <div class="mt-2 flex flex-wrap gap-x-4 gap-y-1">
          <button
            v-if="learnStatus === 'not_heard' || learnStatus === 'failed'"
            type="button"
            class="lp-moments-action text-learn"
            data-testid="learn-clip-mark-done"
            @click="emit('learn-mark-done')"
          >
            {{ t('learn.clipMarkDone', { n: learnDay }) }}
          </button>
          <button type="button" class="lp-moments-action text-accent" data-testid="learn-clip-back" @click="emit('learn-back')">
            {{ t('learn.backToLearn') }}
          </button>
        </div>
        <EpisodeListenChoice
          class="mt-1"
          :time="currentTime"
          testid="learn-clip-listen"
          keep-testid="learn-clip-keep"
          start-testid="learn-clip-play-start"
          @play-on="emit('keep')"
          @from-start="emit('from-start')"
        />
      </div>
      <template v-else-if="done">
        <div data-testid="moments-done">
          <p class="lp-kicker">{{ t('moments.doneKicker') }}</p>
          <p class="mt-1 font-display text-base font-bold leading-snug text-canvas-foreground">
            {{
              t('moments.doneTitle', {
                count: moments.length,
                length: minutes(reelSeconds(moments)),
                episode: minutes(episodeSeconds),
              })
            }}
          </p>
          <div class="mt-2 flex flex-wrap gap-x-4 gap-y-1">
            <button type="button" class="lp-moments-action text-accent" data-testid="moments-from-start" @click="emit('from-start')">
              {{ t('moments.fromStart') }}
            </button>
            <button type="button" class="lp-moments-action" data-testid="moments-back" @click="emit('back')">
              {{ t('moments.backTo', { time: formatTime(returnTo) }) }}
            </button>
          </div>
          <button
            v-if="nextTitle"
            type="button"
            class="mt-1 block min-h-11 w-full text-left"
            data-testid="moments-next-episode"
            @click="emit('next-episode')"
          >
            <span class="block text-sm font-bold text-canvas-foreground">{{ t('moments.nextEpisode') }}</span>
            <span class="block truncate text-xs text-muted">{{ nextTitle }}</span>
          </button>
        </div>
      </template>
      <div v-else-if="current" data-testid="moments-current">
        <p v-if="learnDay" class="lp-kicker" data-testid="learn-clip-kicker">
          {{ t('learn.clipKicker', { n: learnDay }) }} · {{ formatTime(current.startMs / 1000) }}
        </p>
        <div v-else class="flex items-center gap-2">
          <button
            type="button"
            class="lp-moments-step"
            data-testid="moments-prev"
            :aria-label="t('moments.prev')"
            @click="emit('prev')"
          >‹</button>
          <p class="lp-kicker min-w-0 flex-1 text-center">
            {{ t('moments.ofCount', { n: index + 1, total: moments.length }) }} ·
            {{ formatTime(current.startMs / 1000) }}
          </p>
          <button
            type="button"
            class="lp-moments-step"
            data-testid="moments-next"
            :aria-label="t('moments.next')"
            @click="emit('next')"
          >›</button>
        </div>
        <p
          class="mt-1.5 font-display text-base leading-snug text-canvas-foreground line-clamp-6"
          data-testid="moments-text"
        >
          {{ current.text }}
        </p>
        <p v-if="current.speaker" class="mt-1 text-xs text-muted" data-testid="moments-speaker">
          {{ current.speaker }}<template v-if="speakerRoles?.[current.speaker]">, {{ speakerRoles[current.speaker] }}</template>
        </p>
        <TranslatedFrom :lang="current.translatedFrom" />
        <div class="mt-2 h-[3px] rounded-sm bg-canvas-foreground/20" aria-hidden="true">
          <div class="h-full rounded-sm bg-accent" :style="{ width: `${clipProgress * 100}%` }" />
        </div>
        <EpisodeListenChoice
          class="mt-2"
          :time="currentTime"
          testid="moments-listen"
          keep-testid="moments-keep"
          start-testid="moments-play-start"
          @play-on="emit('keep')"
          @from-start="emit('from-start')"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
/* The live insight card's ramp (PlayerView's Zone D), so a moment reads as native to the artwork. */
.moments-body {
  background: linear-gradient(
    to top,
    color-mix(in srgb, var(--lp-canvas) 95%, transparent) 60%,
    color-mix(in srgb, var(--lp-canvas) 88%, transparent) 100%
  );
}
.moments-scrim {
  background: linear-gradient(
    to top,
    color-mix(in srgb, var(--lp-canvas) 95%, transparent) 0%,
    transparent 100%
  );
}
.lp-moments-step {
  display: inline-flex;
  flex: none;
  align-items: center;
  justify-content: center;
  width: 44px;
  height: 44px;
  margin: -8px;
  color: var(--lp-canvas-foreground);
  font-size: 1.25rem;
  line-height: 1;
}
.lp-moments-action {
  display: inline-flex;
  min-height: 44px;
  align-items: center;
  font-size: 0.875rem;
  font-weight: 700;
}
.lp-moments-step:focus-visible,
.lp-moments-action:focus-visible {
  outline: 2px solid var(--lp-accent);
  outline-offset: 2px;
}
</style>
