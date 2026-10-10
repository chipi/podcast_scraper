<script setup lang="ts">
/**
 * The Moments view (operator 2026-10-10): an episode's strongest moments, played back to back.
 *
 * It replaces the player page's left column while a reel runs, and it is deliberately NOT the
 * episode view with fewer controls: the operator could not tell the two apart when Moments was a
 * band on the artwork. So it has its own title ("Moments", amber), the artwork shrinks to a header
 * thumbnail, the current moment is the main thing on screen (the point in display type, who says
 * it, a thin bar for the clip), and an index lists every moment — played ones ticked, the current
 * one marked, any one a tap away.
 *
 * Presentational: the player store owns the reel (`startReel`, `reelGo`, `exitReel`); this emits.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { ReelMoment } from '../stores/player'
import { formatTime } from '../player/transcriptSync'
import CloseIcon from './CloseIcon.vue'
import { minutesValue, reelSeconds } from '../services/moments'

const props = defineProps<{
  title: string
  showTitle: string | null
  artwork: string | null
  moments: ReelMoment[]
  index: number
  done: boolean
  playing: boolean
  /** Seconds on the episode's audio timeline. */
  currentTime: number
  /** Where ✕ returns to, in seconds. */
  returnTo: number
  /** The episode's length in seconds, for the end card. */
  episodeSeconds: number
  /** The next queued episode's title: the end card offers its moments ("speed checking" a queue). */
  nextTitle?: string | null
  /** The episode's other two doors stay reachable from the reel (the obi's Brief and About). */
  hasBrief?: boolean
  hasAbout?: boolean
  /** Speaker name → their role in this episode ("host" / "guest"): "Nora, host". */
  speakerRoles?: Record<string, string>
}>()

const emit = defineEmits<{
  (e: 'go', index: number): void
  (e: 'prev'): void
  (e: 'next'): void
  (e: 'toggle'): void
  (e: 'keep'): void
  (e: 'close'): void
  (e: 'from-start'): void
  (e: 'next-episode'): void
  (e: 'brief'): void
  (e: 'about'): void
}>()

const { t } = useI18n()

const current = computed(() => props.moments[props.index] ?? null)
const totalSeconds = computed(() => reelSeconds(props.moments))
const clipProgress = computed(() => {
  const m = current.value
  if (!m || m.endMs <= m.startMs) return 0
  const at = props.currentTime * 1000
  return Math.max(0, Math.min(1, (at - m.startMs) / (m.endMs - m.startMs)))
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
  <section class="flex flex-col" data-testid="moments-reel" :aria-label="t('moments.title')">
    <div class="flex items-center justify-between gap-3">
      <h1 class="font-display text-2xl font-extrabold text-accent">{{ t('moments.title') }}</h1>
      <button
        type="button"
        class="lp-nav shrink-0"
        data-testid="moments-close"
        :aria-label="t('moments.close')"
        @click="emit('close')"
      >
        <CloseIcon />
      </button>
    </div>

    <div class="mt-2 grid grid-cols-[40px_minmax(0,1fr)] items-center gap-3">
      <img
        v-if="artwork"
        :src="artwork"
        alt=""
        class="h-10 w-10 rounded border border-border object-cover"
      />
      <span v-else class="h-10 w-10 rounded border border-border bg-elevated" />
      <div class="min-w-0">
        <p v-if="showTitle" class="lp-kicker lp-show-name">{{ showTitle }}</p>
        <p class="text-sm font-bold leading-snug" data-testid="moments-episode">
          {{ title }}
          <span class="font-normal text-muted">
            · {{ t('moments.subtitle', { count: moments.length, length: minutes(totalSeconds) }) }}
          </span>
        </p>
      </div>
    </div>

    <!-- The obi's three equal doors, as in the episode view, with Moments turned into Episode: you
         leave where you came in (operator 2026-10-10). Episode = ✕, back to where you were. -->
    <nav class="mt-3 grid grid-flow-col auto-cols-fr border-y border-border" :aria-label="t('player.obiLabel')" data-testid="moments-doors">
      <button type="button" class="lp-moments-door" data-testid="moments-door-episode" @click="emit('close')">
        <span class="lp-kicker">{{ t('moments.door_episode') }}</span>
      </button>
      <button v-if="hasBrief" type="button" class="lp-moments-door" data-testid="moments-door-brief" @click="emit('brief')">
        <span class="lp-kicker">{{ t('kp.title') }}</span>
      </button>
      <button v-if="hasAbout" type="button" class="lp-moments-door" data-testid="moments-door-about" @click="emit('about')">
        <span class="lp-kicker">{{ t('player.about') }}</span>
      </button>
    </nav>

    <ol class="mt-3 flex gap-1" aria-hidden="true">
      <li
        v-for="(m, i) in moments"
        :key="m.insightId"
        class="h-[3px] flex-1 rounded-sm"
        :class="
          segState(i) === 'done'
            ? 'bg-canvas-foreground'
            : segState(i) === 'now'
              ? 'bg-accent'
              : 'bg-overlay'
        "
        :data-state="segState(i)"
        data-testid="moments-segment"
      />
    </ol>

    <!-- The end card replaces the moment once the last one has played. -->
    <div
      v-if="done"
      class="mt-3 rounded border border-border bg-elevated p-4"
      data-testid="moments-done"
    >
      <p class="lp-kicker">{{ t('moments.doneKicker') }}</p>
      <p class="mt-2 font-display text-lg font-bold leading-snug">
        {{
          t('moments.doneTitle', {
            count: moments.length,
            length: minutes(totalSeconds),
            episode: minutes(episodeSeconds),
          })
        }}
      </p>
      <div class="mt-3 flex flex-col gap-2">
        <button
          type="button"
          class="h-11 rounded border border-accent px-3 text-sm font-bold text-accent"
          data-testid="moments-from-start"
          @click="emit('from-start')"
        >
          {{ t('moments.fromStart') }}
        </button>
        <button
          type="button"
          class="h-11 rounded border border-border px-3 text-sm font-bold"
          data-testid="moments-back"
          @click="emit('close')"
        >
          {{ t('moments.backTo', { time: formatTime(returnTo) }) }}
        </button>
        <!-- Two lines: what it does first, so a long title can never truncate "its moments" away. -->
        <button
          v-if="nextTitle"
          type="button"
          class="min-h-11 rounded border border-border px-3 py-2 text-left"
          data-testid="moments-next-episode"
          @click="emit('next-episode')"
        >
          <span class="block text-sm font-bold">{{ t('moments.nextEpisode') }}</span>
          <span class="block truncate text-xs text-muted">{{ nextTitle }}</span>
        </button>
      </div>
    </div>

    <div
      v-else-if="current"
      class="mt-3 rounded border border-border bg-elevated p-4"
      data-testid="moments-current"
    >
      <p class="lp-kicker">
        {{ t('moments.ofCount', { n: index + 1, total: moments.length }) }} ·
        {{ formatTime(current.startMs / 1000) }}
      </p>
      <p class="mt-2 font-display text-lg font-bold leading-snug" data-testid="moments-text">
        {{ current.text }}
      </p>
      <p v-if="current.speaker" class="mt-2 text-sm text-muted" data-testid="moments-speaker">
        {{ current.speaker }}<template v-if="speakerRoles?.[current.speaker]">, {{ speakerRoles[current.speaker] }}</template>
      </p>
      <div class="mt-3 h-[3px] rounded-sm bg-overlay" aria-hidden="true">
        <div class="h-full rounded-sm bg-accent" :style="{ width: `${clipProgress * 100}%` }" />
      </div>
    </div>

    <div class="mt-3 flex items-center justify-center gap-5">
      <button
        type="button"
        class="flex h-11 w-11 items-center justify-center rounded-full border border-border"
        data-testid="moments-prev"
        :aria-label="t('moments.prev')"
        @click="emit('prev')"
      >
        <span aria-hidden="true">|◀</span>
      </button>
      <button
        type="button"
        class="flex h-14 w-14 items-center justify-center rounded-full bg-accent text-accent-foreground"
        data-testid="moments-toggle"
        :aria-label="playing ? t('moments.pause') : t('moments.play')"
        @click="emit('toggle')"
      >
        <span aria-hidden="true">{{ playing ? '❚❚' : '▶' }}</span>
      </button>
      <button
        type="button"
        class="flex h-11 w-11 items-center justify-center rounded-full border border-border"
        data-testid="moments-next"
        :aria-label="t('moments.next')"
        @click="emit('next')"
      >
        <span aria-hidden="true">▶|</span>
      </button>
    </div>

    <button
      v-if="current && !done"
      type="button"
      class="mt-3 h-11 rounded border border-border px-3 text-sm font-bold"
      data-testid="moments-keep"
      @click="emit('keep')"
    >
      {{ t('moments.keepHere', { time: formatTime(currentTime) }) }}
    </button>

    <h2 class="sr-only">{{ t('moments.index') }}</h2>
    <ol class="mt-3 border-t border-border" data-testid="moments-index">
      <li v-for="(m, i) in moments" :key="m.insightId">
        <button
          type="button"
          class="grid w-full grid-cols-[1rem_2.75rem_minmax(0,1fr)] items-baseline gap-2 border-b border-border py-2.5 text-left text-sm"
          :class="segState(i) === 'now' && !done ? 'text-canvas-foreground' : 'text-muted'"
          :aria-current="segState(i) === 'now' && !done ? 'true' : undefined"
          data-testid="moments-index-item"
          @click="emit('go', i)"
        >
          <span class="font-mono text-xs" :class="segState(i) === 'now' && !done ? 'text-accent' : ''">
            {{ segState(i) === 'done' ? '✓' : segState(i) === 'now' ? '▶' : i + 1 }}
          </span>
          <span class="font-mono text-xs tabular-nums">{{ formatTime(m.startMs / 1000) }}</span>
          <span class="line-clamp-2">{{ m.text }}</span>
        </button>
      </li>
    </ol>
  </section>
</template>

<style scoped>
.lp-moments-door {
  display: flex;
  min-height: 44px;
  align-items: center;
  justify-content: center;
  transition: background-color 150ms;
}
.lp-moments-door + .lp-moments-door {
  border-left: 1px solid var(--lp-border);
}
.lp-moments-door:hover {
  background-color: var(--lp-overlay);
}
.lp-moments-door:focus-visible {
  outline: 2px solid var(--lp-accent);
  outline-offset: -2px;
}
</style>
