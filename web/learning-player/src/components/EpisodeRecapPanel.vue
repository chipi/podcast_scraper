<script setup lang="ts">
/**
 * Post-episode recap panel (RFC-122 / #2038) — the reinforcement surface.
 *
 * When an episode finishes (the player store's `justFinished` fires), this replaces the transport
 * IN PLACE on the episode page: what the listener just heard, consolidated. It renders one recap
 * model — summary key points + the single strongest attributed quote + top insights + key topics +
 * storylines — and is kept SHORT enough to sit on a phone without an internal scroll (discovery is
 * the related-episodes rail already on the page, not repeated here). The same model will feed the
 * daily digest email (#2039), so the two surfaces cannot drift.
 *
 * ## The queue end-card (#2038)
 *
 * When there IS a next episode queued, the panel is also an end-card: it counts down and, on zero
 * (or "Play next"), emits `advance` so the shell continues the queue — reinforcement AND the
 * queue's purpose. "Stay" cancels the countdown and keeps the finished player. With nothing queued
 * there is no countdown — the countdown footer collapses to "Back to player".
 *
 * Bridge-only (PRD-035 Principle 4): everything here is transcript-derived text + KG metadata +
 * artwork; the panel never touches audio.
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import type { EpisodeRecap } from '../services/types'

const props = withDefaults(
  defineProps<{
    recap: EpisodeRecap
    /** Countdown length when a next episode is queued; null/omitted = no end-card. */
    autoAdvanceSeconds?: number | null
    /** Title of the queued next episode, for the countdown line (best-effort). */
    nextTitle?: string | null
  }>(),
  { autoAdvanceSeconds: null, nextTitle: null },
)
const emit = defineEmits<{ (e: 'dismiss'): void; (e: 'advance'): void; (e: 'stay'): void }>()

const { t } = useI18n()

// Bullets are the gist; when an episode has none, the prose summary is the fallback lede (one item)
// so the panel always leads with something to consolidate rather than an empty heading. Capped to
// keep the end-card short enough to fit a phone without the whole page becoming a scroll.
const KEY_POINT_LIMIT = 3
const keyPoints = computed<string[]>(() => {
  if (props.recap.key_points.length) return props.recap.key_points.slice(0, KEY_POINT_LIMIT)
  const prose = props.recap.summary_text?.trim()
  return prose ? [prose] : []
})
const quote = computed(() => props.recap.signature_quote)
const insights = computed(() => props.recap.insights)
const topics = computed(() => props.recap.topics ?? [])
const storylines = computed(() => props.recap.storylines ?? [])

// --- end-card countdown (self-contained; the shell owns what `advance` DOES) ------------------
const remaining = ref<number | null>(props.autoAdvanceSeconds)
// The starting length, captured once, so the progress bar has a denominator that never moves.
const totalSeconds = props.autoAdvanceSeconds ?? 0
const progressPct = computed(() =>
  remaining.value !== null && totalSeconds > 0
    ? Math.max(0, Math.min(100, (remaining.value / totalSeconds) * 100))
    : 0,
)
let timer: ReturnType<typeof setInterval> | null = null
function stopTimer(): void {
  if (timer !== null) {
    clearInterval(timer)
    timer = null
  }
}
onMounted(() => {
  if (remaining.value === null || remaining.value <= 0) return
  timer = setInterval(() => {
    if (remaining.value === null) return
    remaining.value -= 1
    if (remaining.value <= 0) {
      stopTimer()
      emit('advance')
    }
  }, 1000)
})
onBeforeUnmount(stopTimer)

function playNow(): void {
  stopTimer()
  emit('advance')
}
function stay(): void {
  stopTimer()
  remaining.value = null // collapse the countdown back to a plain "Back to player" footer
  emit('stay')
}
</script>

<template>
  <section
    class="overflow-hidden rounded-2xl border border-border bg-surface"
    :aria-label="t('player.recapRegion')"
    data-testid="episode-recap-panel"
  >
    <!-- Header: what just happened + the "we took notes" reassurance, with a dismiss back to the
         finished player pinned to the right. -->
    <header class="flex items-start justify-between gap-3 border-b border-border p-3">
      <div class="min-w-0">
        <p class="lp-kicker text-accent">{{ t('player.recapKicker') }}</p>
        <h2 class="mt-0.5 truncate font-display text-lg font-bold text-canvas-foreground">
          {{ recap.title }}
        </h2>
        <p class="mt-1 flex items-center gap-1.5 text-sm text-muted">
          <svg
            viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
            stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4 shrink-0" aria-hidden="true"
          >
            <path d="M4 4h11l5 5v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1Z" />
            <path d="M14 4v5h5M8 13h8M8 17h5" />
          </svg>
          {{ t('player.recapNotes') }}
        </p>
      </div>
      <button
        type="button"
        class="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-border text-muted transition hover:bg-overlay hover:text-canvas-foreground"
        :aria-label="t('player.recapDismiss')"
        :title="t('player.recapDismiss')"
        data-testid="recap-dismiss"
        @click="emit('dismiss')"
      >
        <svg
          viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
          stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" aria-hidden="true"
        >
          <path d="M18 6 6 18M6 6l12 12" />
        </svg>
      </button>
    </header>

    <div class="space-y-4 p-3">
      <!-- Key points — the gist to consolidate. -->
      <section v-if="keyPoints.length" data-testid="recap-key-points">
        <h3 class="lp-kicker mb-1.5 text-muted">{{ t('player.recapKeyPoints') }}</h3>
        <ul class="space-y-1.5">
          <li
            v-for="(point, i) in keyPoints"
            :key="i"
            class="flex gap-2 text-sm leading-relaxed text-canvas-foreground"
          >
            <span class="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" aria-hidden="true" />
            <span>{{ point }}</span>
          </li>
        </ul>
      </section>

      <!-- The anchor: one memorable line, attributed when the graph can name the speaker. -->
      <blockquote
        v-if="quote"
        class="border-l-2 border-accent pl-3"
        data-testid="recap-quote"
      >
        <p class="font-display text-base italic leading-snug text-canvas-foreground">
          “{{ quote.text }}”
        </p>
        <footer v-if="quote.speaker" class="mt-1 text-sm text-muted">
          {{ t('player.recapQuoteBy', { speaker: quote.speaker }) }}
        </footer>
      </blockquote>

      <!-- Top insights (salience-ranked, capped server-side). -->
      <section v-if="insights.length" data-testid="recap-insights">
        <h3 class="lp-kicker mb-1.5 text-muted">{{ t('player.recapInsights') }}</h3>
        <ul class="space-y-2">
          <li
            v-for="ins in insights"
            :key="ins.id"
            class="text-sm leading-relaxed text-canvas-foreground"
          >
            {{ ins.text }}
          </li>
        </ul>
      </section>

      <!-- Key topics — chips into the topic card. -->
      <section v-if="topics.length" data-testid="recap-topics">
        <h3 class="lp-kicker mb-1.5 text-muted">{{ t('player.recapTopics') }}</h3>
        <ul class="flex flex-wrap gap-2">
          <li v-for="tp in topics" :key="tp.id">
            <RouterLink
              :to="{ name: 'topic', params: { id: tp.id } }"
              class="inline-block rounded-full border border-border px-3 py-1 text-sm text-canvas-foreground no-underline transition hover:bg-overlay"
            >
              {{ tp.label }}
            </RouterLink>
          </li>
        </ul>
      </section>

      <!-- Storylines — the theme threads this episode belongs to. -->
      <section v-if="storylines.length" data-testid="recap-storylines">
        <h3 class="lp-kicker mb-1.5 text-muted">{{ t('player.recapStorylines') }}</h3>
        <ul class="space-y-1.5">
          <li v-for="s in storylines" :key="s.id">
            <RouterLink
              :to="{ name: 'storyline', params: { id: s.id } }"
              class="inline-flex items-center gap-1.5 text-sm font-semibold text-accent no-underline"
            >
              <svg
                viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4 shrink-0" aria-hidden="true"
              >
                <path d="m9 6 6 6-6 6" />
              </svg>
              {{ s.label }}
            </RouterLink>
          </li>
        </ul>
      </section>

    </div>

    <!-- Footer: the end-card countdown when a next episode is queued, else a plain dismiss. The
         countdown gets its own full-width line (so the next title is readable, not truncated to a
         couple of letters) + a depleting progress bar, with the actions on the row below. -->
    <footer class="border-t border-border p-3">
      <template v-if="remaining !== null">
        <p class="mb-2 truncate text-sm text-muted" data-testid="recap-countdown">
          <span class="text-canvas-foreground">{{ t('player.recapNextIn', { n: remaining }) }}</span>
          <span v-if="nextTitle"> · {{ nextTitle }}</span>
        </p>
        <!-- Visual countdown: the bar depletes over the wait; `ease-linear` across the 1s tick
             makes it glide rather than step. -->
        <div class="mb-3 h-1 overflow-hidden rounded-full bg-border" aria-hidden="true">
          <div
            class="h-full rounded-full bg-accent transition-[width] duration-1000 ease-linear"
            :style="{ width: `${progressPct}%` }"
            data-testid="recap-progress"
          />
        </div>
        <div class="flex items-center gap-2">
          <button
            type="button"
            class="shrink-0 whitespace-nowrap rounded-full border border-border px-3 py-2 text-sm text-canvas-foreground transition hover:bg-overlay"
            data-testid="recap-stay"
            @click="stay"
          >
            {{ t('player.recapStay') }}
          </button>
          <button
            type="button"
            class="ml-auto shrink-0 whitespace-nowrap rounded-full bg-accent px-4 py-2 text-sm font-semibold text-accent-foreground transition hover:opacity-90"
            data-testid="recap-play-next"
            @click="playNow"
          >
            {{ t('player.recapPlayNext') }}
          </button>
        </div>
      </template>
      <div v-else class="flex justify-end">
        <button
          type="button"
          class="rounded-full bg-accent px-4 py-2 text-sm font-semibold text-accent-foreground transition hover:opacity-90"
          data-testid="recap-back"
          @click="emit('dismiss')"
        >
          {{ t('player.recapDismiss') }}
        </button>
      </div>
    </footer>
  </section>
</template>
