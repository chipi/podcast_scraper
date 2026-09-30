<script setup lang="ts">
/**
 * Transport controls (PRD-039 FR2 / UXS-011). Presentational: state in, events out — the
 * PlayerView owns the <audio> element. Scrubber = an accessible range input; skip ±15/30s;
 * speed cycles through the PRD rate set.
 */
import { computed, ref } from "vue"
import { useI18n } from "vue-i18n"
import type { InsightMarker } from "../player/insightMarkers"
import { TRANSPORT_BUTTON_SIZE } from "../player/transportGeometry"
import { formatTime, PLAYBACK_RATES } from "../player/transcriptSync"

const props = defineProps<{
  playing: boolean
  currentTime: number
  duration: number
  rate: number
  /** Insight density ticks (#1140 skip-guide) — where the substance is. */
  markers?: InsightMarker[]
}>()
const emit = defineEmits<{
  (e: "toggle"): void
  (e: "seek", t: number): void
  (e: "skip", delta: number): void
  (e: "cycle-rate"): void
}>()

const { t } = useI18n()
const max = computed(() => (props.duration > 0 ? props.duration : 0))

// #1140 — a smooth density heat-band behind the ticks: bin the markers across the
// timeline and shade each bin by its summed weight (confidence), so dense stretches
// read dark and skippable ones fade out at a glance. The ticks stay crisp on top.
const DENSITY_BUCKETS = 40
const densityBands = computed(() => {
  const ms = props.markers ?? []
  if (ms.length === 0) return []
  const buckets = new Array<number>(DENSITY_BUCKETS).fill(0)
  for (const m of ms) {
    const i = Math.min(
      DENSITY_BUCKETS - 1,
      Math.max(0, Math.floor((m.pct / 100) * DENSITY_BUCKETS))
    )
    buckets[i] += m.weight
  }
  const peak = Math.max(1e-6, ...buckets)
  return buckets.map((v, i) => ({
    i,
    left: (i / DENSITY_BUCKETS) * 100,
    width: 100 / DENSITY_BUCKETS,
    intensity: v / peak,
  }))
})

function onScrub(ev: Event): void {
  emit("seek", Number((ev.target as HTMLInputElement).value))
}

/**
 * The density strip SEEKS, like the scrubber above it (operator 2026-09-30: "it's more just a
 * chart ... when I tap on it, I go to that place"). It shows where the substance is, so tapping a
 * dark stretch has to take you there. Tap to jump; press and drag to scrub.
 *
 * Pointer only, on purpose: the range input directly above is the keyboard and screen-reader
 * control for the same timeline, so the strip stays an image to assistive tech rather than a
 * second, unlabelled slider.
 */
const densityDragging = ref(false)
function seekFromPointer(ev: PointerEvent): void {
  const el = ev.currentTarget as HTMLElement
  const rect = el.getBoundingClientRect()
  if (rect.width <= 0 || max.value <= 0) return
  const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width))
  emit("seek", Math.round(frac * max.value))
}
function onDensityDown(ev: PointerEvent): void {
  densityDragging.value = true
  ;(ev.currentTarget as HTMLElement).setPointerCapture?.(ev.pointerId)
  seekFromPointer(ev)
}
function onDensityMove(ev: PointerEvent): void {
  if (densityDragging.value) seekFromPointer(ev)
}
function onDensityUp(): void {
  densityDragging.value = false
}
</script>

<template>
  <!-- `px-2 py-3` on phones: the transport row inside is WIDTH-BOUND there (see the row comment).
       12px of side padding was 8px the controls could not spare at 390px; 8px keeps the row off the
       card edge without clipping the speed pill. Tablet+ keeps `p-4`. -->
  <!-- `pt-2` on phones (not `py-3`), and the transport row below carries no top margin (operator
       2026-09-30): the empty band above the buttons was 24px, and on an iPhone the scrubber and
       timestamps fell under the tab bar. Every point saved above them is a point of timeline on
       screen. -->
  <div class="rounded-2xl border border-border bg-surface px-2 pb-3 pt-2 sm:p-4">
    <!--
      A MIRROR (operator 2026-09-30): seven cells — three equal cells, play, three equal cells — and
      every secondary control the same size (TRANSPORT_BUTTON_SIZE), so each button on the left has
      its twin at the same distance on the right:

        transcript · capture · ↺15 · PLAY · 30↻ · output · speed

      It was a flex row of three groups with their own gaps (6px / 8px / 4px) and four button sizes
      (44 / 44 / 40 / 32px), so nothing lined up with its partner. A grid fixes the POSITIONS, not
      just the sizes: an absent control (no transcript, no output route on this platform) leaves its
      cell empty instead of sliding its neighbour inward.

      `minmax(0, 1fr)` side cells split what play leaves over exactly evenly, so play is dead-centre
      by construction. Hit areas stay 44px (`lp-tap`); `design-invariants.spec` guards the fit, the
      44px floor, and a >= 44px pitch. On `lg` the left-hand content moves out (the transcript is a
      side column, capture sits in the masthead) but its cells stay, so the mirror holds there too.
    -->
    <div
      class="grid grid-cols-[repeat(3,minmax(0,1fr))_auto_repeat(3,minmax(0,1fr))] items-center justify-items-center"
      data-testid="player-transport"
    >
      <div class="lg:invisible"><slot name="left-outer" :size="TRANSPORT_BUTTON_SIZE" /></div>
      <div class="lg:invisible"><slot name="left-inner" :size="TRANSPORT_BUTTON_SIZE" /></div>
      <!-- One geometry for every secondary control (#1965): a ghost circle. The play button stays
           the only FILLED shape, so it reads as the primary by contrast rather than by size alone. -->
      <button
        type="button"
        class="flex items-center justify-center rounded-full border border-border text-sm font-bold transition hover:bg-overlay sm:text-base"
        :class="TRANSPORT_BUTTON_SIZE"
        :aria-label="t('player.back15')"
        @click="emit('skip', -15)"
      >
        ↺15
      </button>
      <button
        type="button"
        class="mx-1 flex h-14 w-14 items-center justify-center rounded-full bg-accent text-accent-foreground transition active:scale-95 sm:mx-2 sm:h-16 sm:w-16"
        :aria-label="playing ? t('player.pause') : t('player.play')"
        @click="emit('toggle')"
      >
        <!-- Crisp SVG icons, perfectly centred, identical visual weight in both states. -->
        <svg
          v-if="!playing"
          viewBox="0 0 24 24"
          fill="currentColor"
          class="h-7 w-7"
          aria-hidden="true"
        >
          <path
            d="M8 5.14v13.72a1 1 0 0 0 1.53.85l10.4-6.86a1 1 0 0 0 0-1.7L9.53 4.29A1 1 0 0 0 8 5.14z"
          />
        </svg>
        <svg v-else viewBox="0 0 24 24" fill="currentColor" class="h-7 w-7" aria-hidden="true">
          <rect x="6.5" y="5" width="4.2" height="14" rx="1.4" />
          <rect x="13.3" y="5" width="4.2" height="14" rx="1.4" />
        </svg>
      </button>
      <button
        type="button"
        class="flex items-center justify-center rounded-full border border-border text-sm font-bold transition hover:bg-overlay sm:text-base"
        :class="TRANSPORT_BUTTON_SIZE"
        :aria-label="t('player.forward30')"
        @click="emit('skip', 30)"
      >
        30↻
      </button>
      <div><slot name="right-inner" :size="TRANSPORT_BUTTON_SIZE" /></div>
      <button
        type="button"
        class="flex items-center justify-center rounded-full border border-border text-sm font-bold text-canvas-foreground transition hover:bg-overlay"
        :class="TRANSPORT_BUTTON_SIZE"
        :aria-label="t('player.speed')"
        @click="emit('cycle-rate')"
      >
        {{ rate }}×
      </button>
    </div>
    <!-- Position bar UNDER the play buttons (operator): the transport leads, then the TIMELINE —
         the scrubber with the insight-density strip directly beneath it, annotating the same span —
         and the time readout LAST (operator 2026-09-30). It was scrubber / times / density, which
         read as line, two numbers, line again; the two timeline strips belong together and the
         numbers label their ends. -->
    <input
      type="range"
      min="0"
      :max="max"
      step="1"
      :value="currentTime"
      :aria-label="t('player.scrubber')"
      class="mt-2 w-full accent-accent"
      @input="onScrub"
    />
    <!-- Insight density (#1140 "skip guide"): a tick per insight at its moment; clusters show where
         the substance is, and a tap goes there (see `seekFromPointer`). It stays off the accent
         (#2013) — the scrubber is the accent control; grounded ticks read foreground, opacity =
         confidence (the "weight"). -->
    <!-- The hit area is taller than the 10px strip: `py-2` puts the strip 8px under the scrubber
         (the gap it had as `mt-2`) and `-mb-2` gives the bottom padding back to the times row. A
         fingertip on a 10px band would miss it as often as not. `touch-pan-y` lets a vertical
         swipe that starts here still scroll the page; a horizontal one scrubs. -->
    <div
      v-if="(markers?.length ?? 0) > 0"
      class="-mb-2 cursor-pointer touch-pan-y py-2"
      data-testid="player-density-seek"
      @pointerdown="onDensityDown"
      @pointermove="onDensityMove"
      @pointerup="onDensityUp"
      @pointercancel="onDensityUp"
    >
    <div
      class="pointer-events-none relative h-2.5 w-full"
      role="img"
      data-testid="player-insight-density"
      :aria-label="t('player.insightDensity', { count: markers?.length ?? 0 })"
    >
      <!-- Heat-band: dense stretches read dark, skippable ones fade. -->
      <span
        v-for="b in densityBands"
        :key="'d' + b.i"
        class="absolute top-0 h-2.5 bg-canvas-foreground"
        :style="{ left: b.left + '%', width: b.width + '%', opacity: 0.06 + b.intensity * 0.5 }"
        aria-hidden="true"
        data-testid="player-density-band"
      />
      <!-- Precise ticks on top: one per insight, opacity = confidence. -->
      <span
        v-for="m in markers"
        :key="m.id"
        class="absolute top-0 h-2.5 w-[2px] -translate-x-1/2 rounded-full"
        :class="m.grounded ? 'bg-canvas-foreground' : 'bg-muted'"
        :style="{ left: m.pct + '%', opacity: m.weight }"
        data-testid="player-density-tick"
      />
    </div>
    </div>
    <div class="mt-1 flex justify-between font-mono text-xs text-muted tabular-nums" data-testid="player-times">
      <span>{{ formatTime(currentTime) }}</span>
      <span>{{ formatTime(duration) }}</span>
    </div>
    <span class="sr-only">{{ PLAYBACK_RATES.join(", ") }}</span>
  </div>
</template>
