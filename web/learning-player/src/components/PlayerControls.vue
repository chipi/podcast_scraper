<script setup lang="ts">
/**
 * Transport controls (PRD-039 FR2 / UXS-011). Presentational: state in, events out — the
 * PlayerView owns the <audio> element. Scrubber = an accessible range input; skip ±15/30s;
 * speed cycles through the PRD rate set.
 */
import { computed } from "vue"
import { useI18n } from "vue-i18n"
import type { InsightMarker } from "../player/insightMarkers"
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
</script>

<template>
  <!-- `px-2 py-3` on phones: the transport row inside is WIDTH-BOUND there (see the row comment).
       12px of side padding was 8px the controls could not spare at 390px; 8px keeps the row off the
       card edge without clipping the speed pill. Tablet+ keeps `p-4`. -->
  <div class="rounded-2xl border border-border bg-surface px-2 py-3 sm:p-4">
    <!--
      Play is DEAD-CENTRE, and the row FITS a phone without shrinking any 44px target (operator
      2026-09-13; #2004 item 9). Three groups: an equal-width `flex-1 min-w-0` side, the centre
      transport (↺15 / play / 30↻), an equal-width `flex-1 min-w-0` side.

      Why `flex-1 min-w-0` and not `justify-between`: `justify-between` distributes the GAPS, so an
      uneven pair of side clusters (the right holds queue + speed, the left transcript + capture)
      leaves the play button off-centre — the report was "speed exits the right edge while the left
      has space". Two equal `flex-1` sides centre the middle group as a UNIT; `min-w-0` drops the
      `min-width:auto` content floor so each side renders at exactly free/2 rather than at its own
      content width, which is what makes the centring pixel-exact (measured: play centre == row
      centre at both 390px and 412px). The side groups justify start / end so the outermost controls
      still hug the card edges.

      Nothing drops below a 44px HIT area (#1594): skip buttons keep a 44px hit box via `lp-tap`
      over 40px ink; the play button is 56px on phones. `design-invariants.spec` guards fit
      (overflow <= 0), the 44px floor, and >44px pitch so a future edit cannot silently re-clip it.
      `lg:` collapses to a simple centred flow since the corner slot is `lg:hidden` there.
    -->
    <div class="mt-3 flex items-center gap-1 sm:gap-2 lg:justify-center lg:gap-6">
      <div class="flex min-w-0 flex-1 items-center justify-start gap-1 sm:gap-2 lg:hidden lg:flex-none">
        <slot name="corner" />
      </div>
      <!-- One geometry for every secondary control (#1965): a ghost circle. The row used to be six
           different shapes in a line — rounded-square icon, bare text, filled circle, bare text,
           circle icon, pill — with two of them having no container at all. The play button stays
           the only FILLED shape, so it reads as the primary by contrast rather than by size alone. -->
      <!-- Centre group: the transport proper. Its own group so the equal-width sides centre it as a
           unit, rather than five evenly-spread children. -->
      <div class="flex items-center gap-2 sm:gap-4">
        <button
          type="button"
          class="lp-tap flex h-10 w-10 items-center justify-center rounded-full border border-border text-sm font-bold transition hover:bg-overlay sm:h-11 sm:w-11 sm:text-base"
          :aria-label="t('player.back15')"
          @click="emit('skip', -15)"
        >
          ↺15
        </button>
        <button
          type="button"
          class="flex h-14 w-14 items-center justify-center rounded-full bg-accent text-accent-foreground transition active:scale-95 sm:h-16 sm:w-16"
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
          class="lp-tap flex h-10 w-10 items-center justify-center rounded-full border border-border text-sm font-bold transition hover:bg-overlay sm:h-11 sm:w-11 sm:text-base"
          :aria-label="t('player.forward30')"
          @click="emit('skip', 30)"
        >
          30↻
        </button>
      </div>
      <div class="flex min-w-0 flex-1 items-center justify-end gap-1 sm:gap-2 lg:flex-none">
        <!-- Right affordance next to speed (e.g. the queue button) — pinned with speed so both add
             no row height and don't tilt the centred transport. -->
        <slot name="corner-right" />
        <!-- Speed stays at h-11 (exactly the 44px minimum): it is the outermost control, and its
             right edge now sits flush to the card's inner edge at 390px (measured). Shrinking it
             below 44px would regress tappability for no space the row still needs. -->
        <button
          type="button"
          class="flex h-11 w-11 items-center justify-center rounded-full border border-border text-sm font-bold text-canvas-foreground transition hover:bg-overlay"
          :aria-label="t('player.speed')"
          @click="emit('cycle-rate')"
        >
          {{ rate }}×
        </button>
      </div>
    </div>
    <!-- Position bar UNDER the play buttons (operator): the transport leads, then the scrubber +
         time readout, then the insight-density strip that annotates the same timeline. -->
    <input
      type="range"
      min="0"
      :max="max"
      step="1"
      :value="currentTime"
      :aria-label="t('player.scrubber')"
      class="mt-3 w-full accent-accent"
      @input="onScrub"
    />
    <div class="mt-1 flex justify-between font-mono text-xs text-muted tabular-nums">
      <span>{{ formatTime(currentTime) }}</span>
      <span>{{ formatTime(duration) }}</span>
    </div>
    <!-- Insight density (#1140 "skip guide"): a tick per insight at its moment; clusters show where
         the substance is. Sits BELOW the transport now — the play/scrub controls are what has to be
         reachable without scrolling, so the density strip (a reference, not a control) reads under
         them instead of pushing them down the viewport. Data-viz, not a control, so it stays off the
         accent (#2013): grounded ticks read foreground, opacity = confidence (the "weight"). -->
    <div
      v-if="(markers?.length ?? 0) > 0"
      class="relative mt-3 h-2.5 w-full"
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
    <span class="sr-only">{{ PLAYBACK_RATES.join(", ") }}</span>
  </div>
</template>
