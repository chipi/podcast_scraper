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
  <!-- `p-3` on phones: the transport row inside is width-bound there (see the row comment), and
       15px of padding on each side is 30px the controls cannot have. Tablet+ keeps `p-4`. -->
  <div class="rounded-2xl border border-border bg-surface p-3 sm:p-4">
    <input
      type="range"
      min="0"
      :max="max"
      step="1"
      :value="currentTime"
      :aria-label="t('player.scrubber')"
      class="w-full accent-accent"
      @input="onScrub"
    />
    <div class="mt-1 flex justify-between font-mono text-xs text-muted tabular-nums">
      <span>{{ formatTime(currentTime) }}</span>
      <span>{{ formatTime(duration) }}</span>
    </div>

    <!-- Play is DEAD-CENTRE: back-15 / forward-30 flank it symmetrically in the centred flow;
         the speed toggle is pinned right, and the optional `corner` affordance (transcript toggle
         on mobile) is pinned left — both absolute so they add no height and don't tilt the row. -->
    <!-- px-14 reserves the width the two ABSOLUTE clusters occupy (transcript corner on the left,
         speed + queue on the right). Without it the centred group runs underneath them — which it
         did the moment the secondary controls grew from bare text to 44px circles, overlapping the
         forward-30 button with the queue icon. -->
    <!--
      One flex row, three groups — no absolute clusters, no width reservation (#2004 item 9).

      This used to centre the transport with `px-14` (56px) reserving room for two ABSOLUTELY
      positioned clusters. The reservation was symmetric; the content was not. The right cluster
      holds two 44px controls plus a gap — about 96px — so it overhung its 56px reservation by ~40px
      and landed on the forward-30 button. That is the "queue button squeezed between 30s and 1×"
      report, and it is arithmetic rather than styling.

      `justify-between` with real groups lets flexbox do the distribution, so a cluster can grow
      without colliding with anything. It also fixes it for the LEFT side, which #1592 was about to
      make two items wide as well — the same crush, mirrored.
    -->
    <!--
      Phone gaps are tighter than tablet+ because the row is WIDTH-BOUND on a phone (#2004 item 9,
      third pass). Seven controls at the 44px touch minimum plus a 64px play button is 328px of
      target before a single gap, and a 412px screen leaves ~344px inside the card. It did not fit:
      measured `scrollWidth` 387 in a 342px box, and the speed pill was clipped off the right edge.

      Nothing here drops below a 44px HIT area — that is the floor #1594 established, and shrinking
      targets to win layout would trade a real accessibility property for a cosmetic one. The two
      skip buttons shrink their INK to 40px and keep a 44px hit box via `lp-tap`; the play button
      is 56px on phones. Their centres stay >44px apart, so the boxes still do not overlap.
    -->
    <!-- PL.3 NOT DONE (constraint): equal-width flex-1 sides for perfect centring, and shrinking the
         right edge control, both overflow the 412px row (measured in design-invariants.spec) — the
         transport is already maxed at seven 44px controls + a 64px play button. `justify-between`
         with all controls at the 44px minimum is the layout the row can actually hold. -->
    <div class="mt-3 flex items-center justify-between gap-1 sm:gap-2 lg:justify-center lg:gap-6">
      <div class="flex items-center gap-1 sm:gap-2 lg:hidden">
        <slot name="corner" />
      </div>
      <!-- One geometry for every secondary control (#1965): a ghost circle. The row used to be six
           different shapes in a line — rounded-square icon, bare text, filled circle, bare text,
           circle icon, pill — with two of them having no container at all. The play button stays
           the only FILLED shape, so it reads as the primary by contrast rather than by size alone. -->
      <!-- Centre group: the transport proper. Grouped so `justify-between` yields
           left | centre | right rather than five evenly-spread children. -->
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
      <div class="flex items-center gap-1 sm:gap-2">
        <!-- Right affordance next to speed (e.g. the queue button) — pinned with speed so both add
             no row height and don't tilt the centred transport. -->
        <slot name="corner-right" />
        <!-- PL.4 NOT DONE (constraint, not oversight): shrinking this outermost control below 44px
             needs `lp-tap`, whose 44px hit box then overflows the row's right edge (measured: +4px in
             design-invariants.spec). The row is already maxed at the 44px minimum, so a smaller edge
             control regresses either fit or tappability. Kept at h-11 (exactly 44px) as before. -->
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
