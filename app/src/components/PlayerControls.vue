<script setup lang="ts">
/**
 * Transport controls (PRD-039 FR2 / UXS-011). Presentational: state in, events out — the
 * PlayerView owns the <audio> element. Scrubber = an accessible range input; skip ±15/30s;
 * speed cycles through the PRD rate set.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { InsightMarker } from '../player/insightMarkers'
import { formatTime, PLAYBACK_RATES } from '../player/transcriptSync'

const props = defineProps<{
  playing: boolean
  currentTime: number
  duration: number
  rate: number
  /** Insight density ticks (#1140 skip-guide) — where the substance is. */
  markers?: InsightMarker[]
}>()
const emit = defineEmits<{
  (e: 'toggle'): void
  (e: 'seek', t: number): void
  (e: 'skip', delta: number): void
  (e: 'cycle-rate'): void
}>()

const { t } = useI18n()
const max = computed(() => (props.duration > 0 ? props.duration : 0))

function onScrub(ev: Event): void {
  emit('seek', Number((ev.target as HTMLInputElement).value))
}
</script>

<template>
  <div class="rounded-2xl border border-border bg-surface p-4">
    <!-- Insight density (#1140 "skip guide"): a tick per insight at its moment; clusters show
         where the substance is. Grounded → accent colour; opacity = confidence (the "weight"). -->
    <div
      v-if="(markers?.length ?? 0) > 0"
      class="relative mb-1 h-2.5 w-full"
      role="img"
      data-testid="player-insight-density"
      :aria-label="t('player.insightDensity', { count: markers?.length ?? 0 })"
    >
      <span
        v-for="m in markers"
        :key="m.id"
        class="absolute top-0 h-2.5 w-[2px] -translate-x-1/2 rounded-full"
        :class="m.grounded ? 'bg-accent' : 'bg-muted'"
        :style="{ left: m.pct + '%', opacity: m.weight }"
      />
    </div>
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

    <div class="mt-3 flex items-center justify-center gap-6">
      <button type="button" class="font-bold" :aria-label="t('player.back15')" @click="emit('skip', -15)">
        ↺15
      </button>
      <button
        type="button"
        class="flex h-14 w-14 items-center justify-center rounded-full bg-accent text-2xl text-accent-foreground"
        :aria-label="playing ? t('player.pause') : t('player.play')"
        @click="emit('toggle')"
      >
        {{ playing ? '⏸' : '►' }}
      </button>
      <button type="button" class="font-bold" :aria-label="t('player.forward30')" @click="emit('skip', 30)">
        30↻
      </button>
      <button
        type="button"
        class="rounded-full bg-overlay px-3 py-1 text-sm font-bold text-accent"
        :aria-label="t('player.speed')"
        @click="emit('cycle-rate')"
      >
        {{ rate }}×
      </button>
    </div>
    <span class="sr-only">{{ PLAYBACK_RATES.join(', ') }}</span>
  </div>
</template>
