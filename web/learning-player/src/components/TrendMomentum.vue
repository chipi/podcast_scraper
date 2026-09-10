<script setup lang="ts">
/**
 * ONE momentum presentation shared by every trending surface (BT.4 / F4.2), so a topic and a
 * storyline read identically wherever velocity is shown.
 *
 * - `variant="badge"` — the detail-card idiom (topic card, storyline page): an emerald "↑ Rising ·
 *   N× vs avg" pill + a sparkline. Matches EntitySignals' momentum row.
 * - `variant="rail"` — the trending-rail idiom (Home rails): a direction-coloured "↑ N×" + a
 *   sparkline in the same hue. Matches MomentumRail.
 *
 * `series` is optional — a surface that only has a velocity (no weekly series) still renders the
 * number; the sparkline appears only with ≥2 points.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import Sparkline from './Sparkline.vue'
import { trendArrow, trendColor } from './trending'

const props = withDefaults(
  defineProps<{ velocity: number; series?: number[]; variant?: 'rail' | 'badge' }>(),
  { variant: 'rail' },
)
const { t } = useI18n()
const v = computed(() => Math.round(props.velocity * 10) / 10)
const showSpark = computed(() => (props.series?.length ?? 0) > 1)
</script>

<template>
  <span
    v-if="variant === 'badge'"
    class="inline-flex items-center gap-2 text-sm"
    data-testid="trend-momentum"
  >
    <span
      class="inline-flex items-center gap-1 rounded-full bg-emerald-500/20 px-2 py-0.5 text-xs font-semibold text-emerald-300"
    >
      <span aria-hidden="true">↑</span>
      {{ t('ec.sig_rising') }} · {{ v }}× {{ t('ec.sigVsAvg') }}
    </span>
    <Sparkline v-if="showSpark" :values="series!" class="shrink-0 text-emerald-300" />
  </span>
  <span
    v-else
    class="inline-flex items-center gap-1"
    :style="{ color: trendColor(velocity) }"
    data-testid="trend-momentum"
  >
    <span class="text-xs font-semibold tabular-nums">{{ trendArrow(velocity) }} {{ v }}×</span>
    <Sparkline v-if="showSpark" :values="series!" :width="40" :height="14" />
  </span>
</template>
