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
import { computed } from "vue"
import { useI18n } from "vue-i18n"
import Sparkline from "./Sparkline.vue"
import { trendArrow, trendColor } from "./trending"

const props = withDefaults(
  defineProps<{
    velocity: number
    series?: number[]
    variant?: "rail" | "badge"
    /** Render the pill ONLY (no sparkline) — the topic card shows one universal activity chart
     *  below the pill instead, so the pill must not draw a second, redundant chart (operator). */
    hideSpark?: boolean
  }>(),
  { variant: "rail" }
)
const { t } = useI18n()
const v = computed(() => Math.round(props.velocity * 10) / 10)
const showSpark = computed(() => !props.hideSpark && (props.series?.length ?? 0) > 1)
</script>

<template>
  <span
    v-if="variant === 'badge'"
    class="flex w-full flex-col items-stretch gap-2 text-sm"
    data-testid="trend-momentum"
  >
    <!-- Pill on its own row (self-start so it hugs the left, whitespace-nowrap so "Rising · N× vs
         avg" never breaks in two), then the sparkline FILLS the width below it (operator 2026-09-13
         — a 120px chart floating in a wide card read as unfinished). `w-full` + the svg's
         `preserveAspectRatio="none"` stretches it edge to edge. -->
    <span
      class="inline-flex w-fit items-center gap-1 self-start whitespace-nowrap rounded-full bg-emerald-500/20 px-2.5 py-0.5 text-xs font-semibold text-emerald-300"
    >
      <span aria-hidden="true">↑</span>
      {{ t("ec.sig_rising") }} · {{ v }}× {{ t("ec.sigVsAvg") }}
    </span>
    <Sparkline v-if="showSpark" :values="series!" class="h-8 w-full text-emerald-300" />
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
