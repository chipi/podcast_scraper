<script setup lang="ts">
/**
 * #672 — Mentions-by-month bar chart for the focused Topic / Entity /
 * Person rail panel. Wraps Chart.js bar (vertical) over
 * ``SubjectMentionsTimeline.months``.
 */
import { Chart } from 'chart.js'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { SubjectMentionsTimeline } from '../../utils/subjectMentionsTimeline'
import { useThemeChartReloader } from '../../composables/useThemeChartReloader'
import { chartAxisBorderColor, chartGridColor, chartTicks, rgbaFromToken } from '../../utils/chartTheme'
import { ensureChartJsRegistered } from '../../utils/chartRegister'

const props = defineProps<{
  timeline: SubjectMentionsTimeline
  /** Optional aria-label override; default "Mentions by month". */
  ariaLabel?: string
  /** ``bar`` (default, vertical bars — person Signals) or ``dots`` (a compact
   *  scatter-style time series, no bars — topic Timeline, review N8). */
  variant?: 'bar' | 'dots'
  /** Series noun for the tooltip; default "Mentions". */
  valueLabel?: string
  /** Empty-state copy; default assumes the graph-derived mentions timeline. */
  emptyText?: string
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null

const isEmpty = computed(() => props.timeline.months.length === 0)

const chartHeightPx = computed(() => {
  // Dots read as a horizontal time series — keep it short + fixed. Bars grow
  // with column count (the existing person-Signals behaviour).
  if (props.variant === 'dots') return 116
  const cols = props.timeline.months.length
  return Math.min(220, Math.max(80, 32 + cols * 18))
})

function buildChart(): void {
  ensureChartJsRegistered()
  const el = canvasRef.value
  if (!el || isEmpty.value) {
    if (chart) {
      chart.destroy()
      chart = null
    }
    return
  }
  if (chart) {
    chart.destroy()
    chart = null
  }
  const ctx = el.getContext('2d')
  if (!ctx) return
  const labels = props.timeline.months.map((m) => m.ymd)
  const values = props.timeline.months.map((m) => m.count)
  const fill = rgbaFromToken('--ps-primary', 0.65)
  const stroke = rgbaFromToken('--ps-primary', 0.95)
  const noun = props.valueLabel ?? 'Mentions'
  const isDots = props.variant === 'dots'
  chart = new Chart(ctx, {
    type: isDots ? 'line' : 'bar',
    data: {
      labels,
      datasets: [
        {
          // Empty label so the psEndLabel plugin (fires on any line chart)
          // draws nothing at the last point; the noun lives in the tooltip.
          label: isDots ? '' : noun,
          data: values,
          ...(isDots
            ? {
                showLine: false,
                pointRadius: 3.5,
                pointHoverRadius: 5.5,
                pointBackgroundColor: fill,
                pointBorderColor: stroke,
                borderColor: stroke,
              }
            : {
                backgroundColor: fill,
                borderColor: stroke,
                borderWidth: 1,
              }),
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => items[0]?.label ?? '',
            label: (item) =>
              `${noun}: ${typeof item.parsed.y === 'number' ? item.parsed.y.toLocaleString() : String(item.parsed.y)}`,
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: chartTicks(10),
          border: { display: true, color: chartAxisBorderColor() },
        },
        y: {
          beginAtZero: true,
          ticks: { ...chartTicks(10), precision: 0 },
          border: { display: true, color: chartAxisBorderColor() },
          grid: { color: chartGridColor() },
        },
      },
    },
  })
}

useThemeChartReloader(buildChart)

onMounted(buildChart)

watch(
  () => [props.timeline] as const,
  buildChart,
  { deep: true },
)

onBeforeUnmount(() => {
  chart?.destroy()
  chart = null
})
</script>

<template>
  <div
    role="figure"
    :aria-label="props.ariaLabel ?? 'Mentions by month'"
    data-testid="subject-timeline-chart"
  >
    <p
      v-if="isEmpty"
      class="text-[11px] text-muted"
      data-testid="subject-timeline-chart-empty"
    >
      {{ props.emptyText ?? 'No dated mentions in the loaded graph.' }}
    </p>
    <div
      v-else
      class="relative w-full"
      :style="{ height: `${chartHeightPx}px` }"
    >
      <canvas
        ref="canvasRef"
        class="block h-full w-full"
        aria-hidden="true"
      />
    </div>
  </div>
</template>
