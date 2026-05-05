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
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null

const isEmpty = computed(() => props.timeline.months.length === 0)

const chartHeightPx = computed(() => {
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
  const barFill = rgbaFromToken('--ps-primary', 0.65)
  const barStroke = rgbaFromToken('--ps-primary', 0.95)
  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Mentions',
          data: values,
          backgroundColor: barFill,
          borderColor: barStroke,
          borderWidth: 1,
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
              `Mentions: ${typeof item.parsed.y === 'number' ? item.parsed.y.toLocaleString() : String(item.parsed.y)}`,
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
      No dated mentions in the loaded graph.
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
