<!--
  Episode outcomes as a sorted horizontal bar chart (Tufte: avoid doughnut + legend).
  Filename kept for stable imports from DashboardView.
-->
<script setup lang="ts">
import { Chart } from 'chart.js'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { barEndValuePlugin, setBarEndValueFormatter } from '../../utils/chartBarEndValuePlugin'
import { useThemeChartReloader } from '../../composables/useThemeChartReloader'
import { chartAxisBorderColor, chartGridColor, chartSeriesColors, chartTicks } from '../../utils/chartTheme'
import { ensureChartJsRegistered } from '../../utils/chartRegister'

const props = defineProps<{
  title: string
  segments: Record<string, number>
  insightText?: string
  helpText?: string
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null

const chartHeightPx = 220

const entries = computed(() =>
  Object.entries(props.segments)
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1]),
)

function buildChart(): void {
  ensureChartJsRegistered()
  const el = canvasRef.value
  if (!el) {
    return
  }
  if (chart) {
    setBarEndValueFormatter(chart, null)
    chart.destroy()
    chart = null
  }
  const list = entries.value
  if (list.length === 0) {
    chart = null
    return
  }
  const ctx = el.getContext('2d')
  if (!ctx) {
    return
  }
  const labels = list.map(([k]) => k)
  const values = list.map(([, v]) => v)
  const total = values.reduce((a, b) => a + b, 0)
  const colors = chartSeriesColors(list.length)
  chart = new Chart(ctx, {
    type: 'bar',
    plugins: [barEndValuePlugin],
    data: {
      labels,
      datasets: [
        {
          data: values,
          backgroundColor: colors,
          borderColor: colors,
          borderWidth: 1,
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => items[0]?.label ?? '',
            label: (item) => {
              const raw = item.parsed.x
              const n = typeof raw === 'number' ? raw : Number(raw)
              const pct =
                total > 0 && Number.isFinite(n) ? ((n / total) * 100).toFixed(1) : '0.0'
              return `Count: ${Number.isFinite(n) ? n.toLocaleString() : '—'} (${pct}% of this run)`
            },
          },
        },
      },
      scales: {
        x: {
          beginAtZero: true,
          ticks: { ...chartTicks(10), precision: 0 },
          border: { display: true, color: chartAxisBorderColor() },
          grid: { color: chartGridColor() },
        },
        y: {
          grid: { display: false },
          ticks: chartTicks(10),
          border: { display: true, color: chartAxisBorderColor() },
        },
      },
    },
  })
  setBarEndValueFormatter(chart, (v) => {
    const pct = total > 0 && Number.isFinite(v) ? ((v / total) * 100).toFixed(1) : '0.0'
    return `${Number.isFinite(v) ? v.toLocaleString() : '—'} (${pct}%)`
  })
}

useThemeChartReloader(buildChart)

onMounted(() => {
  buildChart()
})

watch(
  () => [props.segments, props.title, props.helpText, props.insightText] as const,
  () => {
    buildChart()
  },
  { deep: true },
)

onBeforeUnmount(() => {
  if (chart) {
    setBarEndValueFormatter(chart, null)
    chart.destroy()
  }
  chart = null
})
</script>

<template>
  <div class="rounded border border-border bg-surface p-3 text-surface-foreground">
    <h3 class="mb-1 text-sm font-semibold">
      {{ title }}
    </h3>
    <p
      v-if="insightText"
      class="mb-1.5 text-[11px] font-medium leading-snug text-surface-foreground"
    >
      {{ insightText }}
    </p>
    <p
      v-if="helpText"
      class="mb-2 text-[11px] leading-snug text-muted"
    >
      {{ helpText }}
    </p>
    <p
      v-if="entries.length === 0"
      class="text-xs text-muted"
    >
      No segments to chart.
    </p>
    <div
      v-else
      class="relative w-full"
      :style="{ height: `${chartHeightPx}px` }"
    >
      <canvas ref="canvasRef" />
    </div>
  </div>
</template>
