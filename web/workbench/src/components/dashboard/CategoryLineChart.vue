<script setup lang="ts">
import { Chart } from 'chart.js'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { lineEndLabelsPlugin } from '../../utils/chartLineEndLabelsPlugin'
import { useThemeChartReloader } from '../../composables/useThemeChartReloader'
import { chartAxisBorderColor, chartGridColor, chartTicks, rgbaFromToken } from '../../utils/chartTheme'
import { ensureChartJsRegistered } from '../../utils/chartRegister'

const props = defineProps<{
  /** Omit or leave empty to hide the card heading (parent supplies title). */
  title?: string
  labels: string[]
  values: number[]
  yLabel?: string
  insightText?: string
  helpText?: string
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null

const chartHeightPx = 220

function buildChart(): void {
  ensureChartJsRegistered()
  const el = canvasRef.value
  if (!el) {
    return
  }
  if (chart) {
    chart.destroy()
    chart = null
  }
  const labels = props.labels
  const values = props.values
  if (labels.length === 0) {
    chart = null
    return
  }
  const ctx = el.getContext('2d')
  if (!ctx) {
    return
  }
  const line = rgbaFromToken('--ps-primary', 0.9)
  chart = new Chart(ctx, {
    type: 'line',
    plugins: [lineEndLabelsPlugin],
    data: {
      labels,
      datasets: [
        {
          label: props.yLabel ?? 'Count',
          data: values,
          borderColor: line,
          backgroundColor: 'transparent',
          fill: false,
          tension: 0.12,
          pointRadius: 2,
          pointHoverRadius: 4,
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
            title: (items) => {
              const i = items[0]?.dataIndex
              if (i == null) {
                return ''
              }
              return labels[i] ?? ''
            },
            label: (item) => {
              const v = item.parsed.y
              const name = props.yLabel ?? 'Value'
              return `${name}: ${typeof v === 'number' ? v.toLocaleString() : String(v)}`
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { ...chartTicks(10), maxRotation: 45, minRotation: 0 },
          border: { display: true, color: chartAxisBorderColor() },
          grid: { display: false },
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

const hasData = computed(() => props.labels.length > 0 && props.values.length > 0)

useThemeChartReloader(buildChart)

onMounted(() => {
  buildChart()
})

watch(
  () =>
    [props.labels, props.values, props.title, props.yLabel, props.helpText, props.insightText] as const,
  () => {
    buildChart()
  },
  { deep: true },
)

onBeforeUnmount(() => {
  if (chart) {
    chart.destroy()
  }
  chart = null
})
</script>

<template>
  <div class="rounded border border-border bg-surface p-3 text-surface-foreground">
    <h3
      v-if="title"
      class="mb-1 text-sm font-semibold"
    >
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
      v-if="!hasData"
      class="text-xs text-muted"
    >
      No data to chart.
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
