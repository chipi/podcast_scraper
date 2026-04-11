<script setup lang="ts">
import { Chart } from 'chart.js'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { barEndValuePlugin, setBarEndValueFormatter } from '../../utils/chartBarEndValuePlugin'
import { chartGridColor, chartSeriesColors } from '../../utils/chartTheme'
import { ensureChartJsRegistered } from '../../utils/chartRegister'

const props = defineProps<{
  title: string
  /** Stage label + wall seconds (non-negative). */
  stages: { label: string; seconds: number }[]
  insightText?: string
  helpText?: string
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null

const chartHeightPx = 200

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
  const stages = props.stages.filter((s) => s.seconds > 0)
  if (stages.length === 0) {
    chart = null
    return
  }
  const ctx = el.getContext('2d')
  if (!ctx) {
    return
  }
  const labels = stages.map((s) => s.label)
  const values = stages.map((s) => s.seconds)
  const colors = chartSeriesColors(stages.length)
  chart = new Chart(ctx, {
    type: 'bar',
    plugins: [barEndValuePlugin],
    data: {
      labels,
      datasets: [
        {
          label: 'Wall time (s)',
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
            title: (items) => {
              const i = items[0]?.dataIndex
              if (i == null) {
                return ''
              }
              return labels[i] ?? ''
            },
            label: (item) => {
              const v = item.parsed.x
              const n = typeof v === 'number' ? v : Number(v)
              return `Wall time: ${Number.isFinite(n) ? n.toFixed(2) : '—'} s`
            },
          },
        },
      },
      scales: {
        x: {
          beginAtZero: true,
          ticks: { font: { size: 10 } },
          grid: { color: chartGridColor() },
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 10 } },
        },
      },
    },
  })
  setBarEndValueFormatter(chart, (v) => `${Number.isFinite(v) ? v.toFixed(2) : '—'} s`)
}

const hasData = computed(() => props.stages.some((s) => s.seconds > 0))

onMounted(() => {
  buildChart()
})

watch(
  () => [props.stages, props.title, props.helpText, props.insightText] as const,
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
      v-if="!hasData"
      class="text-xs text-muted"
    >
      No stage timings in the latest run summary.
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
