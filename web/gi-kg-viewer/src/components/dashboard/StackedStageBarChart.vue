<script setup lang="ts">
import { Chart } from 'chart.js'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  chartExternalTooltipHandler,
  removeChartExternalTooltip,
} from '../../utils/chartExternalTooltip'
import { chartGridColor, chartSeriesColors } from '../../utils/chartTheme'
import { ensureChartJsRegistered } from '../../utils/chartRegister'

const props = defineProps<{
  title: string
  /** Stage label + wall seconds (non-negative). */
  stages: { label: string; seconds: number }[]
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
  chart?.destroy()
  const stages = props.stages.filter((s) => s.seconds > 0)
  if (stages.length === 0) {
    chart = null
    return
  }
  const ctx = el.getContext('2d')
  if (!ctx) {
    return
  }
  const colors = chartSeriesColors(stages.length)
  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Pipeline time'],
      datasets: stages.map((s, i) => ({
        label: s.label,
        data: [s.seconds],
        stack: 'stages',
        backgroundColor: colors[i],
        borderColor: colors[i],
        borderWidth: 1,
      })),
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: 'bottom',
          labels: { boxWidth: 10, font: { size: 10 } },
        },
        tooltip: {
          enabled: false,
          external: chartExternalTooltipHandler,
          callbacks: {
            title: () => 'Latest pipeline run',
            label: (item) => {
              const lab = item.dataset.label ?? 'Stage'
              const v = item.parsed.x
              const n = typeof v === 'number' ? v : Number(v)
              return `${lab}: ${Number.isFinite(n) ? n.toFixed(2) : '—'} s wall time`
            },
          },
        },
      },
      scales: {
        x: {
          stacked: true,
          beginAtZero: true,
          ticks: { font: { size: 10 } },
          grid: { color: chartGridColor() },
        },
        y: {
          stacked: true,
          grid: { display: false },
          ticks: { font: { size: 10 } },
        },
      },
    },
  })
}

const hasData = computed(() => props.stages.some((s) => s.seconds > 0))

onMounted(() => {
  buildChart()
})

watch(
  () => [props.stages, props.title, props.helpText] as const,
  () => {
    buildChart()
  },
  { deep: true },
)

onBeforeUnmount(() => {
  if (chart) {
    removeChartExternalTooltip(chart)
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
