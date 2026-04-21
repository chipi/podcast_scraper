<script setup lang="ts">
import { Chart } from 'chart.js'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { rgbaFromToken } from '../../utils/chartTheme'
import { ensureChartJsRegistered } from '../../utils/chartRegister'

const props = defineProps<{
  title: string
  labels: string[]
  values: number[]
  insightText?: string
  helpText?: string
  yAxisLabel?: string
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
  const bar = rgbaFromToken('--ps-gi', 0.75)
  const border = rgbaFromToken('--ps-gi', 0.95)
  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Episodes',
          data: values,
          backgroundColor: bar,
          borderColor: border,
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
            title: (items) => {
              const i = items[0]?.dataIndex
              if (i == null) {
                return ''
              }
              return `Publish month: ${labels[i] ?? ''}`
            },
            label: (item) => {
              const v = item.parsed.y
              const name = props.yAxisLabel ?? 'Episodes'
              return `${name}: ${typeof v === 'number' ? v.toLocaleString() : String(v)}`
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { maxRotation: 45, font: { size: 10 } },
          grid: { display: false },
        },
        y: {
          beginAtZero: true,
          ticks: { precision: 0, font: { size: 10 } },
          grid: { display: false },
        },
      },
    },
  })
}

const hasData = computed(() => props.labels.length > 0 && props.values.length > 0)

onMounted(() => {
  buildChart()
})

watch(
  () =>
    [props.labels, props.values, props.title, props.helpText, props.yAxisLabel, props.insightText] as const,
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
