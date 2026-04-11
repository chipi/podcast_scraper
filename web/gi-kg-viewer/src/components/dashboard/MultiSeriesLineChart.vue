<script setup lang="ts">
import { Chart } from 'chart.js'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { lineEndLabelsPlugin } from '../../utils/chartLineEndLabelsPlugin'
import { chartGridColor, rgbaFromToken } from '../../utils/chartTheme'
import { ensureChartJsRegistered } from '../../utils/chartRegister'

export type LineSeriesSpec = {
  label: string
  values: number[]
  /** Optional CSS token for line (e.g. --ps-primary); falls back to palette. */
  colorToken?: string
}

const props = defineProps<{
  title: string
  labels: string[]
  series: LineSeriesSpec[]
  yLabel?: string
  /** One-line takeaway (Tufte: state the insight). */
  insightText?: string
  /** Short explanation under the title (chart semantics). */
  helpText?: string
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null

const chartHeightPx = 240

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
  const series = props.series.filter((s) => s.values.length === labels.length)
  if (labels.length === 0 || series.length === 0) {
    chart = null
    return
  }
  const ctx = el.getContext('2d')
  if (!ctx) {
    return
  }
  const fallbackTokens: `--ps-${string}`[] = [
    '--ps-primary',
    '--ps-gi',
    '--ps-kg',
    '--ps-warning',
    '--ps-success',
  ]
  const datasets = series.map((s, i) => {
    const token = (s.colorToken ?? fallbackTokens[i % fallbackTokens.length]!) as `--ps-${string}`
    const line = rgbaFromToken(token, 0.9)
    return {
      label: s.label,
      data: s.values,
      borderColor: line,
      backgroundColor: 'transparent',
      fill: false,
      tension: 0.12,
      pointRadius: 2,
      pointHoverRadius: 4,
      borderWidth: 2,
    }
  })

  chart = new Chart(ctx, {
    type: 'line',
    plugins: [lineEndLabelsPlugin],
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => {
              const idx = items[0]?.dataIndex
              if (idx == null) {
                return ''
              }
              return labels[idx] ?? ''
            },
            label: (item) => {
              const ds = item.dataset.label ?? 'Series'
              const v = item.parsed.y
              return `${ds}: ${typeof v === 'number' ? v.toLocaleString() : String(v)}`
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { maxRotation: 45, minRotation: 0, font: { size: 10 } },
          grid: { display: false },
        },
        y: {
          beginAtZero: true,
          ticks: { precision: 0, font: { size: 10 } },
          grid: { color: chartGridColor() },
          title: {
            display: Boolean(props.yLabel),
            text: props.yLabel ?? '',
            font: { size: 10 },
          },
        },
      },
    },
  })
}

const hasData = computed(
  () =>
    props.labels.length > 0 &&
    props.series.some((s) => s.values.length === props.labels.length && s.values.length > 0),
)

onMounted(() => {
  buildChart()
})

watch(
  () =>
    [props.labels, props.series, props.title, props.yLabel, props.helpText, props.insightText] as const,
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
