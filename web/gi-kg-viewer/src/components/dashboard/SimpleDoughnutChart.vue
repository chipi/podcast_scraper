<script setup lang="ts">
import { Chart } from 'chart.js'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  chartExternalTooltipHandler,
  removeChartExternalTooltip,
} from '../../utils/chartExternalTooltip'
import { chartSeriesColors } from '../../utils/chartTheme'
import { ensureChartJsRegistered } from '../../utils/chartRegister'

const props = defineProps<{
  title: string
  segments: Record<string, number>
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
  chart?.destroy()
  const list = entries.value
  if (list.length === 0) {
    chart = null
    return
  }
  const ctx = el.getContext('2d')
  if (!ctx) {
    return
  }
  const colors = chartSeriesColors(list.length)
  const values = list.map(([, v]) => v)
  const total = values.reduce((a, b) => a + b, 0)
  chart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: list.map(([k]) => k),
      datasets: [
        {
          data: values,
          backgroundColor: colors,
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: 'right',
          labels: { boxWidth: 10, font: { size: 10 } },
        },
        tooltip: {
          enabled: false,
          external: chartExternalTooltipHandler,
          callbacks: {
            title: (items) => items[0]?.label ?? '',
            label: (item) => {
              const raw = item.parsed
              const n = typeof raw === 'number' ? raw : Number(raw)
              const pct = total > 0 && Number.isFinite(n) ? ((n / total) * 100).toFixed(1) : '0.0'
              return `Episodes: ${Number.isFinite(n) ? n.toLocaleString() : '—'} (${pct}% of this run)`
            },
          },
        },
      },
    },
  })
}

onMounted(() => {
  buildChart()
})

watch(
  () => [props.segments, props.title, props.helpText] as const,
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
