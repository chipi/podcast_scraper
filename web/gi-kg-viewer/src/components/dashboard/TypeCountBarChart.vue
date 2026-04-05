<script setup lang="ts">
import {
  BarController,
  BarElement,
  CategoryScale,
  Chart,
  Legend,
  LinearScale,
  Tooltip,
} from 'chart.js'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { graphNodeLegendLabel } from '../../utils/colors'
import {
  chartExternalTooltipHandler,
  removeChartExternalTooltip,
} from '../../utils/chartExternalTooltip'

Chart.register(BarController, BarElement, CategoryScale, LinearScale, Tooltip, Legend)

const props = defineProps<{
  title: string
  /** Raw type keys (e.g. Episode, Insight) or visual groups. */
  counts: Record<string, number>
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null

const sortedEntries = computed(() =>
  Object.entries(props.counts)
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1]),
)

const chartHeightPx = computed(() =>
  Math.min(480, Math.max(100, sortedEntries.value.length * 26 + 48)),
)

function labelFor(key: string): string {
  return graphNodeLegendLabel(key)
}

function buildChart(): void {
  const el = canvasRef.value
  if (!el) return
  chart?.destroy()
  const entries = sortedEntries.value
  if (entries.length === 0) {
    chart = null
    return
  }
  const labels = entries.map(([k]) => labelFor(k))
  const values = entries.map(([, v]) => v)
  const ctx = el.getContext('2d')
  if (!ctx) return
  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Count',
          data: values,
          backgroundColor: 'rgba(76, 110, 245, 0.65)',
          borderColor: 'rgba(54, 79, 199, 0.9)',
          borderWidth: 1,
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          enabled: false,
          external: chartExternalTooltipHandler,
          callbacks: {
            title: (items) => {
              const i = items[0]?.dataIndex
              if (i == null) return ''
              return entries[i]?.[0] ?? ''
            },
          },
        },
      },
      scales: {
        x: {
          beginAtZero: true,
          ticks: { precision: 0 },
          grid: { color: 'rgba(128, 128, 128, 0.2)' },
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 11 } },
        },
      },
    },
  })
}

onMounted(() => {
  buildChart()
})

watch(
  () => [props.counts, props.title] as const,
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
    <h3 class="mb-2 text-sm font-semibold">
      {{ title }}
    </h3>
    <p
      v-if="sortedEntries.length === 0"
      class="text-xs text-muted"
    >
      No counts to chart.
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
