<script setup lang="ts">
import { Chart } from 'chart.js'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { graphNodeLegendLabel } from '../../utils/colors'
import { chartGridColor } from '../../utils/chartTheme'
import { barEndValuePlugin, setBarEndValueFormatter } from '../../utils/chartBarEndValuePlugin'
import { ensureChartJsRegistered } from '../../utils/chartRegister'

const props = defineProps<{
  title: string
  /** Raw type keys (e.g. Episode, Insight) or visual groups. */
  counts: Record<string, number>
  /** Map raw key → axis label (e.g. manifest `stable_feed_dir` → show title). */
  labelMap?: Record<string, string>
  /** One-line chart explanation under the title. */
  helpText?: string
  insightText?: string
  /**
   * When > 0, draws `count` and `count/total` % at bar ends (horizontal bars only).
   * Avoids passing unstable formatter closures through Vue props.
   */
  barEndPercentTotal?: number
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
  const mapped = props.labelMap?.[key]?.trim()
  if (mapped) {
    return mapped.length > 56 ? `${mapped.slice(0, 55)}…` : mapped
  }
  return graphNodeLegendLabel(key)
}

function buildChart(): void {
  ensureChartJsRegistered()
  const el = canvasRef.value
  if (!el) return
  if (chart) {
    setBarEndValueFormatter(chart, null)
    chart.destroy()
    chart = null
  }
  const entries = sortedEntries.value
  if (entries.length === 0) {
    chart = null
    return
  }
  const labels = entries.map(([k]) => labelFor(k))
  const values = entries.map(([, v]) => v)
  const ctx = el.getContext('2d')
  if (!ctx) return
  const total = props.barEndPercentTotal ?? 0
  const useBarEnd = total > 0
  const extraPlugins = useBarEnd ? [barEndValuePlugin] : []
  chart = new Chart(ctx, {
    type: 'bar',
    plugins: extraPlugins,
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
              const raw = entries[i]?.[0] ?? ''
              return labelFor(raw)
            },
            label: (item) => {
              const i = item.dataIndex
              const raw = entries[i]?.[0] ?? ''
              const v = item.parsed.x
              const lines = [`Count: ${typeof v === 'number' ? v.toLocaleString() : String(v)}`]
              if (props.labelMap?.[raw] && raw && props.labelMap[raw] !== raw) {
                lines.push(`Feed id / dir: ${raw}`)
              }
              return lines
            },
          },
        },
      },
      scales: {
        x: {
          beginAtZero: true,
          ticks: { precision: 0 },
          grid: { color: chartGridColor() },
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 11 } },
        },
      },
    },
  })
  if (useBarEnd && chart) {
    setBarEndValueFormatter(chart, (v) => {
      const pct = (v / total) * 100
      return `${v.toLocaleString()} (${pct.toFixed(1)}%)`
    })
  }
}

onMounted(() => {
  buildChart()
})

watch(
  () =>
    [
      props.counts,
      props.title,
      props.labelMap,
      props.helpText,
      props.insightText,
      props.barEndPercentTotal,
    ] as const,
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
