<script setup lang="ts">
import { Chart } from 'chart.js'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { artifactActivitySeries } from '../../extensions'
import { dailyNewCountsLastDays, type DayKindBucket } from '../../utils/artifactMtimeBuckets'
import { ensureChartJsRegistered } from '../../utils/chartRegister'
import { useThemeChartReloader } from '../../composables/useThemeChartReloader'
import { chartAxisBorderColor, chartGridColor, chartTickColor, chartTicks, rgbaFromToken } from '../../utils/chartTheme'

const props = defineProps<{
  artifactItems: { kind: string; mtime_utc: string }[]
  /** The list was fetched successfully. Until then an empty list means "unknown", and reading
   *  it as 14 silent days is a false alarm on an active corpus. */
  loaded: boolean
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null

/** A build's series first (`artifactActivitySeries`), then the viewer's own KG. */
const series = [
  ...artifactActivitySeries.list(),
  { id: 'kg', kind: 'kg', label: 'KG', token: '--ps-kg' as const, textClass: 'text-kg' },
]

const buckets = computed((): DayKindBucket[] =>
  dailyNewCountsLastDays(props.artifactItems, series.map((x) => x.kind), 30, new Date()),
)

const insight = computed(() => {
  const b = buckets.value
  if (!props.loaded || b.length === 0) {
    return undefined
  }
  let streak = 0
  for (let i = b.length - 1; i >= 0; i -= 1) {
    if (Object.values(b[i]!.counts).every((n) => n === 0)) {
      streak += 1
    } else {
      break
    }
  }
  if (streak >= 14) {
    return 'No new artifacts in 14 days — pipeline may not be running'
  }
  return series
    .map((x) => {
      const last = [...b].reverse().find((row) => (row.counts[x.kind] ?? 0) > 0)
      return `Last ${x.label}: ${last?.day ?? '—'}`
    })
    .join(' · ')
})

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
  const b = buckets.value
  if (b.length === 0) {
    return
  }
  const ctx = el.getContext('2d')
  if (!ctx) {
    return
  }
  const labels = b.map((x) => x.day.slice(5))
  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: series.map((x) => ({
        label: x.label,
        data: b.map((row) => row.counts[x.kind] ?? 0),
        backgroundColor: rgbaFromToken(x.token, 0.8),
        borderWidth: 0,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { mode: 'index', intersect: false },
      },
      scales: {
        x: {
          stacked: false,
          ticks: {
            ...chartTicks(9),
            maxRotation: 0,
            autoSkip: true,
            maxTicksLimit: 8,
          },
          border: { display: true, color: chartAxisBorderColor() },
          grid: { color: chartGridColor() },
        },
        y: {
          stacked: false,
          beginAtZero: true,
          title: {
            display: true,
            text: 'New artifacts',
            color: chartTickColor(),
            font: { size: 10 },
          },
          ticks: { ...chartTicks(10), maxTicksLimit: 5, precision: 0 },
          border: { display: true, color: chartAxisBorderColor() },
          grid: { color: chartGridColor() },
        },
      },
    },
  })
}

useThemeChartReloader(buildChart)

onMounted(() => {
  buildChart()
})
watch(
  () => props.artifactItems,
  () => {
    buildChart()
  },
  { deep: true },
)
onBeforeUnmount(() => {
  chart?.destroy()
  chart = null
})
</script>

<template>
  <div
    class="rounded border border-border bg-surface p-3 text-surface-foreground"
    data-testid="artifact-activity-chart"
  >
    <h3 class="mb-1 text-sm font-semibold">
      New artifacts per day (last 30 UTC days)
    </h3>
    <div class="mb-1 flex gap-4 text-[10px] text-muted">
      <span
        v-for="x in series"
        :key="x.id"
      ><span
        class="font-medium"
        :class="x.textClass"
      >{{ x.label }}</span> new files / day</span>
    </div>
    <div class="h-[220px]">
      <canvas ref="canvasRef" />
    </div>
    <p
      v-if="insight"
      class="mt-2 text-[11px] text-muted"
    >
      {{ insight }}
    </p>
  </div>
</template>
