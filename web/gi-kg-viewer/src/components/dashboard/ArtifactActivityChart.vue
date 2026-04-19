<script setup lang="ts">
import { Chart } from 'chart.js'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { dailyGiKgNewCountsLastDays, type DayGiKgBucket } from '../../utils/artifactMtimeBuckets'
import { ensureChartJsRegistered } from '../../utils/chartRegister'
import { rgbaFromToken } from '../../utils/chartTheme'

const props = defineProps<{
  artifactItems: { kind: string; mtime_utc: string }[]
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null

const buckets = computed((): DayGiKgBucket[] =>
  dailyGiKgNewCountsLastDays(props.artifactItems, 30, new Date()),
)

const insight = computed(() => {
  const b = buckets.value
  if (b.length === 0) {
    return undefined
  }
  let streak = 0
  for (let i = b.length - 1; i >= 0; i -= 1) {
    const row = b[i]!
    if (row.gi + row.kg === 0) {
      streak += 1
    } else {
      break
    }
  }
  if (streak >= 14) {
    return 'No new artifacts in 14 days — pipeline may not be running'
  }
  let lastGi = ''
  let lastKg = ''
  for (let i = b.length - 1; i >= 0; i -= 1) {
    const row = b[i]!
    if (!lastGi && row.gi > 0) {
      lastGi = row.day
    }
    if (!lastKg && row.kg > 0) {
      lastKg = row.day
    }
    if (lastGi && lastKg) {
      break
    }
  }
  return `Last GI: ${lastGi || '—'} · Last KG: ${lastKg || '—'}`
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
  const gi = rgbaFromToken('--ps-gi', 0.8)
  const kg = rgbaFromToken('--ps-kg', 0.8)
  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'GI', data: b.map((x) => x.gi), backgroundColor: gi, borderWidth: 0 },
        { label: 'KG', data: b.map((x) => x.kg), backgroundColor: kg, borderWidth: 0 },
      ],
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
            maxRotation: 0,
            autoSkip: true,
            maxTicksLimit: 8,
            font: { size: 9 },
          },
        },
        y: {
          stacked: false,
          beginAtZero: true,
          title: { display: true, text: 'New artifacts', color: 'var(--ps-muted)' },
          ticks: { maxTicksLimit: 5, precision: 0 },
        },
      },
    },
  })
}

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
      New GI/KG artifacts per day (last 30 UTC days)
    </h3>
    <div class="mb-1 flex gap-4 text-[10px] text-muted">
      <span><span class="font-medium text-gi">GI</span> new files / day</span>
      <span><span class="font-medium text-kg">KG</span> new files / day</span>
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
