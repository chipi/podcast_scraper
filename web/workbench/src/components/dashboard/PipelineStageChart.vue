<script setup lang="ts">
import { Chart } from 'chart.js'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { ensureChartJsRegistered } from '../../utils/chartRegister'
import { useThemeChartReloader } from '../../composables/useThemeChartReloader'
import { chartTickColor, chartTicks, rgbaFromToken } from '../../utils/chartTheme'
import { formatDashboardRunDurationSeconds } from '../../utils/formatDuration'

const props = defineProps<{
  stages: { label: string; seconds: number }[]
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null

const titleInsight = computed(() => {
  const s = props.stages
  if (s.length === 0) {
    return 'Stage timings'
  }
  const max = [...s].sort((a, b) => b.seconds - a.seconds)[0]!
  const tot = s.reduce((a, x) => a + x.seconds, 0)
  const pct = tot > 0 ? Math.round((max.seconds / tot) * 100) : 0
  return `${max.label}: ${pct}% of total run time`
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
  const sorted = [...props.stages].filter((x) => x.seconds > 0).sort((a, b) => b.seconds - a.seconds)
  if (sorted.length === 0) {
    return
  }
  const ctx = el.getContext('2d')
  if (!ctx) {
    return
  }
  const maxS = sorted[0]!.seconds
  const primary = rgbaFromToken('--ps-primary', 0.85)
  const muted = rgbaFromToken('--ps-surface-foreground', 0.35)
  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sorted.map((x) => x.label),
      datasets: [
        {
          data: sorted.map((x) => x.seconds),
          backgroundColor: sorted.map((x) => (x.seconds === maxS ? primary : muted)),
          borderWidth: 0,
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
          callbacks: {
            label: (item) => formatDashboardRunDurationSeconds(Number(item.parsed.x)),
          },
        },
      },
      scales: {
        x: {
          display: false,
          beginAtZero: true,
        },
        y: {
          ticks: chartTicks(11),
          border: { display: false },
          grid: { display: false },
        },
      },
    },
    plugins: [
      {
        id: 'barEndDur',
        afterDatasetsDraw(c) {
          const meta = c.getDatasetMeta(0)
          const ctx2 = c.ctx
          ctx2.save()
          ctx2.fillStyle = chartTickColor()
          ctx2.font = '10px Inter, system-ui, sans-serif'
          ctx2.textAlign = 'left'
          ctx2.textBaseline = 'middle'
          meta.data.forEach((bar, i) => {
            const v = sorted[i]?.seconds ?? 0
            const t = formatDashboardRunDurationSeconds(v)
            if (bar && 'x' in bar && typeof bar.x === 'number') {
              ctx2.fillText(t, bar.x + 6, bar.y)
            }
          })
          ctx2.restore()
        },
      },
    ],
  })
}

useThemeChartReloader(buildChart)

onMounted(() => {
  buildChart()
})
watch(
  () => props.stages,
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
    data-testid="pipeline-stage-timings"
  >
    <h3 class="mb-1 text-sm font-semibold">
      {{ titleInsight }}
    </h3>
    <div class="h-[200px]">
      <canvas ref="canvasRef" />
    </div>
  </div>
</template>
