<script setup lang="ts">
import { Chart } from 'chart.js'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { CoverageByMonthItem } from '../../api/corpusCoverageApi'
import { ensureChartJsRegistered } from '../../utils/chartRegister'
import { rgbaFromToken } from '../../utils/chartTheme'

const props = defineProps<{
  rows: CoverageByMonthItem[]
}>()

const emit = defineEmits<{
  'select-month': [ym: string]
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null

const chartModel = computed(() => {
  const list = props.rows.filter((r) => r.total > 0)
  const labels = list.map((r) => r.month)
  const values = list.map((r) => (r.with_gi / r.total) * 100)
  if (labels.length === 0) {
    return null
  }
  const avg = values.reduce((a, b) => a + b, 0) / values.length
  let minI = 0
  for (let i = 1; i < values.length; i += 1) {
    if (values[i]! < values[minI]!) {
      minI = i
    }
  }
  const below = values.map((v) => v < avg)
  const insightParts: string[] = []
  const belowCount = below.filter(Boolean).length
  if (belowCount > 0) {
    const months = list.filter((_, i) => below[i]).map((r) => r.month)
    insightParts.push(
      `${belowCount} month${belowCount === 1 ? '' : 's'} below average — ${months.join(', ')} need attention`,
    )
  } else if (list.length >= 2) {
    const delta = values[values.length - 1]! - values[0]!
    const dir = delta > 1 ? 'improving' : delta < -1 ? 'softening' : 'stable'
    insightParts.push(`Coverage ${dir}: ${Math.abs(Math.round(delta))}pp from ${labels[0]} to ${labels[labels.length - 1]}`)
  }
  return { labels, values, below, avg, minI, insight: insightParts.join(' · ') || undefined }
})

const monthChartTitle = computed(() => {
  const m = chartModel.value
  if (!m) {
    return 'GI coverage by month'
  }
  const n = m.values.filter((_, i) => m.below[i]).length
  return n > 0 ? `${n} months below average GI coverage` : 'GI coverage by month — at or above average'
})

const lowestMonthLine = computed(() => {
  const m = chartModel.value
  if (!m || !m.labels[m.minI]) {
    return ''
  }
  return `Lowest: ${m.labels[m.minI]} — ${m.values[m.minI]!.toFixed(0)}%`
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
  const m = chartModel.value
  if (!m) {
    return
  }
  const ctx = el.getContext('2d')
  if (!ctx) {
    return
  }
  const gi = rgbaFromToken('--ps-gi', 0.75)
  const warn = rgbaFromToken('--ps-warning', 0.75)
  const borderGi = rgbaFromToken('--ps-gi', 0.95)
  const borderWarn = rgbaFromToken('--ps-warning', 0.95)
  const avgLinePlugin = {
    id: 'psAvgLine',
    afterDatasetsDraw(c: Chart<'bar'>) {
      const yScale = c.scales.y
      const xScale = c.scales.x
      if (!yScale || !xScale) {
        return
      }
      const y = yScale.getPixelForValue(m.avg)
      const ctx2 = c.ctx
      ctx2.save()
      ctx2.strokeStyle = 'var(--ps-border)'
      ctx2.lineWidth = 1
      ctx2.beginPath()
      ctx2.moveTo(xScale.left, y)
      ctx2.lineTo(xScale.right, y)
      ctx2.stroke()
      ctx2.fillStyle = 'var(--ps-muted)'
      ctx2.font = '10px Inter, system-ui, sans-serif'
      ctx2.textAlign = 'right'
      ctx2.fillText(`avg ${m.avg.toFixed(0)}%`, xScale.right, y - 4)
      ctx2.restore()
    },
  }
  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: m.labels,
      datasets: [
        {
          label: 'Coverage %',
          data: m.values,
          backgroundColor: m.values.map((_, i) => (m.below[i] ? warn : gi)),
          borderColor: m.values.map((_, i) => (m.below[i] ? borderWarn : borderGi)),
          borderWidth: 1,
        },
      ],
    },
    options: {
      onClick: (_e, els) => {
        const i = els[0]?.index
        if (i == null) {
          return
        }
        const ym = m.labels[i]
        if (ym) {
          emit('select-month', ym)
        }
      },
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (item) => {
              const v = item.parsed.y
              return typeof v === 'number' ? `${v.toFixed(0)}% GI` : String(v)
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { font: { size: 10 } },
        },
        y: {
          beginAtZero: true,
          max: 100,
          title: { display: true, text: 'Coverage %', color: 'var(--ps-muted)' },
          ticks: { maxTicksLimit: 5 },
        },
      },
    },
    plugins: [avgLinePlugin],
  })
}

onMounted(() => {
  buildChart()
})
watch(
  () => props.rows,
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
    data-testid="coverage-by-month-chart"
  >
    <h3 class="mb-1 text-sm font-semibold">
      {{ monthChartTitle }}
    </h3>
    <p
      v-if="lowestMonthLine"
      class="mb-1 text-[10px] text-muted"
    >
      {{ lowestMonthLine }}
    </p>
    <div
      v-if="chartModel"
      class="h-[220px]"
    >
      <canvas ref="canvasRef" />
    </div>
    <p
      v-else
      class="text-xs text-muted"
    >
      No episode metadata found. Run the pipeline to generate corpus data.
    </p>
    <p
      v-if="chartModel?.insight"
      class="mt-2 text-[11px] text-muted"
    >
      {{ chartModel.insight }}
    </p>
  </div>
</template>
