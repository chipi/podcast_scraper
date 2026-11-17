<script setup lang="ts">
import { Chart } from 'chart.js'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useThemeChartReloader } from '../../composables/useThemeChartReloader'
import { chartAxisBorderColor, chartSeriesColors, chartTicks } from '../../utils/chartTheme'
import { ensureChartJsRegistered } from '../../utils/chartRegister'
import type { TimeseriesSeries } from './timeseriesChart'

/**
 * Generic multi-series monthly timeseries for the Index section. The caller
 * supplies a shared `labels` month axis plus parallel `series` (data aligned to
 * `labels` by index). Each series gets an enable/disable checkbox (its own
 * legend) and a shared month-range filter. Used to plot Episodes (corpus
 * coverage) alongside one line per indexed document type — all by publish month.
 */
const props = defineProps<{
  labels: string[]
  series: TimeseriesSeries[]
  /** Optional caption under the chart. */
  caption?: string
}>()

const enabled = ref<Record<string, boolean>>({})
watch(
  () => props.series,
  (list) => {
    for (const s of list) {
      if (!(s.key in enabled.value)) {
        enabled.value[s.key] = s.defaultEnabled ?? true
      }
    }
  },
  { immediate: true },
)

const fromMonth = ref('')
const toMonth = ref('')

// Default the range to the full span of available data (first…last month). Seed
// once per dataset; a later corpus switch (labels emptied) re-seeds, but user
// edits are preserved while data is present.
let seeded = false
watch(
  () => props.labels,
  (labels) => {
    if (!labels.length) {
      seeded = false
      return
    }
    if (!seeded) {
      fromMonth.value = labels[0]!
      toMonth.value = labels[labels.length - 1]!
      seeded = true
    }
  },
  { immediate: true },
)

/** Indices into `labels` that fall within the [from, to] month range (inclusive). */
const visibleIndices = computed(() => {
  const lo = fromMonth.value.trim()
  const hi = toMonth.value.trim()
  const out: number[] = []
  props.labels.forEach((m, i) => {
    if (lo && m < lo) return
    if (hi && m > hi) return
    out.push(i)
  })
  return out
})
const filteredLabels = computed(() => visibleIndices.value.map((i) => props.labels[i] ?? ''))

const canvasRef = ref<HTMLCanvasElement | null>(null)
let chart: Chart | null = null
const chartHeightPx = 200

function buildChart(): void {
  ensureChartJsRegistered()
  const el = canvasRef.value
  if (!el) return
  if (chart) {
    chart.destroy()
    chart = null
  }
  if (filteredLabels.value.length === 0) return
  const ctx = el.getContext('2d')
  if (!ctx) return // jsdom/happy-dom: no canvas — guarded like the other charts

  const palette = chartSeriesColors(props.series.length)
  const datasets = props.series
    .map((s, i) => ({ s, color: palette[i % palette.length]! }))
    .filter(({ s }) => enabled.value[s.key])
    .map(({ s, color }) => ({
      label: s.label,
      data: visibleIndices.value.map((idx) => s.data[idx] ?? 0),
      borderColor: color,
      backgroundColor: color,
      tension: 0.25,
      pointRadius: 2,
      borderWidth: 1.5,
    }))

  chart = new Chart(ctx, {
    type: 'line',
    data: { labels: filteredLabels.value, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false }, // our checkboxes are the legend
        tooltip: {
          callbacks: {
            title: (items) => `Month: ${filteredLabels.value[items[0]?.dataIndex ?? 0] ?? ''}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { ...chartTicks(10), maxRotation: 45 },
          border: { display: true, color: chartAxisBorderColor() },
          grid: { display: false },
        },
        y: {
          beginAtZero: true,
          ticks: { ...chartTicks(10), precision: 0 },
          border: { display: true, color: chartAxisBorderColor() },
          grid: { display: false },
        },
      },
    },
  })
}

useThemeChartReloader(buildChart)
onMounted(buildChart)
watch([filteredLabels, enabled, () => props.series], buildChart, { deep: true })
onBeforeUnmount(() => {
  if (chart) {
    chart.destroy()
    chart = null
  }
})
</script>

<template>
  <div
    class="rounded border border-border bg-surface p-2"
    data-testid="index-timeseries"
  >
    <div class="mb-1 flex flex-wrap items-center justify-between gap-2">
      <p class="text-[10px] font-medium text-surface-foreground">Documents by month</p>
      <div class="flex items-center gap-1 text-[9px] text-muted">
        <label class="flex items-center gap-0.5">
          from
          <input
            v-model="fromMonth"
            type="month"
            class="rounded border border-border bg-elevated px-1 py-0.5 text-[9px] text-elevated-foreground"
            data-testid="index-date-from"
          >
        </label>
        <label class="flex items-center gap-0.5">
          to
          <input
            v-model="toMonth"
            type="month"
            class="rounded border border-border bg-elevated px-1 py-0.5 text-[9px] text-elevated-foreground"
            data-testid="index-date-to"
          >
        </label>
      </div>
    </div>
    <div class="mb-1 flex flex-wrap gap-2">
      <label
        v-for="s in series"
        :key="s.key"
        class="flex items-center gap-1 text-[9px] text-muted"
      >
        <input
          v-model="enabled[s.key]"
          type="checkbox"
          :data-testid="`index-series-toggle-${s.key}`"
        >
        {{ s.label }}
      </label>
    </div>
    <p
      v-if="!filteredLabels.length"
      class="text-[10px] text-muted"
    >
      No data in the selected range.
    </p>
    <div
      v-else
      class="relative w-full"
      :style="{ height: `${chartHeightPx}px` }"
    >
      <canvas ref="canvasRef" />
    </div>
    <p class="mt-1 text-[9px] text-muted leading-snug">
      {{
        caption ||
        'By publish month. Episodes from corpus coverage; document types from the index ' +
          '(the index records no per-document write time).'
      }}
    </p>
  </div>
</template>
