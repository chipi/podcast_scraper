<script setup lang="ts">
/**
 * Trending topics — Dashboard Intelligence card (the operator's analytical lens
 * on the same signal the player Home surfaces). Reads temporal_velocity: topics
 * "heating up" (last month >= 1.5x their 6-month average, with a mentions floor).
 * Four views to compare: Pills · Sparklines · Over time · Momentum. All SVG.
 * Clicking a topic opens its node view (subject.focusTopic).
 */
import { computed, ref, watch } from 'vue'
import { fetchCachedCorpusEnvelope } from '../../composables/useEnrichmentEnvelopeCache'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'
import { trendArrow, trendColor } from '../../utils/trend'

const shell = useShellStore()
const subject = useSubjectStore()

interface RisingTopic {
  id: string
  label: string
  v: number
  total: number
  series: number[]
}

interface VelocityEnvelope {
  window_months?: string[]
  topics?: Array<{
    topic_id: string
    topic_label?: string
    velocity_last_over_6mo?: number
    total?: number
    monthly_counts?: Record<string, number>
  }>
}

const RISING = 1.5
const MIN_TOTAL = 3
const MAX = 12

const months = ref<string[]>([])
const topics = ref<RisingTopic[]>([])
const loading = ref(false)

async function load(): Promise<void> {
  const root = shell.corpusPath?.trim()
  if (!root || !shell.healthStatus) {
    topics.value = []
    return
  }
  loading.value = true
  const env = await fetchCachedCorpusEnvelope<VelocityEnvelope>(root, 'temporal_velocity').catch(
    () => null,
  )
  loading.value = false
  const rows = env?.data?.topics ?? []
  const axis =
    env?.data?.window_months && env.data.window_months.length
      ? [...env.data.window_months]
      : [...new Set(rows.flatMap((r) => Object.keys(r.monthly_counts ?? {})))].sort()
  months.value = axis
  topics.value = rows
    .filter((x) => (x.velocity_last_over_6mo ?? 0) >= RISING && (x.total ?? 0) >= MIN_TOTAL)
    .sort((a, b) => (b.velocity_last_over_6mo ?? 0) - (a.velocity_last_over_6mo ?? 0))
    .slice(0, MAX)
    .map((x) => ({
      id: x.topic_id,
      label: x.topic_label?.trim() || x.topic_id.replace(/^topic:/, '').replace(/[-_]+/g, ' '),
      v: Math.round((x.velocity_last_over_6mo ?? 0) * 10) / 10,
      total: x.total ?? 0,
      series: axis.map((m) => x.monthly_counts?.[m] ?? 0),
    }))
}
watch(() => [shell.corpusPath, shell.healthStatus], () => void load(), { immediate: true })

function open(id: string): void {
  subject.focusTopic(id)
}

type View = 'chips' | 'sparks' | 'stream' | 'momentum'
const view = ref<View>('chips')
const VIEWS: Array<{ key: View; label: string }> = [
  { key: 'chips', label: 'Pills' },
  { key: 'sparks', label: 'Sparklines' },
  { key: 'stream', label: 'Over time' },
  { key: 'momentum', label: 'Momentum' },
]

// ── Sparkline path (per row) ─────────────────────────────────────────────────
function sparkPath(series: number[], w = 72, h = 22): string {
  const vals = series.length ? series : [0]
  const max = Math.max(1, ...vals)
  const n = vals.length
  return vals
    .map((val, i) => {
      const x = n > 1 ? (i * w) / (n - 1) : w / 2
      const y = h - (val / max) * (h - 2) - 1
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')
}

// ── Stream (stacked area) ────────────────────────────────────────────────────
const SW = 320
const SH = 120
const SPAD_B = 16
const SPAD_T = 6
const STREAM_COLORS = ['#8b5cf6', '#22d3ee', '#f59e0b', '#34d399', '#f472b6', '#60a5fa']
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
function shortMonth(ym: string): string {
  const m = /^(\d{4})-(\d{2})/.exec(ym)
  return m ? (MON[Number(m[2]) - 1] ?? ym) : ym
}
const stream = computed(() => {
  const t = topics.value.slice(0, 6)
  const M = months.value.length
  if (!t.length || M < 2) return { bands: [], xLabels: [] as Array<{ x: number; label: string }> }
  const cum: number[][] = []
  for (let i = 0; i < M; i++) {
    let s = 0
    const row: number[] = []
    for (let k = 0; k < t.length; k++) {
      s += t[k].series[i] ?? 0
      row.push(s)
    }
    cum.push(row)
  }
  const ymax = Math.max(1, ...cum.map((r) => r[r.length - 1]))
  const x = (i: number): number => (M > 1 ? (i / (M - 1)) * SW : SW / 2)
  const y = (v: number): number => SPAD_T + (SH - SPAD_B - SPAD_T) * (1 - v / ymax)
  const bands = t.map((tp, k) => {
    const upper: string[] = []
    const lower: string[] = []
    for (let i = 0; i < M; i++) {
      upper.push(`${x(i).toFixed(1)},${y(cum[i][k]).toFixed(1)}`)
      lower.push(`${x(i).toFixed(1)},${y(k > 0 ? cum[i][k - 1] : 0).toFixed(1)}`)
    }
    return {
      id: tp.id,
      label: tp.label,
      color: STREAM_COLORS[k % STREAM_COLORS.length],
      path: `M${upper.join(' L')} L${lower.reverse().join(' L')} Z`,
    }
  })
  const idxs = M <= 3 ? months.value.map((_, i) => i) : [0, Math.floor((M - 1) / 2), M - 1]
  return { bands, xLabels: idxs.map((i) => ({ x: x(i), label: shortMonth(months.value[i]) })) }
})

// ── Momentum (scatter) ───────────────────────────────────────────────────────
const MW = 320
const MH = 150
const momentum = computed(() => {
  const t = topics.value
  if (!t.length) return []
  const maxTotal = Math.max(1, ...t.map((x) => x.total))
  const maxV = Math.max(1.6, ...t.map((x) => x.v))
  const xOf = (total: number): number => 10 + (total / maxTotal) * (MW - 22)
  const yOf = (v: number): number => MH - 20 - ((v - 1) / (maxV - 1)) * (MH - 32)
  return t.map((tp) => {
    const cx = xOf(tp.total)
    const cy = yOf(tp.v)
    const r = 3 + (tp.total / maxTotal) * 5
    return { id: tp.id, label: tp.label, v: tp.v, cx, cy, r, lx: Math.min(MW - 2, cx + r + 3), ly: Math.max(8, cy - r - 2) }
  })
})

const hasAny = computed(() => topics.value.length > 0)
</script>

<template>
  <section
    class="rounded border border-border bg-surface p-3 text-surface-foreground"
    data-testid="intelligence-trending"
  >
    <div class="mb-2 flex items-center justify-between gap-2">
      <h3 class="text-sm font-semibold">Trending topics</h3>
      <div
        v-if="hasAny"
        role="tablist"
        aria-label="Trending view"
        class="inline-flex flex-wrap gap-0.5 rounded border border-border p-0.5 text-[11px]"
      >
        <button
          v-for="opt in VIEWS"
          :key="opt.key"
          type="button"
          role="tab"
          :aria-selected="view === opt.key"
          :data-testid="`trend-view-${opt.key}`"
          class="rounded px-2 py-0.5 font-medium transition"
          :class="view === opt.key ? 'bg-primary text-primary-foreground' : 'text-muted hover:text-surface-foreground'"
          @click="view = opt.key"
        >
          {{ opt.label }}
        </button>
      </div>
    </div>

    <p v-if="loading" class="text-xs text-muted">Loading…</p>
    <p v-else-if="!hasAny" class="text-xs text-muted" data-testid="intelligence-trending-empty">
      No topics are trending up in this corpus.
    </p>

    <template v-else>
      <p class="mb-2 text-xs text-muted">Heating up across the corpus lately.</p>

      <!-- Pills -->
      <div v-if="view === 'chips'" class="flex flex-wrap gap-1.5" data-testid="trend-chips">
        <button
          v-for="tp in topics"
          :key="tp.id"
          type="button"
          class="inline-flex items-center gap-1.5 rounded-full bg-overlay px-2.5 py-1 text-xs transition hover:bg-elevated"
          data-testid="trend-chip"
          @click="open(tp.id)"
        >
          {{ tp.label }}
          <span class="font-semibold" :style="{ color: trendColor(tp.v) }"
            >{{ trendArrow(tp.v) }} {{ tp.v }}×</span
          >
        </button>
      </div>

      <!-- Sparklines -->
      <ul v-else-if="view === 'sparks'" class="flex flex-col gap-0.5" data-testid="trend-sparks">
        <li v-for="tp in topics.slice(0, 8)" :key="tp.id">
          <button
            type="button"
            class="flex w-full items-center gap-3 rounded px-1.5 py-1 text-left transition hover:bg-overlay"
            data-testid="trend-spark-row"
            @click="open(tp.id)"
          >
            <span class="min-w-0 flex-1 truncate text-xs">{{ tp.label }}</span>
            <span class="shrink-0 text-[11px] font-semibold" :style="{ color: trendColor(tp.v) }"
              >{{ trendArrow(tp.v) }} {{ tp.v }}×</span
            >
            <svg viewBox="0 0 72 22" width="72" height="22" preserveAspectRatio="none" class="shrink-0" :style="{ color: trendColor(tp.v) }" aria-hidden="true">
              <path :d="`${sparkPath(tp.series)} L72,22 L0,22 Z`" fill="currentColor" opacity="0.16" />
              <path :d="sparkPath(tp.series)" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke" />
            </svg>
          </button>
        </li>
      </ul>

      <!-- Over time (stream) -->
      <div v-else-if="view === 'stream'" data-testid="trend-stream">
        <svg :viewBox="`0 0 ${SW} ${SH}`" class="w-full" :style="{ height: `${SH}px` }" role="img">
          <path
            v-for="b in stream.bands"
            :key="b.id"
            :d="b.path"
            :fill="b.color"
            fill-opacity="0.8"
            class="cursor-pointer"
            data-testid="trend-stream-band"
            @click="open(b.id)"
          />
          <text v-for="(lb, i) in stream.xLabels" :key="i" :x="Math.min(SW - 10, Math.max(10, lb.x))" :y="SH - 4" text-anchor="middle" class="fill-muted" style="font-size: 9px">{{ lb.label }}</text>
        </svg>
        <div class="mt-2 flex flex-wrap gap-x-3 gap-y-1">
          <button
            v-for="b in stream.bands"
            :key="b.id"
            type="button"
            class="inline-flex items-center gap-1.5 text-[11px] text-muted transition hover:text-surface-foreground"
            data-testid="trend-stream-legend"
            @click="open(b.id)"
          >
            <span class="h-2.5 w-2.5 shrink-0 rounded-sm" :style="{ backgroundColor: b.color }" />
            {{ b.label }}
          </button>
        </div>
      </div>

      <!-- Momentum (scatter) -->
      <div v-else data-testid="trend-momentum">
        <svg :viewBox="`0 0 ${MW} ${MH}`" class="w-full" :style="{ height: `${MH}px` }" role="img">
          <line :x1="10" :y1="MH - 20" :x2="MW - 12" :y2="MH - 20" class="stroke-border" stroke-width="1" />
          <line :x1="10" :y1="12" :x2="10" :y2="MH - 20" class="stroke-border" stroke-width="1" />
          <text :x="MW - 12" :y="MH - 6" text-anchor="end" class="fill-muted" style="font-size: 8px">more episodes →</text>
          <text :x="12" :y="9" class="fill-muted" style="font-size: 8px">↑ rising faster</text>
          <g v-for="p in momentum" :key="p.id" class="cursor-pointer" data-testid="trend-momentum-point" @click="open(p.id)">
            <circle :cx="p.cx" :cy="p.cy" :r="p.r" :fill="trendColor(p.v)" fill-opacity="0.55" :stroke="trendColor(p.v)" stroke-width="1">
              <title>{{ p.label }} — {{ p.v }}×</title>
            </circle>
            <text :x="p.lx" :y="p.ly" class="fill-surface-foreground" style="font-size: 8px">{{ p.label }}</text>
          </g>
        </svg>
      </div>
    </template>
  </section>
</template>
