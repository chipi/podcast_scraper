<script setup lang="ts">
/** Trending view 3 — the top rising topics as a stacked area over the last
 *  months ("what the library's been about, and what's climbing"). SVG, no lib.
 *  Bands + legend are clickable → the topic card. */
import { computed } from 'vue'
import type { RisingTopic } from './trending'

const props = defineProps<{ topics: RisingTopic[]; months: string[] }>()
const emit = defineEmits<{ (e: 'open', id: string): void }>()

const TOP = 6
const W = 320
const H = 120
const PAD_B = 16
const PAD_T = 6
const COLORS = ['#8b5cf6', '#22d3ee', '#f59e0b', '#34d399', '#f472b6', '#60a5fa']
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
function shortMonth(ym: string): string {
  const m = /^(\d{4})-(\d{2})/.exec(ym)
  return m ? (MON[Number(m[2]) - 1] ?? ym) : ym
}

const shown = computed(() => props.topics.slice(0, TOP))

const geom = computed(() => {
  const t = shown.value
  const M = props.months.length
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
  const x = (i: number): number => (M > 1 ? (i / (M - 1)) * W : W / 2)
  const y = (v: number): number => PAD_T + (H - PAD_B - PAD_T) * (1 - v / ymax)
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
      color: COLORS[k % COLORS.length],
      path: `M${upper.join(' L')} L${lower.reverse().join(' L')} Z`,
    }
  })
  const idxs = M <= 3 ? props.months.map((_, i) => i) : [0, Math.floor((M - 1) / 2), M - 1]
  const xLabels = idxs.map((i) => ({ x: x(i), label: shortMonth(props.months[i]) }))
  return { bands, xLabels }
})
</script>

<template>
  <div data-testid="trend-stream">
    <svg :viewBox="`0 0 ${W} ${H}`" class="w-full" :style="{ height: `${H}px` }" role="img">
      <path
        v-for="b in geom.bands"
        :key="b.id"
        :d="b.path"
        :fill="b.color"
        fill-opacity="0.8"
        stroke="var(--lp-canvas, #0b0b0f)"
        stroke-width="0.5"
        class="cursor-pointer transition-[fill-opacity] hover:fill-opacity-100"
        :data-testid="`trend-stream-band`"
        @click="emit('open', b.id)"
      />
      <text
        v-for="(lb, i) in geom.xLabels"
        :key="i"
        :x="Math.min(W - 10, Math.max(10, lb.x))"
        :y="H - 4"
        text-anchor="middle"
        class="fill-muted"
        style="font-size: 9px"
      >{{ lb.label }}</text>
    </svg>
    <div class="mt-2 flex flex-wrap gap-x-3 gap-y-1">
      <button
        v-for="b in geom.bands"
        :key="b.id"
        type="button"
        class="inline-flex items-center gap-1.5 text-xs text-muted transition hover:text-canvas-foreground"
        data-testid="trend-stream-legend"
        @click="emit('open', b.id)"
      >
        <span class="h-2.5 w-2.5 shrink-0 rounded-sm" :style="{ backgroundColor: b.color }" />
        {{ b.label }}
      </button>
    </div>
  </div>
</template>
