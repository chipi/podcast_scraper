<script setup lang="ts">
/** Trending view 4 — a momentum map: x = volume (total mentions), y = velocity
 *  (× its 6-month average), bubble size = volume. Separates "big & rising" (upper
 *  right) from "niche & spiking" (upper left). SVG, no lib. Points → topic card.
 *
 *  Mobile (#13): the old fixed 320×150 box crowded the bubbles and overlapped the
 *  8px labels into an unreadable mess on a phone. Now it is taller + responsive,
 *  labels appear only on the most-notable points (with greedy vertical de-overlap),
 *  fonts + tap targets are larger, and every bubble keeps a <title> tooltip. */
import { computed } from 'vue'
import { trendColor, type RisingTopic } from './trending'

const props = defineProps<{ topics: RisingTopic[] }>()
const emit = defineEmits<{ (e: 'open', id: string): void }>()

const W = 320
const H = 210
const PAD_L = 12
const PAD_R = 14
const PAD_T = 20
const PAD_B = 26
// Cap on-canvas labels — beyond this they collide on a phone. The rest stay tappable
// with a <title> tooltip. Tuned for a ~340px-wide phone column.
const MAX_LABELS = 5
// Approx label line height for the greedy vertical de-overlap.
const LABEL_H = 13

interface Pt {
  id: string
  label: string
  v: number
  cx: number
  cy: number
  r: number
  showLabel: boolean
  lx: number
  ly: number
  anchor: 'start' | 'end'
}

const points = computed<Pt[]>(() => {
  const t = props.topics
  if (!t.length) return []
  const maxTotal = Math.max(1, ...t.map((x) => x.total))
  const maxV = Math.max(1.6, ...t.map((x) => x.v))
  const xOf = (total: number): number => PAD_L + (total / maxTotal) * (W - PAD_L - PAD_R)
  const yOf = (v: number): number => H - PAD_B - ((v - 1) / (maxV - 1)) * (H - PAD_B - PAD_T)
  // Keep input order for rendering (stable point identity + tests); the ranking below only
  // decides which points earn a text label.
  const base = t.map((tp) => {
    const cx = xOf(tp.total)
    const cy = yOf(tp.v)
    const r = 4 + (tp.total / maxTotal) * 6
    return { id: tp.id, label: tp.label, v: tp.v, cx, cy, r }
  })
  // Label the most notable (highest velocity, then rightmost/most volume), capped, dropping any
  // whose label row would collide vertically with an already-placed one.
  const labelled = new Set<string>()
  const placedY: number[] = []
  for (const p of [...base].sort((a, b) => b.v - a.v || b.cx - a.cx)) {
    if (labelled.size >= MAX_LABELS) break
    const ly = Math.max(PAD_T, p.cy - p.r - 4)
    if (placedY.some((y) => Math.abs(y - ly) < LABEL_H)) continue
    placedY.push(ly)
    labelled.add(p.id)
  }
  return base.map((p) => {
    // Anchor labels toward the interior so they never run off an edge on a narrow canvas.
    const right = p.cx > W / 2
    return {
      ...p,
      showLabel: labelled.has(p.id),
      anchor: right ? ('end' as const) : ('start' as const),
      lx: right ? Math.max(PAD_L, p.cx - p.r - 4) : Math.min(W - PAD_R, p.cx + p.r + 4),
      ly: Math.max(PAD_T, p.cy - p.r - 4),
    }
  })
})
</script>

<template>
  <div data-testid="trend-momentum">
    <svg
      :viewBox="`0 0 ${W} ${H}`"
      class="w-full"
      :style="{ minHeight: '190px' }"
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="Topic momentum: episode volume versus velocity"
    >
      <!-- axes -->
      <line :x1="PAD_L" :y1="H - PAD_B" :x2="W - PAD_R" :y2="H - PAD_B" stroke="currentColor" class="text-border" stroke-width="1" />
      <line :x1="PAD_L" :y1="PAD_T" :x2="PAD_L" :y2="H - PAD_B" stroke="currentColor" class="text-border" stroke-width="1" />
      <text :x="W - PAD_R" :y="H - 8" text-anchor="end" class="fill-muted" style="font-size: 10px">more episodes →</text>
      <text :x="PAD_L" :y="PAD_T - 7" class="fill-muted" style="font-size: 10px">↑ rising faster</text>
      <g
        v-for="p in points"
        :key="p.id"
        class="cursor-pointer"
        data-testid="trend-momentum-point"
        @click="emit('open', p.id)"
      >
        <circle :cx="p.cx" :cy="p.cy" :r="p.r" :fill="trendColor(p.v)" fill-opacity="0.55" :stroke="trendColor(p.v)" stroke-width="1.25">
          <title>{{ p.label }} — {{ p.v }}×</title>
        </circle>
        <text
          v-if="p.showLabel"
          :x="p.lx"
          :y="p.ly"
          :text-anchor="p.anchor"
          class="fill-canvas-foreground"
          style="font-size: 10px"
        >{{ p.label }}</text>
      </g>
    </svg>
  </div>
</template>
