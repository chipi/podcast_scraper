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
const H = 250
const PAD_L = 14
const PAD_R = 16
const PAD_T = 22
const PAD_B = 28
// Cap on-canvas labels — beyond this they collide on a phone. The rest stay tappable
// with a <title> tooltip. Tuned for a ~340px-wide phone column.
const MAX_LABELS = 6
// Approx label line height for the greedy vertical de-overlap.
const LABEL_H = 13

interface Pt {
  id: string
  label: string
  v: number
  total: number
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
  // Bubble size = volume, wide range so differences read at a glance; the floor keeps
  // low-volume topics visible + tappable.
  const rOf = (total: number): number => 6 + (total / maxTotal) * 15
  const pts = t.map((tp) => ({
    id: tp.id,
    label: tp.label,
    v: tp.v,
    total: tp.total,
    cx: xOf(tp.total),
    cy: yOf(tp.v),
    r: rOf(tp.total),
  }))
  // De-overlap: topics with similar volume + velocity land on the same spot, so their
  // bubbles pile into 2–3 blobs and overlapping circles steal each other's taps (the
  // "I see only 3 / wrong topic opens" bug). Push overlapping pairs apart deterministically
  // (index-derived angle when exactly coincident — no RNG), clamped back into the canvas.
  const clampX = (x: number, r: number): number => Math.min(W - PAD_R - r, Math.max(PAD_L + r, x))
  const clampY = (y: number, r: number): number => Math.min(H - PAD_B - r, Math.max(PAD_T + r, y))
  for (let iter = 0; iter < 80; iter += 1) {
    for (let a = 0; a < pts.length; a += 1) {
      for (let b = a + 1; b < pts.length; b += 1) {
        const pa = pts[a]
        const pb = pts[b]
        let dx = pb.cx - pa.cx
        let dy = pb.cy - pa.cy
        let dist = Math.hypot(dx, dy)
        const min = pa.r + pb.r + 3
        if (dist >= min) continue
        if (dist < 0.01) {
          const ang = (a + 1) * 2.399963 // golden angle → stable spread of coincident points
          dx = Math.cos(ang)
          dy = Math.sin(ang)
          dist = 1
        }
        const push = (min - dist) / 2
        const ux = dx / dist
        const uy = dy / dist
        pa.cx = clampX(pa.cx - ux * push, pa.r)
        pa.cy = clampY(pa.cy - uy * push, pa.r)
        pb.cx = clampX(pb.cx + ux * push, pb.r)
        pb.cy = clampY(pb.cy + uy * push, pb.r)
      }
    }
  }
  // Label the most notable (highest velocity, then most volume), capped, dropping any
  // whose label row would collide vertically with an already-placed one.
  const labelled = new Set<string>()
  const placedY: number[] = []
  for (const p of [...pts].sort((a, b) => b.v - a.v || b.total - a.total)) {
    if (labelled.size >= MAX_LABELS) break
    const ly = Math.max(PAD_T, p.cy - p.r - 4)
    if (placedY.some((y) => Math.abs(y - ly) < LABEL_H)) continue
    placedY.push(ly)
    labelled.add(p.id)
  }
  return pts.map((p) => {
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
