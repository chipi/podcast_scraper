<script setup lang="ts">
/** Trending view 4 — a momentum map: x = volume (total mentions), y = velocity
 *  (× its 6-month average), bubble size = volume. Separates "big & rising" (upper
 *  right) from "niche & spiking" (upper left). SVG, no lib. Points → topic card. */
import { computed } from 'vue'
import type { RisingTopic } from './trending'

const props = defineProps<{ topics: RisingTopic[] }>()
const emit = defineEmits<{ (e: 'open', id: string): void }>()

const W = 320
const H = 150
const PAD_L = 10
const PAD_R = 12
const PAD_T = 12
const PAD_B = 20

const points = computed(() => {
  const t = props.topics
  if (!t.length) return []
  const maxTotal = Math.max(1, ...t.map((x) => x.total))
  const maxV = Math.max(1.6, ...t.map((x) => x.v))
  const xOf = (total: number): number => PAD_L + (total / maxTotal) * (W - PAD_L - PAD_R)
  const yOf = (v: number): number => H - PAD_B - ((v - 1) / (maxV - 1)) * (H - PAD_B - PAD_T)
  return t.map((tp) => {
    const cx = xOf(tp.total)
    const cy = yOf(tp.v)
    const r = 3 + (tp.total / maxTotal) * 5
    // Label up-right of the bubble, clamped inside the canvas.
    const lx = Math.min(W - 2, cx + r + 3)
    return { id: tp.id, label: tp.label, v: tp.v, cx, cy, r, lx, ly: Math.max(8, cy - r - 2) }
  })
})
</script>

<template>
  <div data-testid="trend-momentum">
    <svg :viewBox="`0 0 ${W} ${H}`" class="w-full" :style="{ height: `${H}px` }" role="img">
      <!-- axes -->
      <line :x1="PAD_L" :y1="H - PAD_B" :x2="W - PAD_R" :y2="H - PAD_B" stroke="currentColor" class="text-border" stroke-width="1" />
      <line :x1="PAD_L" :y1="PAD_T" :x2="PAD_L" :y2="H - PAD_B" stroke="currentColor" class="text-border" stroke-width="1" />
      <text :x="W - PAD_R" :y="H - 6" text-anchor="end" class="fill-muted" style="font-size: 8px">more episodes →</text>
      <text :x="PAD_L + 2" :y="PAD_T - 3" class="fill-muted" style="font-size: 8px">↑ rising faster</text>
      <g v-for="p in points" :key="p.id" class="cursor-pointer" data-testid="trend-momentum-point" @click="emit('open', p.id)">
        <circle :cx="p.cx" :cy="p.cy" :r="p.r" fill="#8b5cf6" fill-opacity="0.55" stroke="#a78bfa" stroke-width="1">
          <title>{{ p.label }} — {{ p.v }}×</title>
        </circle>
        <text :x="p.lx" :y="p.ly" class="fill-canvas-foreground" style="font-size: 8px">{{ p.label }}</text>
      </g>
    </svg>
  </div>
</template>
