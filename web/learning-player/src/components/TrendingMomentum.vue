<script setup lang="ts">
/** Trending view 4 — a **momentum bubble cloud**. Each topic is a translucent balloon:
 *  size = momentum (velocity × its 6-month average), its own hue from a categorical palette,
 *  a soft gloss highlight. Circle-packed into an organic cloud (the biggest nested in the
 *  middle) rather than an x/y scatter: with only ~12 rising topics — many sharing the same
 *  velocity — a scatter stacks them into one blob with colliding labels. Big balloons carry
 *  their label inside; the rest keep a <title> tooltip. Tap a balloon → its topic card. */
import { computed } from 'vue'
import type { RisingTopic } from './trending'

const props = defineProps<{ topics: RisingTopic[]; themeMemberIds?: Set<string> }>()
const emit = defineEmits<{ (e: 'open', id: string): void }>()

const PAD = 6
const GAP = 3.5

// Categorical palette from the app's own hues (topic / theme / green / accent / peach / amber).
const PALETTE: [number, number, number][] = [
  [201, 182, 255],
  [125, 211, 192],
  [63, 185, 132],
  [255, 106, 61],
  [255, 179, 122],
  [232, 179, 57],
  [124, 230, 176],
]

interface Bubble {
  id: string
  label: string
  short: string
  v: number
  total: number
  cx: number
  cy: number
  r: number
  fill: string
  stroke: string
  showLabel: boolean
  labelSize: number
  colorIndex: number
  theme: boolean
}

function shortLabel(label: string, r: number): string {
  const max = Math.max(3, Math.floor((r * 2) / 7))
  return label.length > max ? `${label.slice(0, max - 1).trimEnd()}…` : label
}

const packed = computed(() => {
  const t = props.topics
  if (!t.length) return { w: 320, h: 180, bubbles: [] as Bubble[] }
  const vs = t.map((x) => x.v)
  const minV = Math.min(...vs)
  const maxV = Math.max(minV + 0.1, ...vs)
  // Size by momentum — the widest-ranging signal, so a 6× topic clearly dwarfs a 1.5× one.
  const rOf = (v: number): number => 16 + ((v - minV) / (maxV - minV)) * 36

  // Keep the original index (for a stable hue) while packing biggest-first.
  const sized = t
    .map((tp, i) => ({ ...tp, r: rOf(tp.v), colorIndex: i }))
    .sort((a, b) => b.r - a.r)

  // Deterministic spiral circle-pack: biggest at the origin, each next spiralled outward to
  // the first non-colliding spot. No RNG → stable render + testable.
  const placed: Array<(typeof sized)[number] & { cx: number; cy: number }> = []
  for (const b of sized) {
    if (!placed.length) {
      placed.push({ ...b, cx: 0, cy: 0 })
      continue
    }
    let cx = 0
    let cy = 0
    for (let step = 0; step < 5000; step += 1) {
      const ang = step * 0.5
      const rad = step * 0.32
      cx = Math.cos(ang) * rad
      cy = Math.sin(ang) * rad
      if (placed.every((p) => Math.hypot(p.cx - cx, p.cy - cy) >= p.r + b.r + GAP)) break
    }
    placed.push({ ...b, cx, cy })
  }

  const minX = Math.min(...placed.map((p) => p.cx - p.r))
  const minY = Math.min(...placed.map((p) => p.cy - p.r))
  const maxX = Math.max(...placed.map((p) => p.cx + p.r))
  const maxY = Math.max(...placed.map((p) => p.cy + p.r))
  const bubbles: Bubble[] = placed.map((p) => {
    const [r, g, bl] = PALETTE[p.colorIndex % PALETTE.length]
    return {
      id: p.id,
      label: p.label,
      short: shortLabel(p.label, p.r),
      v: p.v,
      total: p.total,
      cx: p.cx - minX + PAD,
      cy: p.cy - minY + PAD,
      r: p.r,
      fill: `rgba(${r}, ${g}, ${bl}, 0.42)`,
      stroke: `rgba(${r}, ${g}, ${bl}, 0.92)`,
      showLabel: p.r >= 22,
      labelSize: Math.max(9, Math.min(13, Math.round(p.r / 2.6))),
      colorIndex: p.colorIndex,
      theme: props.themeMemberIds?.has(p.id) ?? false,
    }
  })
  return { w: maxX - minX + PAD * 2, h: maxY - minY + PAD * 2, bubbles }
})
</script>

<template>
  <div data-testid="trend-momentum">
    <svg
      :viewBox="`0 0 ${packed.w} ${packed.h}`"
      class="mx-auto w-full"
      :style="{ maxHeight: '360px' }"
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="Trending topics as a bubble cloud; bigger balloons are rising faster"
    >
      <defs>
        <radialGradient id="momentumGloss" cx="38%" cy="30%" r="72%">
          <stop offset="0%" stop-color="rgba(255,255,255,0.45)" />
          <stop offset="46%" stop-color="rgba(255,255,255,0.07)" />
          <stop offset="100%" stop-color="rgba(255,255,255,0)" />
        </radialGradient>
      </defs>
      <g
        v-for="b in packed.bubbles"
        :key="b.id"
        class="cursor-pointer"
        data-testid="trend-momentum-point"
        @click="emit('open', b.id)"
      >
        <circle :cx="b.cx" :cy="b.cy" :r="b.r" :fill="b.fill" :stroke="b.stroke" stroke-width="1.5">
          <title>{{ b.label }} — {{ b.v }}× · {{ b.total }} episodes{{ b.theme ? ' · in a storyline' : '' }}</title>
        </circle>
        <circle :cx="b.cx" :cy="b.cy" :r="b.r" fill="url(#momentumGloss)" pointer-events="none" />
        <!-- Theme-cluster ("storyline") members get the standard teal ring. -->
        <circle
          v-if="b.theme"
          :cx="b.cx"
          :cy="b.cy"
          :r="b.r + 2.5"
          fill="none"
          stroke="#7dd3c0"
          stroke-width="1.5"
          stroke-dasharray="3 3"
          pointer-events="none"
        />
        <template v-if="b.showLabel">
          <text :x="b.cx" :y="b.cy - 1" text-anchor="middle" class="fill-canvas-foreground font-semibold" :style="{ fontSize: b.labelSize + 'px' }" pointer-events="none">{{ b.short }}</text>
          <text :x="b.cx" :y="b.cy + b.labelSize" text-anchor="middle" class="fill-canvas-foreground" :style="{ fontSize: Math.max(8, b.labelSize - 2) + 'px', opacity: 0.72 }" pointer-events="none">↑ {{ b.v }}×</text>
        </template>
      </g>
    </svg>
  </div>
</template>
