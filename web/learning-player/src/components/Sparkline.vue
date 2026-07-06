<script setup lang="ts">
/**
 * Minimal area+line sparkline (UXS-014 / ) — ONE canonical mini-chart for listening series,
 * used by both the Profile analytics panel and the player's per-episode reach. Draws in
 * `currentColor`, so the parent sets the hue; scales to its viewBox so it fills any box.
 */
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{ values: number[]; width?: number; height?: number }>(),
  { width: 120, height: 32 },
)

const paths = computed(() => {
  const vals = props.values.length ? props.values : [0]
  const max = Math.max(1, ...vals)
  const n = vals.length
  const { width: w, height: h } = props
  const pts = vals.map((v, i) => {
    const x = n > 1 ? (i * w) / (n - 1) : w / 2
    const y = h - (v / max) * (h - 2) - 1
    return `${x.toFixed(1)},${y.toFixed(1)}`
  })
  const line = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p}`).join(' ')
  return { line, area: `${line} L${w},${h} L0,${h} Z` }
})
</script>

<template>
  <svg
    :viewBox="`0 0 ${width} ${height}`"
    :width="width"
    :height="height"
    preserveAspectRatio="none"
    aria-hidden="true"
  >
    <path :d="paths.area" fill="currentColor" opacity="0.16" />
    <path
      :d="paths.line"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      vector-effect="non-scaling-stroke"
    />
  </svg>
</template>
