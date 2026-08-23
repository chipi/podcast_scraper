<script setup lang="ts">
/**
 * Trending shows (RFC-103 §show) — a stack of full-width artwork "slices", composed as LAYERS:
 *   1. background: a horizontal band cut from the show's cover art,
 *   2. a legibility scrim,
 *   3. the cadence sparkline drawn as a full-width glowing "horizon" (gradient area + soft glow)
 *      woven across the whole slice in the trend colour — a graphic layer, not a chart in a column,
 *   4. the show name spanning the whole width on top (truncates only at the edge),
 *   5. a small velocity chip tucked in the corner.
 *
 * Cover art joins from the loaded podcasts list by feed_id (trending show entity_id == feed_id) —
 * no back-end change. Each band links to the show page.
 */
import { computed } from 'vue'
import { useSectionState } from '../composables/useSectionState'
import SectionStatus from './SectionStatus.vue'
import { getTrending } from '../services/api'
import type { Podcast, TrendingEntity } from '../services/types'
import { showArtwork } from '../utils/episode'
import { trendArrow, trendColor, trendDirection } from './trending'

const props = withDefaults(
  defineProps<{ title: string; podcasts: Podcast[]; scope?: 'corpus' | 'mine'; top?: number }>(),
  { scope: 'corpus', top: 5 },
)

// #1591 — a rejection lands in the error phase rather than collapsing into empty, so an outage
// stops rendering identically to "the corpus has no trending shows".
const section = useSectionState<TrendingEntity[]>([])
function load(): Promise<void> {
  return section.load(() => getTrending('show', props.scope, 12))
}
void load()
const shown = computed(() => section.data.value.slice(0, props.top))
const hasAny = computed(() => shown.value.length > 0)

const artById = computed<Record<string, string | null>>(() => {
  const out: Record<string, string | null> = {}
  for (const p of props.podcasts) out[p.feed_id] = showArtwork(p)
  return out
})
function artFor(id: string): string | null {
  return artById.value[id] ?? null
}
// A different horizontal band of each cover per row, so stacked slices vary.
function slicePos(i: number): string {
  return `50% ${20 + i * 15}%`
}
function fallbackBg(i: number): string {
  const h = (i * 61) % 360
  return `linear-gradient(120deg, hsl(${h},60%,32%), hsl(${(h + 40) % 360},60%,18%))`
}
function vFmt(v: number): number {
  return Math.round(v * 10) / 10
}
function titleOf(e: TrendingEntity): string {
  const dir = trendDirection(e.velocity)
  const word = dir === 'up' ? 'rising' : dir === 'down' ? 'cooling' : 'steady'
  return `${e.label} — ${vFmt(e.velocity)}× (${word})`
}

// Full-width sparkline "horizon": the line lives in the lower band of the slice; the area fills
// beneath it. Drawn in a 100×100 viewBox stretched across the slice (preserveAspectRatio none).
function spark(series: number[]): { line: string; area: string } {
  const vals = series.length ? series : [0]
  const max = Math.max(1, ...vals)
  const n = vals.length
  const pts = vals.map((v, i) => {
    const x = n > 1 ? (i / (n - 1)) * 100 : 50
    // Lower ~45% of the slice: peak value → y=52, trough → y=96.
    const y = 52 + (1 - v / max) * 44
    return [x, y] as const
  })
  const line = pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ')
  return { line, area: `${line} L100,100 L0,100 Z` }
}
</script>

<template>
  <section v-if="hasAny || !section.isReady.value" class="mt-7" data-testid="trending-shows-rail">
    <h2 class="lp-section mb-3">{{ title }}</h2>
    <SectionStatus :phase="section.phase.value" :rows="2" @retry="load" />
    <div v-if="hasAny" class="overflow-hidden rounded-2xl border border-border">
      <RouterLink
        v-for="(e, i) in shown"
        :key="e.entity_id"
        :to="{ name: 'podcast', params: { feedId: e.entity_id } }"
        class="group relative block h-14 overflow-hidden no-underline [&:not(:first-child)]:border-t [&:not(:first-child)]:border-black/40"
        :title="titleOf(e)"
        data-testid="trending-show-card"
      >
        <!-- 1. Art slice (or colourful fallback). -->
        <img
          v-if="artFor(e.entity_id)"
          :src="artFor(e.entity_id)!"
          alt=""
          class="absolute inset-0 h-full w-full object-cover transition duration-500 group-hover:scale-105"
          :style="{ objectPosition: slicePos(i) }"
        />
        <div v-else class="absolute inset-0" :style="{ background: fallbackBg(i) }" />
        <!-- 2. Scrim: dark on the left (name), fading right; plus a floor darken. -->
        <div class="absolute inset-0 bg-gradient-to-r from-black/85 via-black/60 to-black/35" />
        <!-- 3. Sparkline HORIZON layer — gradient area + glowing line, full width. -->
        <svg
          class="absolute inset-0 h-full w-full"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          aria-hidden="true"
          :style="{ color: trendColor(e.velocity), filter: `drop-shadow(0 0 3px ${trendColor(e.velocity)})` }"
        >
          <defs>
            <linearGradient :id="`sg-${e.entity_id}`" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="currentColor" stop-opacity="0.5" />
              <stop offset="100%" stop-color="currentColor" stop-opacity="0" />
            </linearGradient>
          </defs>
          <path :d="spark(e.series).area" :fill="`url(#sg-${e.entity_id})`" />
          <path
            :d="spark(e.series).line"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-opacity="0.95"
            stroke-linejoin="round"
            stroke-linecap="round"
            vector-effect="non-scaling-stroke"
          />
        </svg>
        <!-- 4. Name across the whole slice (truncates only at the edge). -->
        <div class="relative flex h-full items-center gap-2.5 px-4">
          <span class="shrink-0 text-sm font-bold tabular-nums text-white/55">{{ i + 1 }}</span>
          <span
            class="min-w-0 flex-1 truncate pr-12 font-display text-lg font-extrabold tracking-tight text-white [text-shadow:0_1px_8px_rgba(0,0,0,0.85)]"
            >{{ e.label }}</span
          >
        </div>
        <!-- 5. Velocity chip, tucked top-right. -->
        <span
          class="absolute right-3 top-2 rounded-full bg-black/50 px-1.5 py-0.5 text-[0.7rem] font-bold tabular-nums backdrop-blur"
          :style="{ color: trendColor(e.velocity) }"
          >{{ trendArrow(e.velocity) }} {{ vFmt(e.velocity) }}×</span
        >
      </RouterLink>
    </div>
  </section>
</template>
