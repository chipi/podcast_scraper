<script setup lang="ts">
/**
 * Trending shows (RFC-103 §show) — a stack of full-width artwork "slices". The top few shows each
 * become a short horizontal band cut from the show's cover art, with a legibility scrim, the show
 * name set large over it, and its weekly-cadence sparkline blended on top in the trend colour
 * (rising green / cooling red). A collage of cover slices — colourful + graphic in a short space.
 *
 * Cover art is joined from the already-loaded podcasts list by feed_id (trending show entity_id ==
 * feed_id), so no extra fetch / back-end change. Each band links to the show page.
 */
import { computed, ref } from 'vue'
import { getTrending } from '../services/api'
import type { Podcast, TrendingEntity } from '../services/types'
import { showArtwork } from '../utils/episode'
import Sparkline from './Sparkline.vue'
import { trendArrow, trendColor, trendDirection } from './trending'

const props = withDefaults(
  defineProps<{ title: string; podcasts: Podcast[]; scope?: 'corpus' | 'mine'; top?: number }>(),
  { scope: 'corpus', top: 5 },
)

const items = ref<TrendingEntity[]>([])
void getTrending('show', props.scope, 12)
  .then((rows) => (items.value = rows))
  .catch(() => (items.value = []))
const shown = computed(() => items.value.slice(0, props.top))
const hasAny = computed(() => shown.value.length > 0)

// feed_id → artwork url (trending show entity_id == feed_id).
const artById = computed<Record<string, string | null>>(() => {
  const out: Record<string, string | null> = {}
  for (const p of props.podcasts) out[p.feed_id] = showArtwork(p)
  return out
})
function artFor(id: string): string | null {
  return artById.value[id] ?? null
}
// A different horizontal band of each cover, so stacked slices don't all show the same region.
function slicePos(i: number): string {
  return `50% ${20 + i * 15}%`
}
// Fallback gradient hue when a show has no artwork — keep the collage colourful.
function fallbackBg(i: number): string {
  const h = (i * 61) % 360
  return `linear-gradient(120deg, hsl(${h},60%,32%), hsl(${(h + 40) % 360},60%,20%))`
}
function vFmt(v: number): number {
  return Math.round(v * 10) / 10
}
function titleOf(e: TrendingEntity): string {
  const dir = trendDirection(e.velocity)
  const word = dir === 'up' ? 'rising' : dir === 'down' ? 'cooling' : 'steady'
  return `${e.label} — ${vFmt(e.velocity)}× (${word})`
}
</script>

<template>
  <section v-if="hasAny" class="mt-7" data-testid="trending-shows-rail">
    <h2 class="lp-section mb-3">{{ title }}</h2>
    <div class="overflow-hidden rounded-2xl border border-border">
      <RouterLink
        v-for="(e, i) in shown"
        :key="e.entity_id"
        :to="{ name: 'podcast', params: { feedId: e.entity_id } }"
        class="group relative flex h-[4.25rem] items-center overflow-hidden no-underline [&:not(:first-child)]:border-t [&:not(:first-child)]:border-black/40"
        :title="titleOf(e)"
        data-testid="trending-show-card"
      >
        <!-- Artwork slice (a horizontal band of the cover), or a colourful fallback gradient. -->
        <img
          v-if="artFor(e.entity_id)"
          :src="artFor(e.entity_id)!"
          alt=""
          class="absolute inset-0 h-full w-full object-cover transition duration-500 group-hover:scale-105"
          :style="{ objectPosition: slicePos(i) }"
        />
        <div v-else class="absolute inset-0" :style="{ background: fallbackBg(i) }" />
        <!-- Left-weighted scrim so the name stays legible over any art. -->
        <div class="absolute inset-0 bg-gradient-to-r from-black/85 via-black/60 to-black/25" />

        <div class="relative flex w-full items-center gap-3 px-4">
          <span class="w-4 shrink-0 text-sm font-bold tabular-nums text-white/60">{{ i + 1 }}</span>
          <span
            class="min-w-0 flex-1 truncate font-display text-lg font-extrabold tracking-tight text-white [text-shadow:0_1px_6px_rgba(0,0,0,0.7)]"
            >{{ e.label }}</span
          >
          <Sparkline
            :values="e.series"
            :width="72"
            :height="28"
            class="shrink-0 drop-shadow-[0_1px_3px_rgba(0,0,0,0.8)]"
            :style="{ color: trendColor(e.velocity) }"
          />
          <span
            class="shrink-0 rounded-full bg-black/55 px-2 py-0.5 text-xs font-bold tabular-nums backdrop-blur"
            :style="{ color: trendColor(e.velocity) }"
            >{{ trendArrow(e.velocity) }} {{ vFmt(e.velocity) }}×</span
          >
        </div>
      </RouterLink>
    </div>
  </section>
</template>
