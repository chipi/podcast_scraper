<script setup lang="ts">
/**
 * Trending shows (RFC-103 §show) — a cover-art carousel. Each show is a card: its artwork with a
 * rank badge, a direction-coloured ×velocity badge, and its weekly-cadence sparkline drawn over a
 * bottom scrim (in the trend colour) — the "sparkling" on top of the art. Title sits below. Cover
 * art is joined from the already-loaded podcasts list by feed_id (trending show entity_id == feed_id),
 * so no extra fetch/back-end change. Cards link to the show page. Hides when nothing is trending.
 */
import { computed, ref } from 'vue'
import { getTrending } from '../services/api'
import type { Podcast, TrendingEntity } from '../services/types'
import { showArtwork } from '../utils/episode'
import CardRail from './CardRail.vue'
import Sparkline from './Sparkline.vue'
import { trendArrow, trendColor, trendDirection } from './trending'

const props = withDefaults(
  defineProps<{ title: string; podcasts: Podcast[]; scope?: 'corpus' | 'mine'; limit?: number }>(),
  { scope: 'corpus', limit: 12 },
)

const items = ref<TrendingEntity[]>([])
void getTrending('show', props.scope, props.limit)
  .then((rows) => (items.value = rows))
  .catch(() => (items.value = []))
const hasAny = computed(() => items.value.length > 0)

// feed_id → artwork url (trending show entity_id == feed_id).
const artById = computed<Record<string, string | null>>(() => {
  const out: Record<string, string | null> = {}
  for (const p of props.podcasts) out[p.feed_id] = showArtwork(p)
  return out
})
function artFor(id: string): string | null {
  return artById.value[id] ?? null
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
    <CardRail>
      <li v-for="(e, i) in items" :key="e.entity_id" class="w-36 shrink-0 sm:w-40">
        <RouterLink
          :to="{ name: 'podcast', params: { feedId: e.entity_id } }"
          class="group block no-underline"
          :title="titleOf(e)"
          data-testid="trending-show-card"
        >
          <div class="relative aspect-square overflow-hidden rounded-2xl border border-border bg-elevated">
            <img
              v-if="artFor(e.entity_id)"
              :src="artFor(e.entity_id)!"
              alt=""
              class="h-full w-full object-cover transition duration-300 group-hover:scale-[1.03]"
            />
            <!-- Rank badge -->
            <span
              class="absolute left-2 top-2 flex h-6 min-w-6 items-center justify-center rounded-full bg-canvas/80 px-1.5 text-xs font-extrabold tabular-nums backdrop-blur"
              >{{ i + 1 }}</span
            >
            <!-- Velocity badge (direction colour) -->
            <span
              class="absolute right-2 top-2 rounded-full bg-canvas/80 px-2 py-0.5 text-xs font-bold tabular-nums backdrop-blur"
              :style="{ color: trendColor(e.velocity) }"
              >{{ trendArrow(e.velocity) }} {{ vFmt(e.velocity) }}×</span
            >
            <!-- Bottom scrim with the sparkline drawn over the art (the "sparkling"). -->
            <div class="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/85 via-black/45 to-transparent px-2.5 pb-2 pt-8">
              <Sparkline
                :values="e.series"
                :width="132"
                :height="30"
                class="w-full drop-shadow"
                :style="{ color: trendColor(e.velocity) }"
              />
            </div>
          </div>
          <div class="mt-1.5 truncate text-sm font-bold text-canvas-foreground">{{ e.label }}</div>
        </RouterLink>
      </li>
    </CardRail>
  </section>
</template>
