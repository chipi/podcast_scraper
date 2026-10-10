<script setup lang="ts">
/**
 * Discover's first rail (ADR-163; Discover arc Step 2.1): episodes worth hearing next, each led by
 * the quote that grounds it and WHY it is here (`RecommendationWhy`). Rising topics and the most
 * engaged episodes (`GET /trending/episodes`), in the page's Mine ⇄ Everyone scope; under Mine,
 * every fifth card is the labelled outside-your-usual share (server-side, ADR-163 §7).
 *
 * The card opens the episode; "Play from" opens it playing at the quote (ADR-163 §8). Both report
 * to Umami through `useRailTelemetry('quotes')`, which also credits the rail with what the listener
 * captures in that visit. A rail with nothing to show hides itself (thin data is not padded out).
 */
import { onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { getTrendingEpisodes, type TrendingEpisodeCard } from '../services/api'
import { episodeArtwork } from '../utils/episode'
import { useRailTelemetry } from '../composables/useRailTelemetry'
import RecommendationWhy from './RecommendationWhy.vue'
import SectionHeading from './SectionHeading.vue'

const props = withDefaults(defineProps<{ scope: 'mine' | 'corpus'; limit?: number }>(), { limit: 6 })
const { t } = useI18n()

const cards = ref<TrendingEpisodeCard[]>([])
let telemetry = useRailTelemetry('quotes')

async function load(): Promise<void> {
  cards.value = await getTrendingEpisodes(props.scope, props.limit).catch(() => [])
  telemetry.shown(cards.value, props.scope === 'mine' ? 'mine' : 'everyone')
}
onMounted(load)
watch(
  () => props.scope,
  () => {
    telemetry = useRailTelemetry('quotes') // a new scope is a new rail impression
    void load()
  },
)
</script>

<template>
  <section v-if="cards.length" class="mt-7" data-testid="quote-cards">
    <SectionHeading :title="t('discover.quotes.title')" />
    <ul class="grid gap-3 lg:grid-cols-2">
      <li v-for="(c, i) in cards" :key="c.episode.slug">
        <article class="flex h-full flex-col gap-3 rounded-xl border border-border p-4" data-testid="quote-card">
          <RouterLink
            :to="{ name: 'player', params: { slug: c.episode.slug } }"
            class="lp-tap flex min-w-0 items-center gap-3"
            data-testid="quote-card-open"
            @click="telemetry.tapped(c, i, 'open')"
          >
            <img
              v-if="episodeArtwork(c.episode)"
              :src="episodeArtwork(c.episode) ?? ''"
              alt=""
              class="h-10 w-10 shrink-0 rounded-md object-cover"
            />
            <span class="min-w-0">
              <span class="line-clamp-2 text-sm font-bold">{{ c.episode.title }}</span>
              <span class="block break-words text-xs text-muted">{{ c.episode.podcast_title }}</span>
            </span>
          </RouterLink>
          <RecommendationWhy :why="c.why" @play="telemetry.tapped(c, i, 'play_from')" />
        </article>
      </li>
    </ul>
  </section>
</template>
