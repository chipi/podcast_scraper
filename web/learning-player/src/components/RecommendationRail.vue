<script setup lang="ts">
/**
 * One Discover rail of recommendations (ADR-163) — the shape every rail shares: episodes, each
 * card led by WHY it is here (`RecommendationWhy`: the reason line, the quote that grounds it,
 * "▶ Play from"), measured in Umami through `useRailTelemetry(rail)`.
 *
 * The rail supplies its id, its title and how to load its cards; everything else is the same for
 * every rail, so a new rail cannot drift in how it explains itself or how it is measured.
 *
 * - The card opens the episode; "Play from" opens it playing at the quote (ADR-163 §8).
 * - A rail with nothing to show hides itself (thin data is not padded out).
 * - `reloadKey` changing (e.g. the page scope) reloads it and counts a fresh impression.
 * - Discover is kept alive, so returning to it reloads too — a highlight made a minute ago must be
 *   able to seed "More from what you captured" without an app restart.
 */
import { onActivated, onMounted, ref, watch } from 'vue'
import type { TrendingEpisodeCard } from '../services/api'
import type { RailId } from '../services/analytics'
import { episodeArtwork } from '../utils/episode'
import { useRailTelemetry } from '../composables/useRailTelemetry'
import RecommendationWhy from './RecommendationWhy.vue'
import SectionHeading from './SectionHeading.vue'

const props = defineProps<{
  rail: RailId
  title: string
  load: () => Promise<TrendingEpisodeCard[]>
  /** Telemetry scope of what this rail shows. */
  scope: 'mine' | 'everyone'
  /** The rail's own testids, passed as literals so the surface map can name them. */
  cardsTestid: string
  cardTestid: string
  openTestid: string
  reloadKey?: string
}>()

const cards = ref<TrendingEpisodeCard[]>([])
let telemetry = useRailTelemetry(props.rail)

async function fetchCards(): Promise<void> {
  cards.value = await props.load().catch(() => [])
  telemetry.shown(cards.value, props.scope)
}
function reload(): void {
  telemetry = useRailTelemetry(props.rail) // a new visit or scope is a new rail impression
  void fetchCards()
}
onMounted(fetchCards)
let mounted = false
onActivated(() => {
  if (mounted) reload() // the first activation is the mount itself
  mounted = true
})
watch(() => props.reloadKey, reload)
</script>

<template>
  <section v-if="cards.length" class="mt-7" :data-testid="cardsTestid">
    <SectionHeading :title="title" />
    <ul class="grid gap-3 lg:grid-cols-2">
      <li v-for="(c, i) in cards" :key="c.episode.slug">
        <article class="flex h-full flex-col gap-3 rounded-xl border border-border p-4" :data-testid="cardTestid">
          <RouterLink
            :to="{ name: 'player', params: { slug: c.episode.slug } }"
            class="lp-tap flex min-w-0 items-center gap-3"
            :data-testid="openTestid"
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
