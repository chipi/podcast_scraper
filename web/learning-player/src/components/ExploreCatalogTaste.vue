<script setup lang="ts">
/**
 * A taste of the catalogue on Explore (operator 2026-10-11: "Trends + a catalog taste"): the newest
 * few episodes or shows, one server page, and "See all" to the full list — `/catalog` for episodes,
 * `/shows` for shows — where the toolbar's search, sort and filters (including Mine) live. Explore
 * stays a page to look around, not a scroll through the whole catalogue.
 */
import { onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import EpisodeCard from './EpisodeCard.vue'
import SectionStatus from './SectionStatus.vue'
import ShowTile from './ShowTile.vue'
import { useSectionState } from '../composables/useSectionState'
import { getPodcastsPage, listEpisodes } from '../services/api'
import type { EpisodeSummary, Podcast } from '../services/types'

const props = defineProps<{ kind: 'episodes' | 'shows' }>()
const { t } = useI18n()

/** Episodes in the taste; shows fill two rows of three tiles. */
const EPISODES = 4
const SHOWS = 6

interface Taste {
  episodes: EpisodeSummary[]
  shows: Podcast[]
  total: number
}
const section = useSectionState<Taste>({ episodes: [], shows: [], total: 0 })

async function fetchTaste(): Promise<Taste> {
  if (props.kind === 'episodes') {
    const page = await listEpisodes({ page: 1, pageSize: EPISODES })
    return { episodes: page.items, shows: [], total: page.total }
  }
  const page = await getPodcastsPage({ limit: SHOWS, sort: 'newest', compact: true })
  return { episodes: [], shows: page.items, total: page.total ?? page.items.length }
}
function load(): Promise<void> {
  return section.load(fetchTaste)
}
onMounted(load)
</script>

<template>
  <div :data-testid="`explore-taste-${kind}`">
    <SectionStatus :phase="section.phase.value" :rows="3" @retry="load" />
    <div v-if="kind === 'episodes' && section.data.value.episodes.length" class="flex flex-col gap-3">
      <EpisodeCard v-for="ep in section.data.value.episodes" :key="ep.slug" :episode="ep" moments-link />
    </div>
    <ul v-if="kind === 'shows' && section.data.value.shows.length" class="grid grid-cols-3 gap-3 sm:grid-cols-6">
      <li v-for="p in section.data.value.shows" :key="p.feed_id"><ShowTile :show="p" followable /></li>
    </ul>
    <RouterLink
      v-if="section.data.value.total"
      :to="{ name: kind === 'episodes' ? 'catalog' : 'shows' }"
      class="lp-tap mt-4 inline-flex min-h-[44px] items-center text-sm font-bold text-accent no-underline"
      :data-testid="`explore-see-all-${kind}`"
    >{{ kind === 'episodes' ? t('explore.seeAllEpisodes', { count: section.data.value.total }) : t('explore.seeAllShows', { count: section.data.value.total }) }} ›</RouterLink>
  </div>
</template>
