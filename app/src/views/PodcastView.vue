<script setup lang="ts">
/**
 * Per-podcast catalog view (PRD-038 FR2): one show's episodes, newest-first, paginated.
 * Header derives the show title + total from the first page (no separate feed endpoint in
 * the MVP). Cards reuse EpisodeCard.
 */
import { onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import EpisodeCard from '../components/EpisodeCard.vue'
import { getPodcasts, listPodcastEpisodes } from '../services/api'
import { showArtwork } from '../utils/episode'
import type { EpisodeSummary, Podcast } from '../services/types'

const PAGE_SIZE = 20
const props = defineProps<{ feedId: string }>()
const { t } = useI18n()

const episodes = ref<EpisodeSummary[]>([])
const total = ref(0)
const page = ref(0)
const hasMore = ref(false)
const loading = ref(false)
const error = ref(false)
const show = ref<Podcast | null>(null)
const descExpanded = ref(false)

const showArt = showArtwork
async function loadShow(): Promise<void> {
  const all = await getPodcasts().catch(() => [] as Podcast[])
  show.value = all.find((p) => p.feed_id === props.feedId) ?? null
}

async function loadMore(): Promise<void> {
  loading.value = true
  error.value = false
  try {
    const next = page.value + 1
    const res = await listPodcastEpisodes(props.feedId, { page: next, pageSize: PAGE_SIZE })
    episodes.value.push(...res.items)
    page.value = next
    total.value = res.total
    hasMore.value = res.has_more
  } catch {
    error.value = true
  } finally {
    loading.value = false
  }
}

function reset(): void {
  episodes.value = []
  page.value = 0
  total.value = 0
  hasMore.value = false
  show.value = null
  descExpanded.value = false
  void loadShow()
  void loadMore()
}

onMounted(() => {
  void loadShow()
  void loadMore()
})
watch(() => props.feedId, reset)
</script>

<template>
  <section>
    <RouterLink :to="{ name: 'catalog' }" class="lp-nav">‹ {{ t('nav.catalog') }}</RouterLink>

    <header class="mb-6 mt-2 flex gap-4 sm:gap-5">
      <img
        v-if="show && showArt(show)"
        :src="showArt(show)!"
        :alt="show.title ?? ''"
        class="h-20 w-20 shrink-0 rounded-xl bg-elevated object-cover sm:h-28 sm:w-28"
      />
      <div class="min-w-0 flex-1">
        <h1 class="font-display text-2xl font-extrabold leading-tight tracking-tight sm:text-3xl">
          {{ show?.title ?? episodes[0]?.podcast_title ?? feedId }}
        </h1>
        <p v-if="total" class="mt-1 text-sm text-muted">
          {{ t('podcast.episodeCount', { count: total }, total) }}
        </p>
        <p
          v-if="show?.description"
          class="mt-2 text-sm leading-relaxed text-muted"
          :class="descExpanded ? '' : 'line-clamp-3'"
        >
          {{ show.description }}
        </p>
        <button
          v-if="show?.description && show.description.length > 180"
          type="button"
          class="mt-1 text-xs font-bold text-accent"
          @click="descExpanded = !descExpanded"
        >
          {{ descExpanded ? t('podcast.showLess') : t('podcast.showMore') }}
        </button>
      </div>
    </header>

    <p v-if="loading && episodes.length === 0" class="text-muted">{{ t('catalog.loading') }}</p>
    <p v-else-if="error && episodes.length === 0" class="text-danger">{{ t('catalog.loadError') }}</p>
    <p v-else-if="episodes.length === 0" class="text-muted">{{ t('catalog.empty') }}</p>

    <div v-else>
      <EpisodeCard v-for="ep in episodes" :key="ep.slug" :episode="ep" />
      <div class="mt-6 flex justify-center">
        <button
          v-if="hasMore"
          type="button"
          :disabled="loading"
          class="rounded-full border border-border px-5 py-2 font-bold disabled:opacity-50"
          @click="loadMore"
        >
          {{ loading ? t('catalog.loading') : t('catalog.loadMore') }}
        </button>
      </div>
    </div>
  </section>
</template>
