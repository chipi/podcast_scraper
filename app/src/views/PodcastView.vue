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
import { listPodcastEpisodes } from '../services/api'
import type { EpisodeSummary } from '../services/types'

const PAGE_SIZE = 20
const props = defineProps<{ feedId: string }>()
const { t } = useI18n()

const episodes = ref<EpisodeSummary[]>([])
const total = ref(0)
const page = ref(0)
const hasMore = ref(false)
const loading = ref(false)
const error = ref(false)

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
  void loadMore()
}

onMounted(loadMore)
watch(() => props.feedId, reset)
</script>

<template>
  <section>
    <RouterLink :to="{ name: 'catalog' }" class="lp-kicker no-underline">‹ {{ t('nav.catalog') }}</RouterLink>

    <header class="mb-5 mt-2">
      <h1 class="font-display text-3xl font-extrabold tracking-tight">
        {{ episodes[0]?.podcast_title ?? feedId }}
      </h1>
      <p v-if="total" class="mt-1 text-sm text-muted">
        {{ t('podcast.episodeCount', { count: total }, total) }}
      </p>
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
