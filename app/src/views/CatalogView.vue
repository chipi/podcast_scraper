<script setup lang="ts">
/**
 * Catalog — global all-episodes view (PRD-038 FR1). Newest-first, paginated with a
 * "load more" control (20/page). Cards degrade cleanly via EpisodeCard.
 */
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import EpisodeCard from '../components/EpisodeCard.vue'
import { listEpisodes } from '../services/api'
import type { EpisodeSummary } from '../services/types'

const PAGE_SIZE = 20
const { t } = useI18n()
const episodes = ref<EpisodeSummary[]>([])
const page = ref(0)
const hasMore = ref(false)
const loading = ref(false)
const error = ref(false)

async function loadMore(): Promise<void> {
  loading.value = true
  error.value = false
  try {
    const next = page.value + 1
    const res = await listEpisodes({ page: next, pageSize: PAGE_SIZE })
    episodes.value.push(...res.items)
    page.value = next
    hasMore.value = res.has_more
  } catch {
    error.value = true
  } finally {
    loading.value = false
  }
}

onMounted(loadMore)
</script>

<template>
  <section>
    <h1 class="mb-5 font-display text-3xl font-extrabold tracking-tight">
      {{ t('catalog.heading') }}
    </h1>

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
