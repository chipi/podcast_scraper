<script setup lang="ts">
/**
 * Catalog — global all-episodes view (PRD-038 FR1). Newest-first, paginated with a "load more"
 * control (20/page). A shared ListToolbar (UXS-014) filters/sorts/searches the list; engaging any
 * control auto-loads the remaining pages so the controls cover the whole catalog, not just what's
 * been paged in.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import EpisodeCard from '../components/EpisodeCard.vue'
import ListToolbar from '../components/ListToolbar.vue'
import { getPodcasts, listEpisodes } from '../services/api'
import type { EpisodeSummary } from '../services/types'

const PAGE_SIZE = 20
const { t } = useI18n()
const episodes = ref<EpisodeSummary[]>([])
const page = ref(0)
const hasMore = ref(false)
const loading = ref(false)
const error = ref(false)

const search = ref('')
const sort = ref('newest')
const filter = ref('all')
const show = ref('')
const shows = ref<{ id: string; label: string }[]>([])
const controlsActive = computed(
  () =>
    search.value.trim() !== '' ||
    sort.value !== 'newest' ||
    filter.value !== 'all' ||
    show.value !== '',
)

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

// Filtering/sorting only makes sense over the whole list — pull the rest in when a control is used.
async function loadAll(): Promise<void> {
  while (hasMore.value && !error.value) await loadMore()
}
watch(controlsActive, (active) => {
  if (active && hasMore.value) void loadAll()
})

const visible = computed<EpisodeSummary[]>(() => {
  let list = episodes.value
  const q = search.value.trim().toLowerCase()
  if (q) {
    list = list.filter(
      (e) =>
        e.title.toLowerCase().includes(q) ||
        (e.podcast_title ?? '').toLowerCase().includes(q),
    )
  }
  if (filter.value === 'insights') list = list.filter((e) => e.has_gi)
  if (show.value) list = list.filter((e) => e.feed_id === show.value)
  const byDate = (e: EpisodeSummary) => e.publish_date ?? ''
  const sorted = [...list]
  if (sort.value === 'newest') sorted.sort((a, b) => byDate(b).localeCompare(byDate(a)))
  else if (sort.value === 'oldest') sorted.sort((a, b) => byDate(a).localeCompare(byDate(b)))
  else if (sort.value === 'title') sorted.sort((a, b) => a.title.localeCompare(b.title))
  return sorted
})

const countLabel = computed(() =>
  controlsActive.value ? t('list.count', { shown: visible.value.length, total: episodes.value.length }) : '',
)

onMounted(async () => {
  await loadMore()
  shows.value = (await getPodcasts().catch(() => []))
    .filter((p) => p.feed_id)
    .map((p) => ({ id: p.feed_id, label: p.title ?? p.feed_id }))
})
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
      <ListToolbar
        v-model:search="search"
        v-model:sort="sort"
        v-model:filter="filter"
        v-model:show="show"
        :shows="shows"
        :count="countLabel"
      />

      <p v-if="visible.length === 0" class="text-muted">{{ t('list.noMatches') }}</p>
      <EpisodeCard v-for="ep in visible" :key="ep.slug" :episode="ep" />

      <div class="mt-6 flex justify-center">
        <button
          v-if="hasMore && !controlsActive"
          type="button"
          :disabled="loading"
          class="rounded-full border border-border px-5 py-2 font-bold disabled:opacity-50"
          @click="loadMore"
        >
          {{ loading ? t('catalog.loading') : t('catalog.loadMore') }}
        </button>
        <p v-else-if="loading && controlsActive" class="text-sm text-muted">{{ t('catalog.loading') }}</p>
      </div>
    </div>
  </section>
</template>
