<script setup lang="ts">
/**
 * Show browse index — every show in the corpus as a square-artwork grid (ShowTile), so you can page
 * through the catalogue by show, not just by episode. A Browse-hub tab (embedded) and a standalone
 * route reached from Home/Library. Tiles are followable so you can follow while browsing.
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import ShowTile from '../components/ShowTile.vue'
import { getPodcasts } from '../services/api'
import type { Podcast } from '../services/types'

// `embedded` — rendered as a tab panel inside the Browse hub (drops heading/back-Home/padding).
withDefaults(defineProps<{ embedded?: boolean }>(), { embedded: false })

const { t } = useI18n()

const shows = ref<Podcast[]>([])
const loading = ref(true)
const error = ref(false)

// Filter + sort so the grid stays browsable as the catalogue grows.
const search = ref('')
const sort = ref<'az' | 'episodes'>('az')
const titleOf = (s: Podcast) => s.title ?? s.feed_id

const visible = computed(() => {
  let list = shows.value.filter((s) => s.feed_id)
  const q = search.value.trim().toLowerCase()
  if (q) list = list.filter((s) => titleOf(s).toLowerCase().includes(q))
  return [...list].sort((a, b) =>
    sort.value === 'episodes'
      ? b.episode_count - a.episode_count || titleOf(a).localeCompare(titleOf(b))
      : titleOf(a).localeCompare(titleOf(b)),
  )
})

onMounted(async () => {
  try {
    shows.value = await getPodcasts()
  } catch {
    error.value = true
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <section
    :class="embedded ? '' : 'mx-auto max-w-3xl px-4 pb-8 pt-4'"
    data-testid="show-browse-view"
  >
    <RouterLink
      v-if="!embedded"
      :to="{ name: 'home' }"
      class="mb-4 inline-flex items-center gap-1 rounded-full border border-border bg-surface px-4 py-2 text-sm font-bold text-canvas-foreground transition hover:bg-overlay"
      data-testid="browse-back-home"
    >
      ‹ {{ t('browse.backHome') }}
    </RouterLink>
    <h1 v-if="!embedded" class="mb-4 font-display text-3xl font-extrabold tracking-tight">
      {{ t('browse.shows') }}
    </h1>

    <p v-if="loading" class="text-muted">{{ t('browse.loading') }}</p>
    <p v-else-if="error" class="text-danger">{{ t('browse.empty') }}</p>
    <template v-else-if="shows.length">
      <!-- Filter + sort — shows grow, so keep the grid searchable + orderable. -->
      <div class="mb-4 flex flex-wrap items-center gap-2">
        <input
          v-model="search"
          type="search"
          :placeholder="t('browse.filterShows')"
          class="min-w-0 flex-1 rounded-full border border-border bg-surface px-4 py-2 text-sm text-canvas-foreground outline-none focus:border-accent"
          data-testid="show-browse-search"
        />
        <select
          v-model="sort"
          class="shrink-0 rounded-full border border-border bg-surface px-3 py-2 text-sm font-semibold text-canvas-foreground outline-none focus:border-accent"
          data-testid="show-browse-sort"
        >
          <option value="az">{{ t('browse.sortShowsAZ') }}</option>
          <option value="episodes">{{ t('browse.sortShowsEpisodes') }}</option>
        </select>
      </div>
      <ul v-if="visible.length" class="grid grid-cols-3 gap-3 sm:grid-cols-4" data-testid="show-browse-grid">
        <li v-for="p in visible" :key="p.feed_id"><ShowTile :show="p" followable /></li>
      </ul>
      <p v-else class="text-muted">{{ t('browse.noShowMatches') }}</p>
    </template>
    <p v-else class="text-muted">{{ t('browse.empty') }}</p>
  </section>
</template>
