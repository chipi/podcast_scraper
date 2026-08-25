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

// Alphabetical so the grid reads as a browsable catalogue; shows with a feed_id only (real feeds).
const sorted = computed(() =>
  [...shows.value]
    .filter((s) => s.feed_id)
    .sort((a, b) => (a.title ?? a.feed_id).localeCompare(b.title ?? b.feed_id)),
)

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
    <ul v-else-if="sorted.length" class="grid grid-cols-3 gap-3 sm:grid-cols-4" data-testid="show-browse-grid">
      <li v-for="p in sorted" :key="p.feed_id"><ShowTile :show="p" followable /></li>
    </ul>
    <p v-else class="text-muted">{{ t('browse.empty') }}</p>
  </section>
</template>
