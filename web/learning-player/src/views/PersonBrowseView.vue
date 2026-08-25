<script setup lang="ts">
/**
 * Person browse index (#1261-6) — discovery beyond the search bar. Trending
 * people (hosts, guests, mentioned) linking to standalone Person pages
 * (`/person/:id`).
 *
 * The trending rail shares Home's TrendingSparkChips treatment (#12): a mini
 * sparkline of each person's monthly shape, sorted hottest-first, collapsed to
 * the top few with a show-more — not a flat, unsorted, ellipsised chip grid.
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import TrendingSparkChips from '../components/TrendingSparkChips.vue'
import type { RisingTopic } from '../components/trending'
import { getTrending } from '../services/api'
import type { TrendingEntity } from '../services/types'

const { t } = useI18n()
const router = useRouter()

const trending = ref<TrendingEntity[]>([])
const loading = ref(true)

// TrendingEntity → the RisingTopic shape TrendingSparkChips renders. The component sorts
// hottest-first and collapses to the top few itself; here we only reshape.
const trendingRows = computed<RisingTopic[]>(() =>
  trending.value.map((e) => ({
    id: e.entity_id,
    label: e.label,
    v: Math.round((e.velocity ?? 0) * 10) / 10,
    total: e.total,
    series: e.series ?? [],
  })),
)

function openPerson(id: string): void {
  void router.push({ name: 'person', params: { id } })
}

onMounted(async () => {
  try {
    trending.value = await getTrending('person', 'corpus', 36).catch(() => [])
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <section class="mx-auto max-w-3xl px-4 pb-8 pt-4" data-testid="person-browse-view">
    <RouterLink
      :to="{ name: 'home' }"
      class="mb-4 inline-flex items-center gap-1 rounded-full border border-border bg-surface px-4 py-2 text-sm font-bold text-canvas-foreground transition hover:bg-overlay"
      data-testid="browse-back-home"
    >
      ‹ {{ t('browse.backHome') }}
    </RouterLink>
    <h1 class="mb-4 font-display text-3xl font-extrabold tracking-tight">
      {{ t('browse.peopleTitle') }}
    </h1>
    <p v-if="loading" class="text-muted">{{ t('browse.loading') }}</p>
    <template v-else>
      <section v-if="trendingRows.length">
        <h2 class="mb-3 font-display text-lg font-bold text-canvas-foreground">
          {{ t('browse.trending') }}
        </h2>
        <TrendingSparkChips :topics="trendingRows" @open="openPerson" />
      </section>
      <p v-else class="text-muted">{{ t('browse.empty') }}</p>
    </template>
  </section>
</template>
