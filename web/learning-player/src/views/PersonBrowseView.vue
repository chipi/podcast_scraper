<script setup lang="ts">
/**
 * Person browse index (#1261-6) — discovery beyond the search bar. Trending
 * people (hosts, guests, mentioned) linking to standalone Person pages
 * (`/person/:id`).
 */
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { getTrending } from '../services/api'
import type { TrendingEntity } from '../services/types'

const { t } = useI18n()

const trending = ref<TrendingEntity[]>([])
const loading = ref(true)

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
    <h1 class="mb-4 font-display text-3xl font-extrabold tracking-tight">
      {{ t('browse.peopleTitle') }}
    </h1>
    <p v-if="loading" class="text-muted">{{ t('browse.loading') }}</p>
    <template v-else>
      <section v-if="trending.length">
        <h2 class="mb-3 font-display text-lg font-bold text-canvas-foreground">
          {{ t('browse.trending') }}
        </h2>
        <ul class="grid grid-cols-2 gap-2 sm:grid-cols-3">
          <li v-for="ent in trending" :key="ent.entity_id">
            <RouterLink
              :to="{ name: 'person', params: { id: ent.entity_id } }"
              class="block rounded-xl border border-border bg-surface px-3 py-2.5 text-sm font-semibold text-canvas-foreground transition hover:bg-overlay"
            >
              {{ ent.label }}
            </RouterLink>
          </li>
        </ul>
      </section>
      <p v-else class="text-muted">{{ t('browse.empty') }}</p>
    </template>
  </section>
</template>
