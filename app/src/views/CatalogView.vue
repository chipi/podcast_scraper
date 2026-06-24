<script setup lang="ts">
/**
 * Catalog (PRD-038) — minimal editorial-bold list scaffold. Fetches the local-corpus
 * episode list and renders cards that link into the Player. Full card enrichment + the
 * podcast view land in C3 (#1082); this proves the API wiring + the visual baseline.
 */
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { listEpisodes } from '../services/api'
import type { EpisodeSummary } from '../services/types'

const { t } = useI18n()
const episodes = ref<EpisodeSummary[]>([])
const loading = ref(true)
const error = ref(false)

onMounted(async () => {
  try {
    episodes.value = (await listEpisodes({ pageSize: 20 })).items
  } catch {
    error.value = true
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <section>
    <h1 class="font-display text-3xl font-extrabold tracking-tight mb-5">
      {{ t('catalog.heading') }}
    </h1>

    <p v-if="loading" class="text-muted">{{ t('catalog.loading') }}</p>
    <p v-else-if="error" class="text-danger">{{ t('catalog.loadError') }}</p>
    <p v-else-if="episodes.length === 0" class="text-muted">{{ t('catalog.empty') }}</p>

    <ul v-else class="flex flex-col gap-px">
      <li v-for="ep in episodes" :key="ep.slug">
        <RouterLink
          :to="{ name: 'player', params: { slug: ep.slug } }"
          class="block border-b border-border py-4 no-underline text-canvas-foreground hover:bg-overlay -mx-2 px-2 rounded"
        >
          <div class="flex items-baseline justify-between gap-3">
            <span class="font-display text-lg font-bold leading-snug">{{ ep.title }}</span>
            <span class="lp-kicker shrink-0">{{ ep.podcast_title }}</span>
          </div>
          <p v-if="ep.summary_preview" class="text-muted text-sm mt-1 line-clamp-2">
            {{ ep.summary_preview }}
          </p>
          <div class="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted">
            <span v-if="ep.publish_date">{{ ep.publish_date }}</span>
            <span v-if="ep.has_gi" class="text-grounded">● {{ t('catalog.insightsBadge') }}</span>
            <span
              class="rounded-full px-2 py-0.5"
              :class="ep.status === 'ready' ? 'text-grounded' : 'text-warning'"
            >
              {{ ep.status === 'ready' ? t('status.ready') : t('status.pending') }}
            </span>
          </div>
        </RouterLink>
      </li>
    </ul>
  </section>
</template>
