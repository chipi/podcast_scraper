<script setup lang="ts">
/**
 * Topic browse index (#1261-6) — discovery beyond the search bar. Trending
 * topics up top, storylines (co-occurrence theme clusters) below. Each row
 * links to the corresponding standalone Topic page (`/topic/:id`) — no
 * palette, no modal, native mobile navigation.
 *
 * Both rails read existing endpoints (``/api/app/trending`` +
 * ``/api/app/theme-clusters``); silent empty on error.
 */
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { getStorylines, getTrending } from '../services/api'
import type { Storyline, TrendingEntity } from '../services/types'

const { t } = useI18n()

const trending = ref<TrendingEntity[]>([])
const storylines = ref<Storyline[]>([])
const loading = ref(true)

onMounted(async () => {
  try {
    const [top, stories] = await Promise.all([
      getTrending('topic', 'corpus', 24).catch(() => []),
      getStorylines(24).catch(() => []),
    ])
    trending.value = top
    storylines.value = stories
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <section class="mx-auto max-w-3xl px-4 pb-8 pt-4" data-testid="topic-browse-view">
    <h1 class="mb-4 font-display text-3xl font-extrabold tracking-tight">
      {{ t('browse.topicsTitle') }}
    </h1>
    <p v-if="loading" class="text-muted">{{ t('browse.loading') }}</p>
    <template v-else>
      <section v-if="trending.length" class="mb-8">
        <h2 class="mb-3 font-display text-lg font-bold text-canvas-foreground">
          {{ t('browse.trending') }}
        </h2>
        <ul class="grid grid-cols-2 gap-2 sm:grid-cols-3">
          <li v-for="ent in trending" :key="ent.entity_id">
            <RouterLink
              :to="{ name: 'topic', params: { id: ent.entity_id } }"
              class="block rounded-xl border border-border bg-surface px-3 py-2.5 text-sm font-semibold text-canvas-foreground transition hover:bg-overlay"
            >
              {{ ent.label }}
            </RouterLink>
          </li>
        </ul>
      </section>

      <section v-if="storylines.length">
        <h2 class="mb-3 font-display text-lg font-bold text-canvas-foreground">
          {{ t('browse.storylines') }}
        </h2>
        <ul class="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <li v-for="story in storylines" :key="story.id">
            <RouterLink
              :to="{ name: 'topic', params: { id: story.anchor_topic_id } }"
              class="block rounded-xl border border-border bg-surface px-3 py-2.5 text-sm font-semibold text-canvas-foreground transition hover:bg-overlay"
            >
              {{ story.label }}
              <span class="lp-kicker ml-1 text-xs font-normal">
                {{ t('browse.topicCount', story.size) }}
              </span>
            </RouterLink>
          </li>
        </ul>
      </section>

      <p v-if="!trending.length && !storylines.length" class="text-muted">
        {{ t('browse.empty') }}
      </p>
    </template>
  </section>
</template>
