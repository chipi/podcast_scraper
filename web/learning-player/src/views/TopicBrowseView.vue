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
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import TrendingSparkChips from '../components/TrendingSparkChips.vue'
import type { RisingTopic } from '../components/trending'
import { getStorylines, getTrending } from '../services/api'
import type { Storyline, TrendingEntity } from '../services/types'

// `embedded` — rendered as a tab panel inside the Browse hub: drop the page heading, the
// back-to-Home button and the outer page padding (the hub provides all three). Standalone (from
// Home) keeps them.
withDefaults(defineProps<{ embedded?: boolean }>(), { embedded: false })

const { t } = useI18n()
const router = useRouter()

const trending = ref<TrendingEntity[]>([])
const storylines = ref<Storyline[]>([])
const loading = ref(true)

// TrendingEntity → the RisingTopic shape TrendingSparkChips renders (sparkline + ×velocity, sorted
// hottest-first, collapsed to the top few) — matching Home's trending treatment (#11).
const trendingRows = computed<RisingTopic[]>(() =>
  trending.value.map((e) => ({
    id: e.entity_id,
    label: e.label,
    v: Math.round((e.velocity ?? 0) * 10) / 10,
    total: e.total,
    series: e.series ?? [],
  })),
)

function openTopic(id: string): void {
  void router.push({ name: 'topic', params: { id } })
}

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
  <section
    :class="embedded ? '' : 'mx-auto max-w-3xl px-4 pb-8 pt-4'"
    data-testid="topic-browse-view"
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
      {{ t('browse.topicsTitle') }}
    </h1>
    <p v-if="loading" class="text-muted">{{ t('browse.loading') }}</p>
    <template v-else>
      <section v-if="trendingRows.length" class="mb-8">
        <h2 class="mb-3 font-display text-lg font-bold text-canvas-foreground">
          {{ t('browse.trending') }}
        </h2>
        <TrendingSparkChips :topics="trendingRows" @open="openTopic" />
      </section>

      <section v-if="storylines.length">
        <h2 class="mb-3 font-display text-lg font-bold text-canvas-foreground">
          {{ t('browse.storylines') }}
        </h2>
        <ul class="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <li v-for="story in storylines" :key="story.id">
            <RouterLink
              :to="{ name: 'topic', params: { id: story.anchor_topic_id } }"
              class="block rounded-xl border border-border bg-surface px-3 py-2.5 text-sm truncate font-semibold text-canvas-foreground transition hover:bg-overlay"
              :title="story.label"
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
