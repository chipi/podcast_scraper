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
import { computed, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import TrendingSparkChips from '../components/TrendingSparkChips.vue'
import TrendWindowTabs from '../components/TrendWindowTabs.vue'
import {
  THEME_NEUTRAL,
  THEME_PALETTE,
  type RisingTopic,
  type TopicTheme,
} from '../components/trending'
import { getTrending, type TrendWindow } from '../services/api'
import type { TrendingEntity } from '../services/types'

// `embedded` — rendered as a tab panel inside the Browse hub: drop the page heading, the
// back-to-Home button and the outer page padding (the hub provides all three). Standalone (from
// Home) keeps them.
withDefaults(defineProps<{ embedded?: boolean }>(), { embedded: false })

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
    role: e.role ?? null, // host/guest/mentioned badge (#people role tags)
  }))
)

// Each chip gets a distinct hue + coloured sparkline, matching Home's trending treatment — coloured
// by velocity rank (people have no storyline membership to colour by), brightest-first.
const trendingTheme = computed<Record<string, TopicTheme>>(() => {
  const ranked = [...trending.value].sort(
    (a, b) => (b.velocity ?? 0) - (a.velocity ?? 0) || (b.total ?? 0) - (a.total ?? 0)
  )
  const map: Record<string, TopicTheme> = {}
  ranked.forEach((e, i) => {
    map[e.entity_id] = { color: THEME_PALETTE[i % THEME_PALETTE.length], label: null, group: i }
  })
  return map
})

function openPerson(id: string): void {
  void router.push({ name: 'person', params: { id } })
}

// RFC-103 R2 — the trend window (1m/3m/6m/1y); default 3m. Changing it refetches.
const window = ref<TrendWindow>('3m')
async function loadTrending(): Promise<void> {
  trending.value = await getTrending('person', 'corpus', 50, window.value).catch(() => [])
}
watch(window, loadTrending)

onMounted(async () => {
  try {
    await loadTrending()
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <section
    :class="embedded ? '' : 'mx-auto max-w-3xl px-4 pb-8 pt-4'"
    data-testid="person-browse-view"
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
      {{ t('browse.peopleTitle') }}
    </h1>
    <p v-if="loading" class="text-muted">{{ t('browse.loading') }}</p>
    <template v-else>
      <section>
        <div class="mb-3 flex items-center justify-between gap-2">
          <h2 class="font-display text-lg font-bold text-canvas-foreground">
            {{ t('browse.trending') }}
          </h2>
          <TrendWindowTabs v-model="window" />
        </div>
        <TrendingSparkChips
          v-if="trendingRows.length"
          :topics="trendingRows"
          :topic-theme="trendingTheme"
          :neutral-color="THEME_NEUTRAL"
          :collapse-at="20"
          @open="openPerson"
        />
        <p v-else class="text-sm text-muted">{{ t('browse.trendingEmpty') }}</p>
      </section>
    </template>
  </section>
</template>
