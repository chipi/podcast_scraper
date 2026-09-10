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
import { computed, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import TrendingSparkChips from '../components/TrendingSparkChips.vue'
import TrendWindowTabs from '../components/TrendWindowTabs.vue'
import SectionStatus from '../components/SectionStatus.vue'
import {
  THEME_NEUTRAL,
  THEME_PALETTE,
  type RisingTopic,
  type TopicTheme,
} from '../components/trending'
import { getStorylines, getTrending, type TrendWindow } from '../services/api'
import { isArrayCache, readCached, writeCached } from '../services/contentCache'
import type { Storyline, TrendingEntity } from '../services/types'

// `embedded` — rendered as a tab panel inside the Browse hub: drop the page heading, the
// back-to-Home button and the outer page padding (the hub provides all three). Standalone (from
// Home) keeps them.
withDefaults(defineProps<{ embedded?: boolean }>(), { embedded: false })

const { t } = useI18n()
const router = useRouter()

const trending = ref<TrendingEntity[]>([])
const storylines = ref<Storyline[]>([])
// Top 10, expandable (BT.2) — the storyline list can run long; show the strongest ten with a toggle.
const STORYLINES_TOP = 10
const storylinesExpanded = ref(false)
const visibleStorylines = computed(() =>
  storylinesExpanded.value ? storylines.value : storylines.value.slice(0, STORYLINES_TOP),
)
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
  }))
)

// Give each chip a distinct hue + coloured sparkline, the way Home's trending topics read (the
// storyline palette). Home colours by storyline membership; the browse index has no membership to
// hand, so colour by velocity rank — every chip still gets its own colour, brightest-first.
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

function openTopic(id: string): void {
  void router.push({ name: 'topic', params: { id } })
}

// Storylines open their own full page (F4.5), keyed by the anchor topic id.
function openStoryline(s: Storyline): void {
  if (s.anchor_topic_id) void router.push({ name: 'storyline', params: { id: s.anchor_topic_id } })
}

// RFC-103 R2 — the trend window (1m/3m/6m/1y); default 3m. Changing it refetches trending only.
const window = ref<TrendWindow>('3m')
/**
 * `.catch(() => [])` collapsed a FAILURE into emptiness — the #1591 defect, which meant Browse →
 * Topics offline rendered as a corpus with no topics rather than as a page we could not load
 * (#1909). Cached per window: switching to 6m offline should not blank a 3m list we do have.
 */
const stale = ref(false)
async function loadTrending(): Promise<void> {
  const key = `browse.topics.${window.value}`
  try {
    const rows = await getTrending('topic', 'corpus', 50, window.value)
    trending.value = rows
    stale.value = false
    void writeCached(key, rows)
  } catch {
    const cached = await readCached<typeof trending.value>(key, isArrayCache)
    trending.value = cached ?? []
    stale.value = !!cached?.length
  }
}
watch(window, loadTrending)

onMounted(async () => {
  try {
    const [, stories] = await Promise.all([
      loadTrending(),
      getStorylines(24).catch(async () => (await readCached<typeof storylines.value>('browse.storylines', isArrayCache)) ?? []),
    ])
    storylines.value = stories
    if (stories.length) void writeCached('browse.storylines', stories)
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
    <!-- Standalone, never chained into a neighbouring v-if/v-else: slotting a notice into
         such a chain once made the final v-else (the content) unreachable. -->
    <p v-if="stale" class="mb-3 text-sm text-muted" data-testid="browse-stale-topics">
      {{ t('browse.stale') }}
    </p>
    <!-- F1.3: reserve the list shape while loading (no jump). A failed load falls back to cache or
         an empty section, never a hard error here (cache-fallback design, #1591). -->
    <SectionStatus v-if="loading" phase="loading" :rows="6" />
    <template v-else>
      <section class="mb-8">
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
          :collapse-at="10"
          :step="10"
          @open="openTopic"
        />
        <p v-else class="text-sm text-muted">{{ t('browse.trendingEmpty') }}</p>
      </section>

      <section v-if="storylines.length">
        <h2 class="mb-3 font-display text-lg font-bold text-canvas-foreground">
          {{ t('browse.storylines') }}
        </h2>
        <!-- BT.3: single-column rows with a theme swatch + label + count + chevron, so the storylines
             list reads as the same KIND of list as the topics above. Storylines carry no velocity or
             series, so they get the row chrome (not the sparkline) of the topic rows. -->
        <ul class="flex flex-col">
          <li v-for="(story, i) in visibleStorylines" :key="story.id">
            <button
              type="button"
              class="flex w-full items-center gap-3 border-b border-border py-2 text-left transition hover:bg-overlay"
              :title="story.label"
              data-testid="browse-storyline"
              @click="openStoryline(story)"
            >
              <span
                class="h-2.5 w-2.5 shrink-0 rounded-full"
                :style="{ backgroundColor: THEME_PALETTE[i % THEME_PALETTE.length] }"
                aria-hidden="true"
              />
              <span class="min-w-0 flex-1 truncate text-sm font-semibold text-surface-foreground">{{ story.label }}</span>
              <span class="lp-kicker shrink-0">{{ t('browse.topicCount', story.size) }}</span>
              <span class="shrink-0 text-muted" aria-hidden="true">›</span>
            </button>
          </li>
        </ul>
        <button
          v-if="storylines.length > STORYLINES_TOP"
          type="button"
          class="mt-2 px-2 py-1 text-xs font-semibold text-accent transition hover:opacity-80"
          data-testid="storylines-expand"
          :aria-expanded="storylinesExpanded"
          @click="storylinesExpanded = !storylinesExpanded"
        >
          {{
            storylinesExpanded
              ? t('home.showLess')
              : t('home.showMore', { count: storylines.length - STORYLINES_TOP })
          }}
        </button>
      </section>
    </template>
  </section>
</template>
