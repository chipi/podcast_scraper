<script setup lang="ts">
/**
 * Explore (ADR-163 §1, operator 2026-10-11) — everything for looking around, moved off Discover so
 * Discover keeps one job ("what to hear next"):
 *
 * - **Trends** — rising topics, themes, storylines and people (`TrendsSection`), in the shared
 *   Mine ⇄ Everyone scope. `?trends=<kind>` selects a kind and scrolls to it.
 * - **The catalog** — Episodes · Shows (`CatalogView` / `ShowBrowseView`, embedded), anchored at
 *   `#catalog`. `?tab=episodes|shows` picks the tab; `?from=&state=` reach the episode list
 *   (What's new's "all ›"); `?sort=trending&view=list` is trending shows' "See all" from Discover.
 *
 * A panel mounts the first time its tab opens, then stays (v-show): switching back never refetches.
 */
import { computed, nextTick, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
defineOptions({ name: 'ExploreView' })
import Tabs from '../components/Tabs.vue'
import { toRankBucket, track } from '../services/analytics'
import { panelAttrs, type TabSpec } from '../components/tabs'
import CatalogView from './CatalogView.vue'
import ShowBrowseView from './ShowBrowseView.vue'
import { useVisitedTabs } from '../composables/useVisitedTabs'
import TrendsSection from '../components/TrendsSection.vue'
import TrendingScopeButton from '../components/TrendingScopeButton.vue'
import { scrollBehavior } from '../utils/motion'

const { t } = useI18n()
const route = useRoute()
const router = useRouter()

type Kind = 'topic' | 'theme' | 'storyline' | 'person'
type Tab = 'episodes' | 'shows'

function onEntityOpen(p: { kind: Kind; id: string; rank: number }): void {
  const name =
    p.kind === 'topic' ? 'topic' : p.kind === 'person' ? 'person' : p.kind === 'theme' ? 'theme' : 'storyline'
  track('trends_row_click', { kind: p.kind, rank: toRankBucket(p.rank) })
  track('entity_open', { kind: p.kind, presentation: 'page', source: 'explore_trends' })
  void router.push({ name, params: { id: p.id } })
}

const TRENDS_KINDS = ['topic', 'theme', 'storyline', 'person'] as const
const trendsKind = computed(() => {
  const q = String(route.query.trends || '')
  return (TRENDS_KINDS as readonly string[]).includes(q) ? (q as Kind) : undefined
})
// `?trends=<kind>` also scrolls Trends into view: selecting the tab alone is invisible below the
// fold (operator 2026-09-18). Declared after `trends`/`trendsKind` (an immediate watch above the
// consts it reads throws a TDZ ReferenceError at setup).
const trends = ref<{ trendsEl: HTMLElement | null } | null>(null)
watch(
  trendsKind,
  (k) => {
    if (k) void nextTick(() => trends.value?.trendsEl?.scrollIntoView({ behavior: scrollBehavior(), block: 'start' }))
  },
  { immediate: true },
)

const TAB_KEYS: { key: Tab; labelKey: string }[] = [
  { key: 'episodes', labelKey: 'browse.episodes' },
  { key: 'shows', labelKey: 'browse.shows' },
]
const tabs = computed<TabSpec<Tab>[]>(() =>
  TAB_KEYS.map((tb) => ({ key: tb.key, label: t(tb.labelKey), testid: `browse-tab-${tb.key}` })),
)
const initial = String(route.query.tab || '')
const tab = ref<Tab>(TAB_KEYS.some((tb) => tb.key === initial) ? (initial as Tab) : 'episodes')
const visitedTabs = useVisitedTabs(tab)
// Reported on CHANGE, so arriving is not counted as choosing the default tab (#2267).
watch(tab, (next) => track('browse_tab_view', { tab: next }))
watch(
  () => route.query.tab,
  (v) => {
    const q = String(v || '')
    if (TAB_KEYS.some((tb) => tb.key === q)) tab.value = q as Tab
  },
)
// Trending shows' "See all" on Discover arrives on Shows, sorted by trending, as a list.
const showsSort = computed(() => (route.query.sort === 'trending' ? ('trending' as const) : undefined))
const showsView = computed(() => (route.query.view === 'list' ? ('list' as const) : undefined))
</script>

<template>
  <section class="lp-page pb-8" data-testid="explore-view">
    <div class="mb-4 flex items-center justify-between gap-3">
      <h1 class="font-display text-3xl font-extrabold tracking-tight">{{ t('explore.title') }}</h1>
      <TrendingScopeButton testid="discover-scope" />
    </div>

    <div class="lg:flex lg:items-start lg:gap-8">
      <div class="lg:w-1/2 lg:pr-4">
        <TrendsSection ref="trends" :kind="trendsKind" @open="onEntityOpen" />
      </div>
      <div class="hidden lg:block lg:w-1/2" aria-hidden="true" />
    </div>

    <!-- `#catalog`: links into the list land on it (a hash lets the router wait and hold it). -->
    <div id="catalog" class="scroll-mt-4">
      <h2 class="lp-section mb-3 mt-8">{{ t('browse.catalogTitle') }}</h2>
      <Tabs v-model="tab" :tabs="tabs" :label="t('explore.title')" id-prefix="browse" equal-width class="mb-6" />
      <div v-show="tab === 'episodes'" v-bind="panelAttrs('browse', 'episodes')" data-testid="browse-panel-episodes"><template v-if="visitedTabs.has('episodes')"><CatalogView embedded /></template></div>
      <div v-show="tab === 'shows'" v-bind="panelAttrs('browse', 'shows')" data-testid="browse-panel-shows"><template v-if="visitedTabs.has('shows')"><ShowBrowseView embedded :initial-sort="showsSort" :initial-view="showsView" /></template></div>
    </div>
  </section>
</template>
