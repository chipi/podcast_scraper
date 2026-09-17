<script setup lang="ts">
/**
 * Browse hub (#14, revised) — one tabbed page, discovery-first: Topics · People · Episodes · Shows,
 * Topics active by default (operator 2026-09-14). The corpus indexes render INLINE as tab panels rather than three
 * buttons that navigate away (which was a navigation hustle). Each panel reuses the standalone index
 * view in `embedded` mode, which drops its page heading and — for Topics/People — its back-to-Home
 * button (that button is only meaningful on the standalone routes reached from Home, not here).
 *
 * v-show (not v-if) keeps each panel mounted so switching tabs never refetches; supports ?tab= for
 * deep links.
 */
import { computed, nextTick, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
defineOptions({ name: 'BrowseView' }) // stable name for <keep-alive :include> (App.vue)
import Tabs from '../components/Tabs.vue'
import { panelAttrs, type TabSpec } from '../components/tabs'
import CatalogView from './CatalogView.vue'
import ShowBrowseView from './ShowBrowseView.vue'
import DiscoveryExplorer from '../components/DiscoveryExplorer.vue'
import TrendingShowsRail from '../components/TrendingShowsRail.vue'
import { getPodcasts } from '../services/api'
import type { Podcast } from '../services/types'
import { scrollBehavior } from '../utils/motion'

const { t } = useI18n()
const route = useRoute()
const router = useRouter()

// Trending-shows area at the very top of Discover (operator 2026-09-14): the catalogue supplies the
// cover art the rail joins by feed_id (same as Home). "See all →" drops into the Shows tab below,
// where the trending sort + sparklines let you see how each is trending.
const catalogue = ref<Podcast[]>([])
void getPodcasts()
  .then((rows) => (catalogue.value = rows))
  .catch(() => (catalogue.value = []))

// "See all →" on the trending-shows rail lands on the Shows tab pre-sorted by Trending, so the
// destination matches the rail you came from (operator 2026-09-14).
const showsSort = ref<'trending' | undefined>(undefined)
// List view too, so the per-row trend sparklines are visible on arrival (operator 2026-09-14).
const showsView = ref<'list' | undefined>(undefined)
// The content band sits below the trending rail + the Trends dashboard, so switching the tab is
// invisible from the top. Scroll it into view so "See all" actually lands you on the list.
const bandEl = ref<HTMLElement | null>(null)
function onShowsSeeAll(): void {
  showsSort.value = 'trending'
  showsView.value = 'list'
  tab.value = 'shows'
  void nextTick(() => bandEl.value?.scrollIntoView({ behavior: scrollBehavior(), block: 'start' }))
}

type Kind = 'topic' | 'storyline' | 'person'
type Tab = 'episodes' | 'shows'

// Discover page = the shared DiscoveryExplorer under "Trends" (Topics/Storylines/People, the
// "what/who"), then the CONTENT band below — just Episodes · Shows (operator 2026-09-14). Tapping a
// row opens the entity as a full page here (Home opens overlays instead); the explorer's own
// "See all →" link handles the deep /trends page per kind.
function onEntityOpen(p: { kind: Kind; id: string }): void {
  const name = p.kind === 'topic' ? 'topic' : p.kind === 'person' ? 'person' : 'storyline'
  void router.push({ name, params: { id: p.id } })
}
const TAB_KEYS: { key: Tab; labelKey: string }[] = [
  { key: 'episodes', labelKey: 'browse.episodes' },
  { key: 'shows', labelKey: 'browse.shows' },
]
const tabs = computed<TabSpec<Tab>[]>(() =>
  TAB_KEYS.map((tb) => ({ key: tb.key, label: t(tb.labelKey), testid: `browse-tab-${tb.key}` })),
)
// `?trends=topic|storyline|person` selects the kind inside the trends section. Deliberately NOT
// `?tab=`: that one drives THIS view's own Episodes/Shows tabs, so reusing it would both miss the
// trends tab and reset the page to Episodes (operator 2026-09-17).
const TRENDS_KINDS = ['topic', 'storyline', 'person'] as const
const trendsKind = computed(() => {
  const q = String(route.query.trends || '')
  return (TRENDS_KINDS as readonly string[]).includes(q)
    ? (q as 'topic' | 'storyline' | 'person')
    : undefined
})
const initial = String(route.query.tab || '')
const tab = ref<Tab>(TAB_KEYS.some((tb) => tb.key === initial) ? (initial as Tab) : 'episodes')

// This view is kept-alive (App.vue), so setup runs once — without this watch a later in-app
// navigation to ?tab=<other> (e.g. Home's "Browse people" chip after the hub was already opened on
// Topics) would leave the stale tab selected. Re-sync whenever the query tab changes.
watch(
  () => route.query.tab,
  (v) => {
    const q = String(v || '')
    if (TAB_KEYS.some((tb) => tb.key === q)) tab.value = q as Tab
  }
)
</script>

<template>
  <section class="mx-auto max-w-3xl px-4 pb-8" data-testid="browse-view">
    <h1 class="mb-4 font-display text-3xl font-extrabold tracking-tight">
      {{ t('browse.hubTitle') }}
    </h1>

    <!-- Trending shows, above the entity dashboard (operator 2026-09-14): the standard ShowTile in a
         horizontal row, top 5 (the `tiles` variant — Home keeps the full-width slices). Each links to
         its show; "See all →" opens the Shows tab below. -->
    <TrendingShowsRail
      :title="t('home.trendingShows')"
      :podcasts="catalogue"
      variant="tiles"
      :top="5"
      see-all
      @see-all="onShowsSeeAll"
    />

    <!-- The entity trends are their own section (operator 2026-09-14): the SAME tabbed DiscoveryList
         Home uses, capped at 10 here (5 on Home). The "Trends" title + "See all →" ride one header
         row (like trending shows); the link targets the active kind tab. Tapping a row opens the
         entity page. -->
    <div class="mt-4">
      <DiscoveryExplorer
        id="trends"
        :collapsed="10"
        see-all
        :kind="trendsKind"
        :title="t('browse.trendsTitle')"
        @open="onEntityOpen"
      />
    </div>

    <!-- Content band below the dashboard: the things you actually play. Two tabs spread equally
         across the row (operator 2026-09-14) rather than sitting cramped on the left. `scroll-mt`
         leaves a little breathing room when "See all" scrolls this into view. -->
    <div ref="bandEl" class="scroll-mt-4">
      <!-- A section heading over the Episodes · Shows tabs, parallel to "Trends" above (operator
           2026-09-14). -->
      <h2 class="lp-section mb-3 mt-8">{{ t('browse.catalogTitle') }}</h2>
      <Tabs
        v-model="tab"
        :tabs="tabs"
        :label="t('browse.hubTitle')"
        id-prefix="browse"
        equal-width
        class="mb-6"
      />

      <div v-show="tab === 'episodes'" v-bind="panelAttrs('browse', 'episodes')" data-testid="browse-panel-episodes"><CatalogView embedded /></div>
      <div v-show="tab === 'shows'" v-bind="panelAttrs('browse', 'shows')" data-testid="browse-panel-shows"><ShowBrowseView embedded :initial-sort="showsSort" :initial-view="showsView" /></div>
    </div>
  </section>
</template>
