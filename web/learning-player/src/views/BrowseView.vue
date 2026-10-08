<script setup lang="ts">
/**
 * Browse hub (#14, revised) — one tabbed page, discovery-first: Topics · People · Episodes · Shows,
 * Topics active by default (operator 2026-09-14). The corpus indexes render INLINE as tab panels rather than three
 * buttons that navigate away (which was a navigation hustle). Each panel reuses the standalone index
 * view in `embedded` mode, which drops its page heading and — for Topics/People — its back-to-Home
 * button (that button is only meaningful on the standalone routes reached from Home, not here).
 *
 * A panel's content mounts the first time its tab opens and then stays (v-show), so switching back
 * never refetches and a tab never opened loads nothing (useVisitedTabs). Supports ?tab= for deep links.
 */
import { computed, nextTick, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
defineOptions({ name: 'BrowseView' }) // stable name for <keep-alive :include> (App.vue)
import Tabs from '../components/Tabs.vue'
import { toRankBucket, track } from '../services/analytics'
import { panelAttrs, type TabSpec } from '../components/tabs'
import CatalogView from './CatalogView.vue'
import ShowBrowseView from './ShowBrowseView.vue'
import { useVisitedTabs } from '../composables/useVisitedTabs'
import SearchSection from '../components/SearchSection.vue'
import TrendsSection from '../components/TrendsSection.vue'
import TrendingShowsRail from '../components/TrendingShowsRail.vue'
import { scrollBehavior } from '../utils/motion'

const { t } = useI18n()
const route = useRoute()
const router = useRouter()


// Trending-shows area at the very top of Discover (operator 2026-09-14): the rail looks up the
// covers of the shows it shows. "See all →" drops into the Shows tab below, where the trending
// sort + sparklines let you see how each is trending.

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

type Kind = 'topic' | 'theme' | 'storyline' | 'person'
type Tab = 'episodes' | 'shows'

// Discover page = the shared DiscoveryExplorer under "Trends" (Topics/Storylines/People, the
// "what/who"), then the CONTENT band below — just Episodes · Shows (operator 2026-09-14). Tapping a
// row opens the entity as a full page here (Home opens overlays instead); the explorer's own
// "See all →" on the explorer deep-links back into this same section per kind (`?trends=`).
function onEntityOpen(p: { kind: Kind; id: string; rank: number }): void {
  const name =
    p.kind === 'topic' ? 'topic' : p.kind === 'person' ? 'person' : p.kind === 'theme' ? 'theme' : 'storyline'
  // `presentation: 'page'`: Trends opens entities as full pages (Home's overlay-card Trends went
  // away 2026-10-07).
  track('trends_row_click', { kind: p.kind, rank: toRankBucket(p.rank) })
  track('entity_open', { kind: p.kind, presentation: 'page', source: 'browse' })
  void router.push({ name, params: { id: p.id } })
}
const TAB_KEYS: { key: Tab; labelKey: string }[] = [
  { key: 'episodes', labelKey: 'browse.episodes' },
  { key: 'shows', labelKey: 'browse.shows' },
]
const tabs = computed<TabSpec<Tab>[]>(() =>
  TAB_KEYS.map((tb) => ({ key: tb.key, label: t(tb.labelKey), testid: `browse-tab-${tb.key}` })),
)
// `?trends=topic|theme|storyline|person` selects the kind inside the trends section. Deliberately NOT
// `?tab=`: that one drives THIS view's own Episodes/Shows tabs, so reusing it would both miss the
// trends tab and reset the page to Episodes (operator 2026-09-17).
const TRENDS_KINDS = ['topic', 'theme', 'storyline', 'person'] as const
const trendsKind = computed(() => {
  const q = String(route.query.trends || '')
  return (TRENDS_KINDS as readonly string[]).includes(q)
    ? (q as Kind)
    : undefined
})
/**
 * `?trends=<kind>` must also SCROLL the section into view.
 *
 * Selecting the tab alone is invisible: the Trends block sits below the trending-shows rail, so a
 * chip tap from Home landed at the top of Browse with the change off-screen — indistinguishable
 * from the link not working, which is how it was reported (operator 2026-09-18).
 *
 * Declared AFTER `trends`/`trendsKind` deliberately: an `immediate` watch placed above the consts
 * it reads throws a TDZ ReferenceError at setup that neither the build nor the unit suite catches.
 */
const trends = ref<{ trendsEl: HTMLElement | null } | null>(null)
watch(
  trendsKind,
  (k) => {
    if (!k) return
    void nextTick(() =>
      trends.value?.trendsEl?.scrollIntoView({ behavior: scrollBehavior(), block: 'start' }),
    )
  },
  { immediate: true },
)

const initial = String(route.query.tab || '')
const tab = ref<Tab>(TAB_KEYS.some((tb) => tb.key === initial) ? (initial as Tab) : 'episodes')
// Panels mount on first visit, then stay (v-show): a tab never opened fetches and decodes nothing.
const visitedTabs = useVisitedTabs(tab)
// Which browse surface people actually use (#2267). Reported on CHANGE rather than on mount, so
// arriving at the hub is not counted as choosing the default tab — otherwise `episodes` would
// always lead simply because it is first.
watch(tab, (next) => track('browse_tab_view', { tab: next }))

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
  <!-- The one page width (`lp-page`, operator 2026-10-05): no gutter or column of its own. Discover
       once added `px-4` and a 768px column on top of the shell, so the SAME Trends component was
       narrower here than on Home. Home and Discover are one screen family; they size alike. -->
  <section class="lp-page pb-8" data-testid="browse-view">
    <h1 class="mb-4 font-display text-3xl font-extrabold tracking-tight">
      {{ t('browse.hubTitle') }}
    </h1>

    <!-- Trending shows, above the entity dashboard (operator 2026-09-14): the standard rail of
         standard ShowTiles, top 5. Each links to its show; "See all →" opens the Shows tab below. -->
    <TrendingShowsRail
      :title="t('home.trendingShows')"
      :top="5"
      see-all
      @see-all="onShowsSeeAll"
    />

    <!-- Search, then Trends — the same two sections, in the same order, that Home renders (operator
         2026-10-05). Search sits between trending shows and Trends, where the page turns from
         "what's popular" to "go find something" (operator 2026-09-20). Left half from `lg` up, in the
         SAME two-column row Home uses (`lg:flex lg:gap-8`, two halves), so the column is the same
         width on both pages by construction. Discover's right half is empty for now: what goes there
         is an open design question (operator 2026-10-05). -->
    <div class="lg:flex lg:items-start lg:gap-8">
      <div class="lg:w-1/2 lg:pr-4">
        <SearchSection prefix="browse" />
        <TrendsSection ref="trends" :kind="trendsKind" @open="onEntityOpen" />
      </div>
      <div class="hidden lg:block lg:w-1/2" aria-hidden="true" />
    </div>

    <!-- Content band below the dashboard: the things you actually play. Two tabs spread equally
         across the row (operator 2026-09-14) rather than sitting cramped on the left. `scroll-mt`
         leaves a little breathing room when "See all" scrolls this into view. -->
    <!-- `#catalog` is the anchor links into the band use (operator 2026-10-08: Home's "Browse all"
         landed at the top of Discover, two screens above the list). A hash lets the router wait for
         the band, land on it and hold it while the rails above load; a query alone scrolls to top. -->
    <div id="catalog" ref="bandEl" class="scroll-mt-4">
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

      <div v-show="tab === 'episodes'" v-bind="panelAttrs('browse', 'episodes')" data-testid="browse-panel-episodes"><template v-if="visitedTabs.has('episodes')"><CatalogView embedded /></template></div>
      <div v-show="tab === 'shows'" v-bind="panelAttrs('browse', 'shows')" data-testid="browse-panel-shows"><template v-if="visitedTabs.has('shows')"><ShowBrowseView embedded :initial-sort="showsSort" :initial-view="showsView" /></template></div>
    </div>
  </section>
</template>
