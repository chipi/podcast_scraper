<script setup lang="ts">
/**
 * The discovery section shared by Home and Discover (operator 2026-09-14): ONE tabbed `DiscoveryList`
 * over four entity KINDS (Topics / Themes / Storylines / People — Themes added 2026-10-05) with two little switches — sort
 * (Rising = velocity, Trending = volume) and scope (Corpus ⇄ Mine). Extracted from HomeView so the
 * two surfaces cannot drift.
 *
 * `collapsed` caps the rows (Home 5, Discover 10). `seeAll` swaps the list's inline "show more" —
 * which links OUT to Discover's trends section — for an in-place expand control in the section
 * header, because on Discover that link would point at the page you are already on. Opening a row is
 * the PARENT's call (Home opens overlays, Discover navigates), so it is emitted.
 */
import { computed, ref, watch } from "vue"
import { useI18n } from "vue-i18n"
import Tabs from "./Tabs.vue"
import SegmentedSwitch from "./SegmentedSwitch.vue"
import { panelAttrs, type TabSpec } from "./tabs"
import DiscoveryList from "./DiscoveryList.vue"
import { useAuthStore } from "../stores/auth"
import { useTrendingScope } from "../composables/useTrendingScope"

type Kind = "topic" | "theme" | "storyline" | "person"

const props = withDefaults(
  defineProps<{
    collapsed?: number
    seeAll?: boolean
    title?: string
    /**
     * Which kind tab to open on. Arrives from the route so Home's "See all →" can land on Discover
     * with the SAME tab the reader was looking at (operator 2026-09-17) — before this the explorer
     * always opened on Topics, so "See all" from the People rail dropped you somewhere else.
     */
    kind?: Kind
  }>(),
  { collapsed: 5, seeAll: false, title: "", kind: undefined }
)
// `rank` rides through to the opener (#2267): only the list knows which position was tapped,
// and only the view knows which rail it is, so the payload has to carry it across.
const emit = defineEmits<{
  (e: "open", payload: { kind: Kind; id: string; rank: number }): void
}>()

const { t } = useI18n()
const auth = useAuthStore()
const { scope: trendingScope, setScope: setTrendingScope } = useTrendingScope()

const DISCOVERY_TABS = [
  { key: "topic", labelKey: "home.tabTopics" },
  // Themes — topics that MEAN the same thing — were the one grouping missing from Trends
  // (operator 2026-10-05). Ordered as everywhere else the kinds are listed: Topics, Themes,
  // Storylines, People (Profile › Interests, the palette in tokens.css).
  { key: "theme", labelKey: "home.themes" },
  { key: "storyline", labelKey: "home.storylines" },
  { key: "person", labelKey: "home.tabPeople" },
] as const
const discoveryTab = ref<Kind>(props.kind ?? "topic")
// Kept-alive view: setup runs once, so a later navigation carrying a different kind has to be
// picked up here or the stale tab stays selected — same trap BrowseView documents for its own tabs.
watch(
  () => props.kind,
  (k) => {
    if (k) discoveryTab.value = k
  }
)
// Collapse on every tab change. Expanding People and landing on a 30-row Topics list you never
// asked to open would be its own surprise, and the control's label would be describing a state you
// did not choose.
const expandedAll = ref(false)
const total = ref(0)
watch(discoveryTab, () => {
  expandedAll.value = false
  // Also clear the COUNT. It is the previous tab's until the new fetch emits, so a tab with more
  // than `collapsed` rows briefly lent its control to a tab that fits — a button appearing and
  // then vanishing on its own.
  total.value = 0
})

const discoverySort = ref<"rising" | "trending">("rising")
const discoveryTabs = computed<TabSpec<Kind>[]>(() =>
  DISCOVERY_TABS.map((tb) => ({ key: tb.key, label: t(tb.labelKey), testid: `discovery-tab-${tb.key}` }))
)
</script>

<template>
  <div data-testid="discovery-explorer">
    <!-- Optional section header (Discover): a "Trends" title with the expand control on the same
         row, mirroring the trending-shows header (operator 2026-09-14). Home passes no title, so
         this row is absent.

         It used to be a RouterLink to `{ name: 'browse', query: { trends: discoveryTab } }`. The
         only surface that renders this header IS Browse — "Discover" in the tab bar is the `browse`
         route — so the link pointed at the page you were already on, carrying the kind you were
         already reading. Vue Router navigated, the query changed, and nothing moved. "I click all,
         and nothing really happens" (operator 2026-09-19) was exactly right.

         The list beneath is capped at `collapsed` (10 here) with the remainder unreachable, so
         uncapping it is what this control should always have done. A button, not a link, because it
         does not go anywhere. It renders only when rows are genuinely hidden: this replaces a
         control that did nothing, so one that sometimes does nothing would miss the point. -->
    <div v-if="title" class="mb-3 flex items-center justify-between gap-2">
      <h2 class="lp-section">{{ title }}</h2>
      <button
        v-if="seeAll && total > collapsed"
        type="button"
        class="lp-tap shrink-0 whitespace-nowrap text-sm font-bold text-accent"
        :aria-expanded="expandedAll"
        :aria-controls="panelAttrs('discovery', discoveryTab).id"
        data-testid="discovery-see-all"
        @click="expandedAll = !expandedAll"
      >{{ expandedAll ? t("home.showLess") : t("home.seeAll") }} {{ expandedAll ? "‹" : "›" }}</button>
    </div>

    <!-- Kind pills on their own row; the two switches under them, in WORDS (operator 2026-10-07):
         the icon-only person and chart buttons left readers guessing what they did. -->
    <Tabs
      v-model="discoveryTab"
      :tabs="discoveryTabs"
      :label="t('home.discoveryTabs')"
      id-prefix="discovery"
      variant="pill"
      dense
      class="mb-2 min-w-0"
    />
    <div class="mb-3 flex flex-wrap items-center gap-2" data-testid="discovery-switches">
      <SegmentedSwitch
        v-if="auth.isAuthenticated"
        :model-value="trendingScope"
        :label="t('home.trendingScopeLabel')"
        :options="[
          { value: 'mine', label: t('home.trendsYou'), testid: 'home-trending-scope' },
          { value: 'corpus', label: t('home.trendsEveryone'), testid: 'home-trending-scope-everyone' },
        ]"
        data-testid="discovery-scope"
        @update:model-value="setTrendingScope($event)"
      />
      <SegmentedSwitch
        v-model="discoverySort"
        :label="t('home.discoverySortGroup')"
        :options="[
          { value: 'rising', label: t('home.tabRising'), testid: 'discovery-sort-rising' },
          { value: 'trending', label: t('home.trendsMostTalked'), testid: 'discovery-sort-trending' },
        ]"
        data-testid="discovery-sort"
      />
    </div>

    <!-- `panelAttrs` already mints this panel's id; the header's expand control points
         `aria-controls` at THAT one. A second hardcoded id here would be dropped by Vue (duplicate
         attribute) and the control would reference an element that does not exist. -->
    <div v-bind="panelAttrs('discovery', discoveryTab)">
      <DiscoveryList
        :key="discoveryTab"
        :kind="discoveryTab"
        :sort="discoverySort"
        :scope="trendingScope"
        :collapsed="collapsed"
        :hide-more="seeAll"
        :expanded="expandedAll"
        @open="emit('open', $event)"
        @count="total = $event"
        @show-everyone="setTrendingScope('corpus')"
      />
    </div>
  </div>
</template>
