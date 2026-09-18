<script setup lang="ts">
/**
 * The discovery section shared by Home and Discover (operator 2026-09-14): ONE tabbed `DiscoveryList`
 * over three entity KINDS (Topics / Storylines / People) with two little switches — sort
 * (Rising = velocity, Trending = volume) and scope (Corpus ⇄ Mine). Extracted from HomeView so the
 * two surfaces cannot drift.
 *
 * `collapsed` caps the rows (Home 5, Discover 10). `seeAll` swaps the list's inline "show more" for a
 * "See all →" link into the full /trends page on the active tab. Opening a row is the PARENT's call
 * (Home opens overlays, Discover navigates), so it is emitted.
 */
import { computed, ref, watch } from "vue"
import { useI18n } from "vue-i18n"
import { RouterLink } from "vue-router"
import Tabs from "./Tabs.vue"
import { panelAttrs, type TabSpec } from "./tabs"
import DiscoveryList from "./DiscoveryList.vue"
import { useAuthStore } from "../stores/auth"
import { useTrendingScope } from "../composables/useTrendingScope"

type Kind = "topic" | "storyline" | "person"

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
const emit = defineEmits<{ (e: "open", payload: { kind: Kind; id: string }): void }>()

const { t } = useI18n()
const auth = useAuthStore()
const { scope: trendingScope, setScope: setTrendingScope } = useTrendingScope()

const DISCOVERY_TABS = [
  { key: "topic", labelKey: "home.tabTopics" },
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
const discoverySort = ref<"rising" | "trending">("rising")
const discoveryTabs = computed<TabSpec<Kind>[]>(() =>
  DISCOVERY_TABS.map((tb) => ({ key: tb.key, label: t(tb.labelKey), testid: `discovery-tab-${tb.key}` }))
)
</script>

<template>
  <div data-testid="discovery-explorer">
    <!-- Optional section header (Discover): a "Trends" title with the "See all →" on the same row,
         mirroring the trending-shows header (operator 2026-09-14). It targets whichever kind tab is
         active. Home passes no title, so this row is absent. -->
    <div v-if="title" class="mb-3 flex items-center justify-between gap-2">
      <h2 class="lp-section">{{ title }}</h2>
      <RouterLink
        v-if="seeAll"
        :to="{ name: 'browse', query: { trends: discoveryTab } }"
        class="shrink-0 whitespace-nowrap text-sm font-bold text-accent no-underline"
        data-testid="discovery-see-all"
      >
        {{ t("home.seeAll") }} ›
      </RouterLink>
    </div>

    <!-- Kind pills + the sort/scope icon cluster on ONE row (operator). -->
    <div class="mb-3 flex items-center gap-2">
      <Tabs
        v-model="discoveryTab"
        :tabs="discoveryTabs"
        :label="t('home.discoveryTabs')"
        id-prefix="discovery"
        variant="pill"
        class="min-w-0"
      />
      <div class="ml-auto flex shrink-0 items-center gap-1.5">
        <!-- Rising ⇄ Trending — the glyph IS the active sort; tap re-sorts client-side. -->
        <button
          type="button"
          data-testid="discovery-sort"
          class="lp-tap flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border text-muted transition hover:text-canvas-foreground"
          :aria-label="t('home.discoverySortLabel', { mode: discoverySort === 'rising' ? t('home.tabRising') : t('home.tabTrending') })"
          :title="discoverySort === 'rising' ? t('home.tabRising') : t('home.tabTrending')"
          @click="discoverySort = discoverySort === 'rising' ? 'trending' : 'rising'"
        >
          <svg v-if="discoverySort === 'rising'" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" aria-hidden="true"><path d="M3 17l6-6 4 4 7-7" /><path d="M17 8h4v4" /></svg>
          <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" aria-hidden="true"><path d="M5 20V10M12 20V4M19 20v-7" /></svg>
        </button>
        <!-- Corpus ⇄ Mine scope — icon circle; active (accent) = My listening. -->
        <button
          v-if="auth.isAuthenticated"
          type="button"
          data-testid="home-trending-scope"
          class="lp-tap flex h-8 w-8 shrink-0 items-center justify-center rounded-full border transition"
          :class="trendingScope === 'mine' ? 'border-accent bg-accent text-accent-foreground' : 'border-border text-muted hover:text-canvas-foreground'"
          :aria-pressed="trendingScope === 'mine'"
          :aria-label="t('home.trendingScopeLabel')"
          :title="trendingScope === 'mine' ? t('home.trendingScopeMine') : t('home.trendingScopeAll')"
          @click="setTrendingScope(trendingScope === 'mine' ? 'corpus' : 'mine')"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" aria-hidden="true"><circle cx="12" cy="8" r="4" /><path d="M4 21a8 8 0 0 1 16 0" /></svg>
        </button>
      </div>
    </div>

    <div v-bind="panelAttrs('discovery', discoveryTab)">
      <DiscoveryList
        :kind="discoveryTab"
        :sort="discoverySort"
        :scope="trendingScope"
        :collapsed="collapsed"
        :hide-more="seeAll"
        @open="emit('open', $event)"
      />
    </div>
  </div>
</template>
