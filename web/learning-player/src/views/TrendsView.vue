<script setup lang="ts">
/**
 * Entity trends page (operator 2026-09-14) — the full 3-tab discovery reached from the Discover
 * dashboard's "See all ›": Topics · Storylines · People, each the shared `DiscoveryList` with the
 * Rising⇄Trending sort + Corpus⇄Mine scope + window. The dashboard is the glance; this is the depth.
 */
import { computed, ref } from "vue"
import { useI18n } from "vue-i18n"
import { RouterLink, useRoute, useRouter } from "vue-router"
import Tabs from "../components/Tabs.vue"
import { panelAttrs, type TabSpec } from "../components/tabs"
import DiscoveryList from "../components/DiscoveryList.vue"
import { useTrendingScope } from "../composables/useTrendingScope"
import { useAuthStore } from "../stores/auth"

type Kind = "topic" | "storyline" | "person"
const { t } = useI18n()
const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const { scope, setScope } = useTrendingScope()
const sort = ref<"rising" | "trending">("rising")

const KIND_TABS: { key: Kind; labelKey: string }[] = [
  { key: "topic", labelKey: "home.tabTopics" },
  { key: "storyline", labelKey: "home.storylines" },
  { key: "person", labelKey: "home.tabPeople" },
]
const tabs = computed<TabSpec<Kind>[]>(() =>
  KIND_TABS.map((k) => ({ key: k.key, label: t(k.labelKey), testid: `trends-tab-${k.key}` }))
)
const initial = String(route.query.tab || "")
const tab = ref<Kind>(KIND_TABS.some((k) => k.key === initial) ? (initial as Kind) : "topic")

function onOpen(p: { kind: Kind; id: string }): void {
  const name = p.kind === "topic" ? "topic" : p.kind === "person" ? "person" : "storyline"
  void router.push({ name, params: { id: p.id } })
}
</script>

<template>
  <section class="mx-auto max-w-3xl px-4 pb-8 pt-4" data-testid="trends-view">
    <RouterLink :to="{ name: 'browse' }" class="lp-nav mb-4" data-testid="trends-back">
      ‹ {{ t("browse.hubTitle") }}
    </RouterLink>
    <h1 class="mb-4 font-display text-3xl font-extrabold tracking-tight">{{ t("home.trendingNow") }}</h1>

    <div class="mb-4 flex items-center gap-2">
      <Tabs
        v-model="tab"
        :tabs="tabs"
        :label="t('home.trendingNow')"
        id-prefix="trends"
        variant="pill"
        class="min-w-0"
      />
      <div class="ml-auto flex shrink-0 items-center gap-1.5">
        <button
          type="button"
          data-testid="trends-sort"
          class="lp-tap flex h-8 w-8 items-center justify-center rounded-full border border-border text-muted transition hover:text-canvas-foreground"
          :aria-label="t('home.discoverySortLabel', { mode: sort === 'rising' ? t('home.tabRising') : t('home.tabTrending') })"
          :title="sort === 'rising' ? t('home.tabRising') : t('home.tabTrending')"
          @click="sort = sort === 'rising' ? 'trending' : 'rising'"
        >
          <svg v-if="sort === 'rising'" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" aria-hidden="true"><path d="M3 17l6-6 4 4 7-7" /><path d="M17 8h4v4" /></svg>
          <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" aria-hidden="true"><path d="M5 20V10M12 20V4M19 20v-7" /></svg>
        </button>
        <button
          v-if="auth.isAuthenticated"
          type="button"
          data-testid="trends-scope"
          class="lp-tap flex h-8 w-8 items-center justify-center rounded-full border transition"
          :class="scope === 'mine' ? 'border-accent bg-accent text-accent-foreground' : 'border-border text-muted hover:text-canvas-foreground'"
          :aria-pressed="scope === 'mine'"
          :aria-label="t('home.trendingScopeLabel')"
          :title="scope === 'mine' ? t('home.trendingScopeMine') : t('home.trendingScopeAll')"
          @click="setScope(scope === 'mine' ? 'corpus' : 'mine')"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" aria-hidden="true"><circle cx="12" cy="8" r="4" /><path d="M4 21a8 8 0 0 1 16 0" /></svg>
        </button>
      </div>
    </div>

    <div v-bind="panelAttrs('trends', tab)">
      <DiscoveryList :kind="tab" :sort="sort" :scope="scope" @open="onOpen" />
    </div>
  </section>
</template>
