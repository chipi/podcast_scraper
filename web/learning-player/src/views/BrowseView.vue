<script setup lang="ts">
/**
 * Discover — what to hear next, and only that (ADR-163 §1). Route name `browse` (kept: deep links,
 * the bottom nav and the masthead all point here).
 *
 * A taste, not a scroll (operator 2026-10-11). Search leads — on a phone this tab is the way into
 * it. Then the recommendation rails (`RecommendationRail`, each card saying why it is here), paged
 * on the SERVER: one `GET /discover/rails` gives every rail's first page in page order, one card per
 * episode across the page, and each rail's "Show more" asks for its next page. The rails: the quote
 * cards, then under Mine more from what you captured, continue the thread, people you keep hearing,
 * another view on your topics. Then "Minutes on" and trending shows. Trends and the Episodes ·
 * Shows catalog are on Explore (its own tab); old `/browse?trends=…` / `?tab=…` / `#catalog` links
 * redirect there (router).
 *
 * One Mine ⇄ Everyone switch for the whole page (ADR-162), shared with Explore and Search.
 *
 * Cold start (ADR-163 §6), decided by the server: under Mine with nothing in the listener's world,
 * the quote rail is Everyone's, `fallback`, shown under a note with "Pick topics" (the interests
 * picker, whose follows widen Mine); saving reloads the rails as theirs.
 */
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
defineOptions({ name: 'BrowseView' }) // stable name for <keep-alive :include> (App.vue)
import SearchSection from '../components/SearchSection.vue'
import RecommendationRail from '../components/RecommendationRail.vue'
import { getDiscoverRails, getDiscoverRailPage, type DiscoverRail } from '../services/api'
import { useAuthStore } from '../stores/auth'
import TrendingShowsRail from '../components/TrendingShowsRail.vue'
import TrendingScopeButton from '../components/TrendingScopeButton.vue'
import InterestsPicker from '../components/InterestsPicker.vue'
import ReelTopicsRow from '../components/ReelTopicsRow.vue'
import LearnThisWeek from '../components/LearnThisWeek.vue'
import { computed, onActivated, onDeactivated, ref, watch } from 'vue'
import { useTrendingScope } from '../composables/useTrendingScope'

const { t } = useI18n()
const router = useRouter()
const { scope: trendingScope, setScope: setTrendingScope } = useTrendingScope()
const auth = useAuthStore()
const pickerOpen = ref(false)

// The rails, paged on the server (operator 2026-10-11): ONE request for every rail's first page,
// in page order, one card per episode across the page; each rail asks for more itself.
const railScope = computed<'mine' | 'corpus'>(() =>
  auth.isAuthenticated && trendingScope.value === 'mine' ? 'mine' : 'corpus',
)
const rails = ref<DiscoverRail[]>([])
let latest = 0
async function loadRails(): Promise<void> {
  const run = ++latest
  const got = await getDiscoverRails(railScope.value).catch(() => [] as DiscoverRail[])
  if (run === latest) rails.value = got
}
watch(railScope, loadRails, { immediate: true })
// Kept alive: a return reloads, so a highlight made a minute ago can seed its rail (the first
// activation is the mount itself, and is skipped).
let leftOnce = false
onDeactivated(() => (leftOnce = true))
onActivated(() => {
  if (leftOnce) void loadRails()
})

function railPage(id: DiscoverRail['rail']): DiscoverRail | undefined {
  return rails.value.find((r) => r.rail === id)
}
/** "Show more" for one rail: the next page from the server, leaving out what the rails above
 *  showed first (the page's one-card-per-episode rule). */
function moreFor(id: DiscoverRail['rail']) {
  return (offset: number, limit: number) => {
    const above: string[] = []
    for (const r of rails.value) {
      if (r.rail === id) break
      above.push(...r.items.map((c) => c.episode.slug))
    }
    return getDiscoverRailPage(id, { scope: railScope.value, offset, limit, exclude: above })
  }
}
const railTelemetryScope = computed<'mine' | 'everyone'>(() =>
  railScope.value === 'mine' ? 'mine' : 'everyone',
)

// Trending shows' "See all" opens the full show list sorted by trending, as a list, so the
// destination matches the rail it came from (operator 2026-09-14).
function onShowsSeeAll(): void {
  void router.push({ name: 'shows', query: { sort: 'trending', view: 'list' } })
}
</script>

<template>
  <section class="lp-page pb-8" data-testid="browse-view">
    <div class="mb-4 flex items-center justify-between gap-3">
      <h1 class="font-display text-3xl font-extrabold tracking-tight">{{ t('browse.hubTitle') }}</h1>
      <div class="flex items-center gap-3">
        <TrendingScopeButton testid="discover-scope" />
      </div>
    </div>

    <!-- Search first (operator 2026-10-11): on a phone the Discover tab IS the way into search
         (no tab, no header magnifier there), so it leads rather than sitting under every rail. -->
    <div class="lg:flex lg:items-start lg:gap-8">
      <div class="lg:w-1/2 lg:pr-4">
        <SearchSection prefix="browse" />
      </div>
      <div class="hidden lg:block lg:w-1/2" aria-hidden="true" />
    </div>

    <!-- Episode-first (ADR-163 §2), each rail a taste — its first page, "Show more" for the next:
         the quote cards, then under Mine more from what you captured, continue the thread, people
         you keep hearing and another view on your topics. -->
    <RecommendationRail
      v-if="railPage('quotes')"
      rail="quotes"
      cards-testid="quote-cards"
      card-testid="quote-card"
      open-testid="quote-card-open"
      :title="t('discover.quotes.title')"
      :page="railPage('quotes')!"
      :more="moreFor('quotes')"
      :scope="railTelemetryScope"
      @changed="loadRails"
    >
      <template #fallback-note>
        <p class="-mt-1 mb-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-muted" data-testid="discover-cold-start">
          <span>{{ t('discover.coldStart.note') }}</span>
          <button
            type="button"
            class="lp-tap font-bold text-accent"
            data-testid="discover-pick-topics"
            @click="pickerOpen = true"
          >{{ t('discover.coldStart.pick') }}</button>
        </p>
      </template>
    </RecommendationRail>
    <RecommendationRail
      v-if="railPage('captured')"
      rail="captured"
      cards-testid="captured-cards"
      card-testid="captured-card"
      open-testid="captured-card-open"
      :title="t('discover.captured.title')"
      :page="railPage('captured')!"
      :more="moreFor('captured')"
      :scope="railTelemetryScope"
      @changed="loadRails"
    />
    <RecommendationRail
      v-if="railPage('thread')"
      rail="thread"
      cards-testid="thread-cards"
      card-testid="thread-card"
      open-testid="thread-card-open"
      :title="t('discover.thread.title')"
      :page="railPage('thread')!"
      :more="moreFor('thread')"
      :scope="railTelemetryScope"
      @changed="loadRails"
    />
    <RecommendationRail
      v-if="railPage('people')"
      rail="people"
      cards-testid="people-cards"
      card-testid="people-card"
      open-testid="people-card-open"
      :title="t('discover.people.title')"
      :page="railPage('people')!"
      :more="moreFor('people')"
      :scope="railTelemetryScope"
      @changed="loadRails"
    />
    <RecommendationRail
      v-if="railPage('another_view')"
      rail="another_view"
      cards-testid="another-view-cards"
      card-testid="another-view-card"
      open-testid="another-view-card-open"
      :title="t('discover.anotherView.title')"
      :page="railPage('another_view')!"
      :more="moreFor('another_view')"
      :scope="railTelemetryScope"
      @changed="loadRails"
    />

    <!-- Goal-shaped listening (ADR-163 §8 amendment): fifteen minutes on a topic, across episodes. -->
    <ReelTopicsRow :scope="railScope" />
    <!-- Learn X this week (operator 2026-10-11): a seven-day path, one step a day. Signed in only. -->
    <LearnThisWeek v-if="auth.isAuthenticated" :scope="railScope" />

    <TrendingShowsRail
      :title="t('home.trendingShows')"
      :scope="trendingScope"
      :top="5"
      see-all
      @see-all="onShowsSeeAll"
      @show-everyone="setTrendingScope('corpus')"
    />

    <InterestsPicker
      v-if="pickerOpen"
      trigger="discover_cold_start"
      @close="pickerOpen = false"
      @saved="loadRails"
    />
  </section>
</template>
