<script setup lang="ts">
/**
 * Discover — what to hear next, and only that (ADR-163 §1). Route name `browse` (kept: deep links,
 * the bottom nav and the masthead all point here).
 *
 * Episode-first (§2): the recommendation rails lead (`RecommendationRail` — each card says why it
 * is here and plays from the quote): the quote cards, then, under Mine, more from what you
 * captured, new in the storylines you follow, the people you keep hearing on other shows, and
 * another view on your topics; trending shows sit below them, then search. Trends and the Episodes ·
 * Shows catalog moved to Explore (operator 2026-10-11), reached from this header; old
 * `/browse?trends=…` / `?tab=…` / `#catalog` links redirect there (router).
 *
 * One Mine ⇄ Everyone switch for the whole page (ADR-162), shared with Explore and Search.
 *
 * Cold start (ADR-163 §6): under Mine with nothing in the listener's world yet, the quote cards
 * fall back to Everyone's, labelled as Everyone's, with "Pick topics" — the interests picker,
 * whose follows widen Mine (`world_episode_set`) so the next load is theirs.
 */
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
defineOptions({ name: 'BrowseView' }) // stable name for <keep-alive :include> (App.vue)
import SearchSection from '../components/SearchSection.vue'
import RecommendationRail from '../components/RecommendationRail.vue'
import {
  getAnotherViewRecommendations,
  getCapturedRecommendations,
  getPeopleRecommendations,
  getThreadRecommendations,
  getTrendingEpisodes,
  getWorldEpisodeSlugs,
} from '../services/api'
import { useAuthStore } from '../stores/auth'
import TrendingShowsRail from '../components/TrendingShowsRail.vue'
import TrendingScopeButton from '../components/TrendingScopeButton.vue'
import InterestsPicker from '../components/InterestsPicker.vue'
import { ref } from 'vue'
import { useTrendingScope } from '../composables/useTrendingScope'

const { t } = useI18n()
const router = useRouter()
const { scope: trendingScope, setScope: setTrendingScope } = useTrendingScope()
const auth = useAuthStore()
const pickerOpen = ref(false)
/** Everyone's quote cards — only while the listener's world is EMPTY (cold start). An empty Mine
 *  rail for someone with history is a quiet week, not a cold start: it stays empty rather than
 *  telling them their picks "begin once you listen". */
async function coldStartCards() {
  if ((await getWorldEpisodeSlugs()).length) return []
  return getTrendingEpisodes('corpus', 6)
}
const picked = ref(0) // bumps the quote rail's reloadKey after a save, so Mine reloads as theirs

// Trending shows' "See all" opens Explore's Shows tab sorted by trending, as a list, so the
// destination matches the rail it came from (operator 2026-09-14).
function onShowsSeeAll(): void {
  void router.push({
    name: 'explore',
    query: { tab: 'shows', sort: 'trending', view: 'list' },
    hash: '#catalog',
  })
}
</script>

<template>
  <section class="lp-page pb-8" data-testid="browse-view">
    <div class="mb-4 flex items-center justify-between gap-3">
      <h1 class="font-display text-3xl font-extrabold tracking-tight">{{ t('browse.hubTitle') }}</h1>
      <div class="flex items-center gap-3">
        <RouterLink
          :to="{ name: 'explore' }"
          class="lp-tap whitespace-nowrap text-sm font-bold text-accent"
          data-testid="discover-explore-link"
        >{{ t('discover.exploreLink') }} ›</RouterLink>
        <TrendingScopeButton testid="discover-scope" />
      </div>
    </div>

    <!-- Episode-first (ADR-163 §2): the quote cards, then — under Mine, for a signed-in listener —
         more from what they captured (Step 2.2), new in the storylines they follow (Step 2.3) and
         the people they keep hearing and another view on their topics (Step 3), then trending
         shows. -->
    <RecommendationRail
      rail="quotes"
      cards-testid="quote-cards"
      card-testid="quote-card"
      open-testid="quote-card-open"
      :title="t('discover.quotes.title')"
      :load="() => getTrendingEpisodes(trendingScope, 6)"
      :scope="trendingScope === 'mine' ? 'mine' : 'everyone'"
      :reload-key="`${trendingScope}:${picked}`"
      :fallback="trendingScope === 'mine' ? coldStartCards : undefined"
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
      v-if="auth.isAuthenticated && trendingScope === 'mine'"
      rail="captured"
      cards-testid="captured-cards"
      card-testid="captured-card"
      open-testid="captured-card-open"
      :title="t('discover.captured.title')"
      :load="() => getCapturedRecommendations(6)"
      scope="mine"
    />
    <RecommendationRail
      v-if="auth.isAuthenticated && trendingScope === 'mine'"
      rail="thread"
      cards-testid="thread-cards"
      card-testid="thread-card"
      open-testid="thread-card-open"
      :title="t('discover.thread.title')"
      :load="() => getThreadRecommendations(6)"
      scope="mine"
    />
    <RecommendationRail
      v-if="auth.isAuthenticated && trendingScope === 'mine'"
      rail="people"
      cards-testid="people-cards"
      card-testid="people-card"
      open-testid="people-card-open"
      :title="t('discover.people.title')"
      :load="() => getPeopleRecommendations(6)"
      scope="mine"
    />
    <RecommendationRail
      v-if="auth.isAuthenticated && trendingScope === 'mine'"
      rail="another_view"
      cards-testid="another-view-cards"
      card-testid="another-view-card"
      open-testid="another-view-card-open"
      :title="t('discover.anotherView.title')"
      :load="() => getAnotherViewRecommendations(6)"
      scope="mine"
    />

    <TrendingShowsRail
      :title="t('home.trendingShows')"
      :scope="trendingScope"
      :top="5"
      see-all
      @see-all="onShowsSeeAll"
      @show-everyone="setTrendingScope('corpus')"
    />

    <div class="lg:flex lg:items-start lg:gap-8">
      <div class="lg:w-1/2 lg:pr-4">
        <SearchSection prefix="browse" />
      </div>
      <div class="hidden lg:block lg:w-1/2" aria-hidden="true" />
    </div>
    <InterestsPicker
      v-if="pickerOpen"
      trigger="discover_cold_start"
      @close="pickerOpen = false"
      @saved="picked++"
    />
  </section>
</template>
