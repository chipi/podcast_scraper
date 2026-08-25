<script setup lang="ts">
/**
 * Home — the Learning Hub (PRD-042 / UXS-012). Adaptive hero: resume-state (Continue) when
 * signed-in with in-progress history, else discover-state ("Ask your library" + Featured).
 * Corpus search is prominent in both states. Sections (What's new / Recommended / Your shows)
 * hide cleanly when empty or signed-out. All data from the real /api/app/* surface.
 */
import { computed, onActivated, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
defineOptions({ name: 'HomeView' }) // stable name for <keep-alive :include> (App.vue)
import { RouterLink, useRouter } from 'vue-router'
import {
  getDiscover,
  getEpisode,
  getPlaybackList,
  getPodcasts,
  getRelated,
  recordDiscoverClick,
} from '../services/api'
import type { EpisodeDetail, EpisodeSummary, Podcast, Storyline } from '../services/types'
import { formatTime } from '../player/transcriptSync'
import { formatDuration } from '../utils/format'
import { episodeArtwork } from '../utils/episode'
import { useAuthStore } from '../stores/auth'
import { useLibraryStore } from '../stores/library'
import { useSectionState } from '../composables/useSectionState'
import { useUserPreferencesStore } from '../stores/userPreferences'
import { useInterestsStore } from '../stores/interests'
import EntityCard from '../components/EntityCard.vue'
import StorylineCard from '../components/StorylineCard.vue'
import InterestsPicker from '../components/InterestsPicker.vue'
import MomentumRail from '../components/MomentumRail.vue'
import TrendingShowsRail from '../components/TrendingShowsRail.vue'
import QueueButton from '../components/QueueButton.vue'
import SectionStatus from '../components/SectionStatus.vue'
import ShowTile from '../components/ShowTile.vue'
import Storylines from '../components/Storylines.vue'
import TrendingTopics from '../components/TrendingTopics.vue'
import YourWeek from '../components/YourWeek.vue'

const INTERESTS_DISMISSED_KEY = 'lp.interests.dismissed'

const { t } = useI18n()
const router = useRouter()
const auth = useAuthStore()
const library = useLibraryStore()
const userPrefs = useUserPreferencesStore()
const interests = useInterestsStore()

// USERPREFS-1 key for the "set your interests" dismissal (gh #1213).
// localStorage remains the fast-path fallback until the server responds.
const INTERESTS_DISMISSED_PREF_KEY = 'lp.interests.dismissed'

const whatsNew = useSectionState<EpisodeSummary[]>([])
const latest = computed(() => whatsNew.data.value)
const catalogue = ref<Podcast[]>([])
/**
 * Following and Continue get the same contract as every other section (#1591, S7).
 *
 * These two were the last holdouts on `.catch(() => [])`, and they are the two most personal
 * sections on the page — so a library outage rendered "follow something to get started" to a user
 * who follows thirty shows, and a playback outage silently swapped the resume hero for the discover
 * hero. An outage that looks like a new account is the exact defect #1591 exists to kill; I fixed
 * it in the sections around these and not in these.
 */
const followsSection = useSectionState<null>(null)
const continueSection = useSectionState<{ detail: EpisodeDetail; position: number }[]>([])
const recSection = useSectionState<EpisodeSummary[]>([])
const recommended = computed(() => recSection.data.value)
const continueItems = computed(() => continueSection.data.value)
const query = ref('')

// Trending-topic chip → open the topic entity card (overlay), same surface as Search.
const cardTarget = ref<{ kind: 'person' | 'topic'; id: string } | null>(null)
// #4 — Rising now / Trending topics / Storylines are three views of "what's hot"; stacked, they made
// Home very tall. Fold them into one tabbed area (Rising default). v-show (not v-if) keeps each rail
// mounted so switching tabs doesn't refetch; TrendingTopics still lazy-loads via its own observer.
const DISCOVERY_TABS = [
  { key: 'rising', label: 'home.risingNow' },
  { key: 'trending', label: 'home.trending' },
  { key: 'storylines', label: 'home.storylines' },
] as const
const discoveryTab = ref<(typeof DISCOVERY_TABS)[number]['key']>('rising')
// #9 — a tapped storyline opens ITS OWN sheet (titled with the storyline, listing member topics),
// not one member's topic card. Opening a member from that sheet then swaps to the topic entity card.
const storylineTarget = ref<Storyline | null>(null)
function openStorylineTopic(id: string): void {
  storylineTarget.value = null
  cardTarget.value = { kind: 'topic', id }
}

// First-Home dismissible "set your interests" card → opens the picker (PRD-043 FR4 / 3.5).
const interestsDismissed = ref(false)
const pickerOpen = ref(false)
// Only offer the "choose interests" card to users who have NOT already picked any — the bug was it
// showed even to users with a full interest set. Gate on the store being loaded so it never flashes
// before we know, and it hides the instant interests exist.
const showInterestsCard = computed(
  () =>
    auth.isAuthenticated &&
    interests.loaded &&
    interests.ids.length === 0 &&
    !interestsDismissed.value,
)

function dismissInterests(): void {
  interestsDismissed.value = true
  try {
    localStorage.setItem(INTERESTS_DISMISSED_KEY, '1')
  } catch {
    /* private mode / storage disabled — the card just reappears next load */
  }
  // USERPREFS-1 (#1213) — write-through so the dismissal syncs across devices.
  // silent-degrade: userPrefs.set is no-op when the server is unavailable.
  void userPrefs.set(INTERESTS_DISMISSED_PREF_KEY, true)
}

async function onInterestsSaved(): Promise<void> {
  dismissInterests()
  // Re-pull discovery so a personalized order (when the flag is on) takes effect immediately.
  await loadWhatsNew()
}

/**
 * What's New load (#1591). Failure is NOT collapsed into emptiness — that collapse is why a total
 * API outage rendered the same page as a brand-new account. A rejection lands in the error phase,
 * which renders a message and a retry.
 */
function loadWhatsNew(): Promise<void> {
  return whatsNew.load(async () => (await getDiscover(8)).items)
}

/** #1591 — Recommended, same contract: a rejection is an error phase, not an empty list. */
function loadRecommended(): Promise<void> {
  const top = continueItems.value[0]
  if (!top) return Promise.resolve()
  return recSection.load(async () => (await getRelated(top.detail.slug)).items)
}

const resumeState = computed(() => auth.isAuthenticated && continueItems.value.length > 0)
// Editorial ranked "What's new": a featured #1 + ranked rows — all on screen, no scroll.
const wnFeatured = computed(() => latest.value[0] ?? null)
const wnRows = computed(() => latest.value.slice(1, 6))
const rank = (i: number) => String(i + 2).padStart(2, '0')
const resumeTop = computed(() => continueItems.value[0] ?? null)
const resumeArt = episodeArtwork
/**
 * Resolve the user's followed shows into full `Podcast` records.
 *
 * The library API returns subscriptions (feed_id + title + added_at), not catalogue metadata, so
 * artwork and episode counts are joined from the public catalogue. A followed feed that isn't in
 * the corpus still renders — from its stored title — rather than vanishing.
 */
async function loadFollowedShows(): Promise<void> {
  // The catalogue loads for EVERYONE, not just signed-in users. It is public corpus metadata, and
  // more than one surface joins against it — most importantly TrendingShowsRail, which resolves the
  // artwork for trending shows the user does not follow. #1585 narrowed `shows` from "the whole
  // catalogue" to "shows you follow" without re-auditing the rail still reading it, so every
  // signed-out visitor, and every unfollowed show, silently lost its cover art to the generated
  // gradient fallback. Nothing failed, because the fallback is a valid render.
  await followsSection.load(async () => {
    // Both halves must succeed for "you follow nothing" to be a truthful render: the catalogue
    // supplies artwork, the library supplies the follows themselves.
    const [cat] = await Promise.all([
      getPodcasts(),
      auth.isAuthenticated ? library.ensureLoaded() : Promise.resolve(),
    ])
    catalogue.value = cat
    return null
  })
}

/**
 * Derived, not assigned, so following a show from the empty state moves it into the grid instantly
 * — the action completes where it was offered, with no reload and no navigation.
 *
 * A followed feed that has left the corpus still renders from its stored title rather than
 * silently vanishing.
 */
const shows = computed<Podcast[]>(() => {
  if (!auth.isAuthenticated) return []
  const byId = new Map(catalogue.value.map((p) => [p.feed_id, p]))
  return library.items.map(
    (i) =>
      byId.get(i.feed_id) ?? {
        feed_id: i.feed_id,
        title: i.title,
        artwork_url: null,
        image_url: null,
        description: null,
        episode_count: 0,
      },
  )
})

/**
 * What the empty state offers. Following is only discoverable today from a show page, so an empty
 * "Your shows" that merely *describes* following makes the user go find it. These tiles carry the
 * follow control itself, so the section teaches the capability and completes it in one place.
 */
const suggestedShows = computed<Podcast[]>(() =>
  catalogue.value.filter((p) => !library.has(p.feed_id)).slice(0, 6),
)

/**
 * Home caps the shows grid and links out for the rest (#1584). Unbounded, this section grows without
 * limit as the corpus does — it was the "taking all that real estate" half of the complaint. 5 and
 * 11 leave room for the See-all tile to complete a row at 3 columns (mobile) and 4 (desktop).
 */
const SHOWS_ON_HOME = 11
const visibleShows = computed(() => shows.value.slice(0, SHOWS_ON_HOME))
const epArt = episodeArtwork

function goSearch(q: string): void {
  const term = q.trim()
  if (term) void router.push({ name: 'search', query: { q: term } })
}

onMounted(async () => {
  try {
    interestsDismissed.value = localStorage.getItem(INTERESTS_DISMISSED_KEY) === '1'
  } catch {
    interestsDismissed.value = false
  }
  // USERPREFS-1 (#1213) — read the server preferences (hydrated once at
  // app init in main.ts). Server value wins over localStorage. Reading
  // is synchronous; if the payload arrives later, the value is picked up
  // on the next Home mount.
  const remote = userPrefs.get<boolean>(INTERESTS_DISMISSED_PREF_KEY)
  if (remote === true) interestsDismissed.value = true
  // Load the user's chosen interests so the "choose interests" card only shows when there are none
  // (fire-and-forget: the card stays hidden until this resolves, then appears only if empty).
  if (auth.isAuthenticated) void interests.ensureLoaded()
  await loadWhatsNew()
  // "Your shows" means the shows you follow. UXS-014:102 decided this ("we don't show the whole
  // corpus as 'your shows'") and gated it on subscriptions being user-curated — which they now are,
  // since follow-show shipped. The corpus catalogue lives in Browse. Artwork/titles still come from
  // the public catalogue, since the library rows carry only feed_id + title (#1585).
  void loadFollowedShows()
})

// Continue-listening + its recommendations are VOLATILE — they change the moment you play something.
// onActivated fires on the first mount AND on every return to Home from a kept-alive navigation
// (App.vue), unlike onMounted which runs once. So returning to Home refreshes the resume hero without
// factory-refreshing the whole page (#1 — the rest stays cached).
onActivated(async () => {
  if (auth.isAuthenticated || !auth.loaded) {
    await loadContinue()
    // Recommended = peers of the most-recent play (v1 heuristic; PRD-041 supersedes).
    if (continueItems.value[0]) await loadRecommended()
  } else {
    continueSection.phase.value = 'ready' // signed out: nothing to resume is the truth, not a gap
  }
})

/** Extracted so the error state can offer a real retry rather than a dead end. */
async function loadContinue(): Promise<void> {
  await continueSection.load(async () => {
      // A failure here must NOT collapse to "nothing in progress" — that silently swaps the resume
      // hero for the discover hero and drops Recommended, with no sign anything went wrong.
      const positions = await getPlaybackList()
      // `finished` episodes are not in progress. Without it, an episode you heard to the end sat
      // here forever — the last cadence save left it parked seconds from its end — and reopening it
      // resumed at end-epsilon and immediately auto-advanced away again.
      const inProgress = positions
        .filter((p) => p.position_seconds > 1 && !p.finished)
        .slice(0, 6)
      const hydrated = await Promise.all(
        inProgress.map((p) =>
          getEpisode(p.slug)
            .then((detail) => ({ detail, position: p.position_seconds }))
            .catch(() => null), // one unreadable episode is not an outage
        ),
      )
    return hydrated.filter((x): x is { detail: EpisodeDetail; position: number } => !!x)
  })
}
</script>

<template>
  <section>
    <!-- Adaptive hero -->
    <!-- The hero must not lie about your history. A failed playback fetch used to collapse to []
         and silently swap the resume hero for the discover hero, so a user mid-episode was told to
         start exploring and their place looked lost (#1591, S7). -->
    <SectionStatus
      v-if="auth.isAuthenticated && !continueSection.isReady.value"
      :phase="continueSection.phase.value"
      :rows="1"
      @retry="loadContinue"
    />
    <div
      v-else-if="resumeState && resumeTop"
      class="relative overflow-hidden rounded-2xl border border-border"
    >
      <img v-if="resumeArt(resumeTop.detail)" :src="resumeArt(resumeTop.detail)!" alt="" class="absolute inset-0 h-full w-full object-cover opacity-30" />
      <div class="relative p-5">
        <span class="lp-kicker text-grounded">{{ t('home.continue') }}</span>
        <h1 class="mt-1 font-display text-2xl font-extrabold leading-tight tracking-tight">
          {{ resumeTop.detail.title }}
        </h1>
        <p class="mt-1 text-sm text-muted">{{ resumeTop.detail.podcast_title }}</p>
        <div class="mt-3 h-1 rounded bg-overlay">
          <div
            class="h-1 rounded bg-accent"
            :style="{ width: Math.min(100, (resumeTop.position / (resumeTop.detail.duration_seconds || 1)) * 100) + '%' }"
          />
        </div>
        <RouterLink
          :to="{ name: 'player', params: { slug: resumeTop.detail.slug } }"
          class="mt-3 inline-flex items-center gap-2 rounded-full bg-accent px-5 py-2 font-bold text-accent-foreground no-underline"
        >
          ► {{ t('home.resume') }} · {{ formatTime(resumeTop.position) }}
        </RouterLink>
      </div>
    </div>
    <div v-else class="rounded-2xl border border-border bg-surface p-5">
      <span class="lp-kicker text-topic">{{ t('home.askKicker') }}</span>
      <h1 class="mt-2 font-display text-3xl font-extrabold leading-none tracking-tight">
        {{ t('home.askTitle') }}
      </h1>
      <p class="mt-2 text-sm text-muted">{{ t('home.askTagline') }}</p>
    </div>

    <!-- Search bar (prominent in both states) -->
    <form class="mt-3 flex gap-2" @submit.prevent="goSearch(query)">
      <label class="sr-only" for="home-search">{{ t('home.askKicker') }}</label>
      <input
        id="home-search"
        v-model="query"
        type="search"
        :placeholder="t('home.askPlaceholder')"
        class="min-w-0 flex-1 rounded-full border border-border bg-surface px-4 py-3 text-sm"
      />
      <button type="submit" class="rounded-full bg-accent px-5 py-3 font-bold text-accent-foreground">
        {{ t('search.title') }}
      </button>
    </form>

    <!-- Set-your-interests card (first visit; dismissible) — opens the cluster picker -->
    <section
      v-if="showInterestsCard"
      class="mt-4 flex items-center gap-3 rounded-2xl border border-accent bg-overlay p-4"
    >
      <span class="min-w-0 flex-1">
        <span class="block font-bold">{{ t('interests.cardTitle') }}</span>
        <span class="block text-sm text-muted">{{ t('interests.cardBody') }}</span>
      </span>
      <!--
        The two controls are ONE stacked group, not two siblings of the text.
        Side by side they were both `shrink-0` on the same row, so together they claimed the width
        the copy needed: on a 390px screen the title wrapped to two lines and the body to four, in a
        card whose whole job is a one-line ask. Stacking "Not now" under the primary button returns
        that width to the left column and puts the dismiss where it reads as secondary — beneath the
        action it declines, rather than competing beside it.
      -->
      <span class="flex shrink-0 flex-col items-stretch gap-1.5">
        <button
          type="button"
          class="rounded-full bg-accent px-4 py-2 text-sm font-bold text-accent-foreground"
          @click="pickerOpen = true"
        >
          {{ t('interests.cardCta') }}
        </button>
        <button type="button" class="text-sm text-muted" @click="dismissInterests">
          {{ t('interests.dismiss') }}
        </button>
      </span>
    </section>

    <!-- Your Week — the personal digest, in-app (#1412). The highlight of the page: the first
         curated, personalized block, above the editorial/global sections. Self-hides when
         signed-out or nothing's due. Compact/full is a synced per-user preference. -->
    <YourWeek />

    <!-- What's new — editorial ranked: a featured #1 + ranked rows, all on screen, NO scroll.
         Renders while loading and on error too (#1591): the section header is the thing that tells
         you this content exists, so hiding it on failure made an outage indistinguishable from a
         cold corpus. Only a successful-but-empty load hides — the system has nothing to show and
         there is no action the user can take. -->
    <section v-if="wnFeatured || !whatsNew.isReady.value" class="mt-7">
      <div class="mb-3 flex items-baseline justify-between">
        <h2 class="lp-section">{{ t('home.whatsNew') }}</h2>
        <RouterLink :to="{ name: 'catalog' }" class="text-sm font-bold text-accent no-underline">
          {{ t('home.browseAll') }} →
        </RouterLink>
      </div>

      <SectionStatus :phase="whatsNew.phase.value" :rows="3" @retry="loadWhatsNew" />

      <template v-if="wnFeatured">
      <!-- Featured #01 -->
      <div class="relative">
      <!-- Queue toggle in the artwork's upper-right (same over-image treatment as the player hero);
           sibling of the link, not nested in the <a>. -->
      <QueueButton
        :slug="wnFeatured.slug"
        class="absolute right-3 top-3 z-30 bg-canvas/80 backdrop-blur"
      />
      <RouterLink
        :to="{ name: 'player', params: { slug: wnFeatured.slug } }"
        class="relative block overflow-hidden rounded-2xl border border-border no-underline text-canvas-foreground"
        @click="recordDiscoverClick(wnFeatured.slug, 0)"
      >
        <img
          v-if="epArt(wnFeatured)"
          :src="epArt(wnFeatured)!"
          alt=""
          class="absolute inset-0 h-full w-full object-cover opacity-30"
        />
        <div class="absolute inset-0 bg-gradient-to-t from-canvas to-transparent" />
        <span
          class="pointer-events-none absolute left-3 top-1 font-display text-[5rem] font-extrabold leading-none text-white/10"
          aria-hidden="true"
        >01</span>
        <div class="relative flex min-h-[12rem] flex-col justify-end p-5 sm:min-h-[16rem] sm:p-6">
          <span class="lp-kicker text-grounded">{{ wnFeatured.podcast_title }}</span>
          <h3 class="mt-1 font-display text-2xl font-extrabold leading-tight tracking-tight">
            {{ wnFeatured.title }}
          </h3>
          <p class="mt-2 flex items-center gap-2 text-sm text-muted">
            <span v-if="formatDuration(wnFeatured.duration_seconds)">{{ formatDuration(wnFeatured.duration_seconds) }}</span>
            <span v-if="wnFeatured.has_gi" class="text-grounded">● {{ t('catalog.insightsBadge') }}</span>
          </p>
        </div>
      </RouterLink>
      </div>

      <!-- Ranked rows 02–06 -->
      <ul class="mt-2">
        <li v-for="(ep, i) in wnRows" :key="ep.slug" class="flex items-center gap-2">
          <RouterLink
            :to="{ name: 'player', params: { slug: ep.slug } }"
            class="group flex min-w-0 flex-1 items-center gap-4 rounded-xl px-2 py-3 no-underline text-canvas-foreground hover:bg-overlay"
            @click="recordDiscoverClick(ep.slug, i + 1)"
          >
            <span
              class="w-9 shrink-0 text-center font-display text-2xl font-extrabold tracking-tight text-disabled"
              aria-hidden="true"
            >{{ rank(i) }}</span>
            <span class="min-w-0 flex-1">
              <span class="block truncate font-bold leading-tight">{{ ep.title }}</span>
              <span class="lp-kicker mt-0.5 block">{{ ep.podcast_title }}</span>
            </span>
            <span class="shrink-0 text-muted transition group-hover:text-accent" aria-hidden="true">▶</span>
          </RouterLink>
          <QueueButton :slug="ep.slug" class="mr-1" />
        </li>
      </ul>
      </template>
    </section>

    <!-- #1261-9: browse-all entry points — otherwise the standalone
         /browse/topics and /browse/people routes are dead code. Compact
         two-link strip so the trending rails below still lead. -->
    <nav
      class="mt-6 flex flex-wrap gap-2 text-sm font-semibold"
      :aria-label="t('home.browseNavLabel')"
      data-testid="home-browse-nav"
    >
      <RouterLink
        :to="{ name: 'browse-topics' }"
        class="rounded-full border border-border bg-surface px-3 py-1.5 text-canvas-foreground no-underline transition hover:bg-overlay"
      >
        {{ t('home.browseTopics') }} →
      </RouterLink>
      <RouterLink
        :to="{ name: 'browse-people' }"
        class="rounded-full border border-border bg-surface px-3 py-1.5 text-canvas-foreground no-underline transition hover:bg-overlay"
      >
        {{ t('home.browsePeople') }} →
      </RouterLink>
    </nav>

    <!--
      #4 — one tabbed "what's hot" area instead of three stacked rails. Rising now (read-time EWMA
      anchored to today), Trending topics (last month vs its own 6-month average) and Storylines
      (theme clusters — topics discussed together) are related measures that made Home very tall when
      stacked; the tabs keep them comparable one tap apart without the height. Each panel uses v-show
      so its rail stays mounted (no refetch on switch); the tab label replaces each rail's heading.
    -->
    <section class="mt-7" data-testid="home-discovery">
      <div
        role="tablist"
        :aria-label="t('home.discoveryTabs')"
        class="mb-3 inline-flex gap-1 rounded-full border border-border bg-surface p-1"
      >
        <button
          v-for="tab in DISCOVERY_TABS"
          :key="tab.key"
          type="button"
          role="tab"
          :aria-selected="discoveryTab === tab.key"
          :data-testid="`discovery-tab-${tab.key}`"
          class="rounded-full px-3 py-1.5 text-sm font-bold transition"
          :class="
            discoveryTab === tab.key
              ? 'bg-accent text-accent-foreground'
              : 'text-muted hover:text-canvas-foreground'
          "
          @click="discoveryTab = tab.key"
        >
          {{ t(tab.label) }}
        </button>
      </div>

      <div v-show="discoveryTab === 'rising'" role="tabpanel">
        <MomentumRail
          kind="topic"
          :title="t('home.risingNow')"
          hide-heading
          @open="cardTarget = { kind: 'topic', id: $event.entity_id }"
        />
      </div>
      <div v-show="discoveryTab === 'trending'" role="tabpanel">
        <TrendingTopics hide-heading @open="cardTarget = { kind: 'topic', id: $event }" />
      </div>
      <div v-show="discoveryTab === 'storylines'" role="tabpanel">
        <Storylines hide-heading @open="storylineTarget = $event" />
      </div>
    </section>

    <!-- Trending shows (RFC-103 §show): cover-art carousel with the cadence sparkline over the art;
         cards link to the show page. Artwork joined from the loaded podcasts list by feed_id. -->
    <!-- The CATALOGUE, not `shows`: this rail shows what is trending across the corpus, which is
         mostly shows the user does not follow. `shows` would resolve almost none of their art. -->
    <TrendingShowsRail :title="t('home.trendingShows')" :podcasts="catalogue" />

    <!-- Recommended — no-scroll responsive grid -->
    <section v-if="recommended.length || (resumeState && !recSection.isReady.value)" class="mt-7">
      <h2 class="lp-section mb-3">{{ t('home.recommended') }}</h2>
      <SectionStatus :phase="recSection.phase.value" :rows="2" @retry="loadRecommended" />
      <ul class="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
        <li v-for="ep in recommended.slice(0, 8)" :key="ep.slug" class="relative">
          <QueueButton :slug="ep.slug" class="absolute right-2 top-2 z-10 bg-canvas/70 backdrop-blur" />
          <RouterLink :to="{ name: 'player', params: { slug: ep.slug } }" class="block no-underline text-canvas-foreground">
            <img v-if="epArt(ep)" :src="epArt(ep)!" alt="" class="aspect-square w-full rounded-xl object-cover bg-elevated" />
            <div v-else class="aspect-square w-full rounded-xl bg-elevated" />
            <!-- Reserved height, not just a clamp: a 1-line title beside a 2-line one still leaves
                 rows ragged. Kicker truncates so a long show name can't wrap and undo it (#1584). -->
            <div class="mt-2 line-clamp-2 min-h-[2.5rem] text-sm font-bold leading-tight">{{ ep.title }}</div>
            <div class="lp-kicker mt-0.5 truncate">{{ ep.podcast_title }}</div>
          </RouterLink>
        </li>
      </ul>
    </section>

    <InterestsPicker v-if="pickerOpen" @close="pickerOpen = false" @saved="onInterestsSaved" />

    <!-- Your shows — the shows you FOLLOW (UXS-014:102), not the corpus catalogue.
         Shown to any signed-in user, empty or not: a signed-in listener following nothing needs to
         learn the capability exists, and a section that silently vanishes can't teach it. -->
    <section v-if="auth.isAuthenticated" class="mt-7">
      <h2 class="lp-section mb-3">{{ t('home.shows') }}</h2>
      <!-- Loading/error BEFORE the empty state, or an outage renders "follow something to get
           started" to someone who follows thirty shows (#1591). -->
      <SectionStatus :phase="followsSection.phase.value" :rows="1" @retry="loadFollowedShows" />
      <!-- Empty state carries the ACTION, not a description of it. An empty section is worth
           rendering only when the user can do something about it — and then it has to actually
           offer the doing. Following is otherwise reachable only from a show page, so a prose
           nudge would send you off to find it. -->
      <div
        v-if="followsSection.isReady.value && !shows.length"
        class="rounded-xl border border-dashed border-border p-4"
      >
        <p class="text-sm text-muted">{{ t('home.showsEmpty') }}</p>
        <ul v-if="suggestedShows.length" class="mt-3 grid grid-cols-3 gap-3 sm:grid-cols-6">
          <li v-for="p in suggestedShows" :key="p.feed_id">
            <ShowTile :show="p" followable />
          </li>
        </ul>
        <RouterLink
          :to="{ name: 'catalog' }"
          class="mt-3 inline-block text-xs font-bold text-accent no-underline"
        >
          {{ t('home.showsBrowse') }}
        </RouterLink>
      </div>
      <!-- v-else-if, not v-else: during loading/error there is nothing truthful to show here, and
           a bare v-else would render an empty grid under the skeleton. -->
      <ul v-else-if="shows.length" class="grid grid-cols-3 gap-3 sm:grid-cols-4">
        <li v-for="p in visibleShows" :key="p.feed_id">
          <ShowTile :show="p" />
        </li>
        <!-- Home is a dispatch surface, not an index: cap the grid so its length stays constant
             however many shows you follow, and hand off for the rest. -->
        <li v-if="shows.length > visibleShows.length">
          <RouterLink
            :to="{ name: 'library', query: { tab: 'shows' } }"
            class="flex aspect-square items-center justify-center rounded-xl border border-dashed border-border p-2 text-center text-xs font-bold text-accent no-underline"
          >
            {{ t('home.seeAllShows', { count: shows.length }) }}
          </RouterLink>
        </li>
      </ul>
    </section>

    <EntityCard
      v-if="cardTarget"
      :kind="cardTarget.kind"
      :id="cardTarget.id"
      @close="cardTarget = null"
    />
    <StorylineCard
      v-if="storylineTarget"
      :id="storylineTarget.id"
      :label="storylineTarget.label"
      :anchor-topic-id="storylineTarget.anchor_topic_id"
      @open-topic="openStorylineTopic"
      @close="storylineTarget = null"
    />
  </section>
</template>
