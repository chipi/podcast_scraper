<script setup lang="ts">
/**
 * Home — the Learning Hub (PRD-042 / UXS-012). Adaptive hero: resume-state (Continue) when
 * signed-in with in-progress history, else discover-state ("Ask your library" + Featured).
 * Corpus search is prominent in both states. Sections (What's new / Recommended / Your shows)
 * hide cleanly when empty. Login-first (RFC-120): this view is authed-only. All data from the
 * real /api/app/* surface.
 */
import { computed, onActivated, onMounted, ref } from "vue"
import { useI18n } from "vue-i18n"
defineOptions({ name: "HomeView" }) // stable name for <keep-alive :include> (App.vue)
import { RouterLink, useRouter } from "vue-router"
import Tabs from "../components/Tabs.vue"
import { panelAttrs, type TabSpec } from "../components/tabs"
import {
  getDiscover,
  getEpisode,
  getPlaybackList,
  getPodcasts,
  getRelated,
  getTrendingTopics,
  recordDiscoverClick,
} from "../services/api"
import type { EpisodeDetail, EpisodeSummary, Podcast, Storyline } from "../services/types"
import { formatTime } from "../player/transcriptSync"
import { formatDuration } from "../utils/format"
import { episodeArtwork } from "../utils/episode"
import { useAuthStore } from "../stores/auth"
import { useLibraryStore } from "../stores/library"
import { allPositions } from "../services/playbackPositions"
import { localArtworkFor, localKnowledgeFor } from "../services/downloads"
import { useDownloadsStore } from "../stores/downloads"
import { anyStale, useSectionState } from "../composables/useSectionState"
import StaleNotice from "../components/StaleNotice.vue"
import { useUserPreferencesStore } from "../stores/userPreferences"
import { useInterestsStore } from "../stores/interests"
import { useCompletedStore } from "../stores/completed"
import EntityCard from "../components/EntityCard.vue"
import InterestsPicker from "../components/InterestsPicker.vue"
import KeyVoicesRail from "../components/KeyVoicesRail.vue"
import MomentumRail from "../components/MomentumRail.vue"
import TrendingShowsRail from "../components/TrendingShowsRail.vue"
import EpisodeActions from "../components/EpisodeActions.vue"
import SectionStatus from "../components/SectionStatus.vue"
import ShowTile from "../components/ShowTile.vue"
import Storylines from "../components/Storylines.vue"
import TrendingTopics from "../components/TrendingTopics.vue"
import RecapPrompt from "../components/RecapPrompt.vue"
import YourWeek from "../components/YourWeek.vue"

const INTERESTS_DISMISSED_KEY = "lp.interests.dismissed"

const { t } = useI18n()
const router = useRouter()
const auth = useAuthStore()
const library = useLibraryStore()
const userPrefs = useUserPreferencesStore()
const interests = useInterestsStore()
const completed = useCompletedStore()

// USERPREFS-1 key for the "set your interests" dismissal (gh #1213).
// localStorage remains the fast-path fallback until the server responds.
const INTERESTS_DISMISSED_PREF_KEY = "lp.interests.dismissed"

const whatsNew = useSectionState<EpisodeSummary[]>([], { cacheKey: "home.whatsnew" })
const latest = computed(() => whatsNew.data.value)
/**
 * Following and Continue get the same contract as every other section (#1591, S7).
 *
 * These two were the last holdouts on `.catch(() => [])`, and they are the two most personal
 * sections on the page — so a library outage rendered "follow something to get started" to a user
 * who follows thirty shows, and a playback outage silently swapped the resume hero for the discover
 * hero. An outage that looks like a new account is the exact defect #1591 exists to kill; I fixed
 * it in the sections around these and not in these.
 */
// Was `useSectionState<null>` with the catalogue assigned as a side effect, which put the one
// thing worth caching outside the section that fetched it — so offline this rail had nothing to
// hydrate from and rendered an error card over follows the library store already had (#1909).
const followsSection = useSectionState<Podcast[]>([], { cacheKey: "home.catalogue" })
const continueSection = useSectionState<{ detail: EpisodeDetail; position: number }[]>([], {
  cacheKey: "home.continue",
})
const recSection = useSectionState<EpisodeSummary[]>([], { cacheKey: "home.recommended" })
const recommended = computed(() => recSection.data.value)
// An episode the user marked played is finished — it drops out of Continue (PL.6). Reactive: it
// disappears the moment mark-as-played toggles, no refetch.
const continueItems = computed(() =>
  continueSection.data.value.filter((x) => !completed.has(x.detail.slug))
)
const query = ref("")

// Trending-topic chip → open the topic entity card (overlay), same surface as Search.
const cardTarget = ref<{ kind: "person" | "topic"; id: string } | null>(null)
// #4 — Rising now / Trending topics / Storylines are three views of "what's hot"; stacked, they made
// Home very tall. Fold them into one tabbed area (Rising default). v-show (not v-if) keeps each rail
// mounted so switching tabs doesn't refetch; TrendingTopics still lazy-loads via its own observer.
const DISCOVERY_TABS = [
  { key: "rising", labelKey: "home.risingNow" },
  { key: "trending", labelKey: "home.trending" },
  { key: "storylines", labelKey: "home.storylines" },
] as const
type DiscoveryTab = (typeof DISCOVERY_TABS)[number]["key"]
const discoveryTab = ref<DiscoveryTab>("rising")
// Shared tab strip (#1594 item 7): this strip had roles and panels but no `aria-controls` pair
// between them, and no arrow-key movement.
const discoveryTabs = computed<TabSpec<DiscoveryTab>[]>(() =>
  DISCOVERY_TABS.map((tb) => ({
    key: tb.key,
    label: t(tb.labelKey),
    testid: `discovery-tab-${tb.key}`,
  }))
)
// #9 / F4.5 — a tapped storyline opens its own full PAGE (titled with the storyline, listing member
// topics + top episodes + people), keyed by the anchor topic id.
function openStoryline(s: Storyline): void {
  if (s.anchor_topic_id) void router.push({ name: "storyline", params: { id: s.anchor_topic_id } })
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
    !interestsDismissed.value
)

function dismissInterests(): void {
  interestsDismissed.value = true
  try {
    localStorage.setItem(INTERESTS_DISMISSED_KEY, "1")
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

const catalogue = computed<Podcast[]>(() => followsSection.data.value)

/**
 * Refresh everything the notice is speaking for.
 *
 * The four rails below own their own fetches and are not reachable from here, so they are remounted
 * by key rather than reached into — an explicit retry is exactly when losing their scroll-deferred
 * fetch is the point. The sections this view owns are reloaded directly.
 */
const railKey = ref(0)
const retrying = ref(false)
async function retryStale(): Promise<void> {
  if (retrying.value) return
  retrying.value = true
  railKey.value += 1
  try {
    await Promise.all([loadWhatsNew(), loadFollowedShows(), loadContinue()])
    if (continueItems.value[0]) await loadRecommended()
  } finally {
    retrying.value = false
  }
}
const resumeState = computed(() => auth.isAuthenticated && continueItems.value.length > 0)
// Editorial ranked "What's new": a featured #1 + ranked rows — all on screen, no scroll.
const wnFeatured = computed(() => latest.value[0] ?? null)
const wnRows = computed(() => latest.value.slice(1, 6))
const rank = (i: number) => String(i + 2).padStart(2, "0")
const resumeTop = computed(() => continueItems.value[0] ?? null)
// "Jump back in" (H.5): every OTHER in-progress listen beyond the resume hero, so multiple active
// episodes are all reachable (cap a handful for the rail).
const jumpBackIn = computed(() => continueItems.value.slice(1, 8))
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
    return cat
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
      }
  )
})

/**
 * What the empty state offers. Following is only discoverable today from a show page, so an empty
 * "Your shows" that merely *describes* following makes the user go find it. These tiles carry the
 * follow control itself, so the section teaches the capability and completes it in one place.
 */
const suggestedShows = computed<Podcast[]>(() =>
  catalogue.value.filter((p) => !library.has(p.feed_id)).slice(0, 6)
)

/**
 * Home caps the shows grid and links out for the rest (#1584). Unbounded, this section grows without
 * limit as the corpus does — it was the "taking all that real estate" half of the complaint. 5 and
 * 11 leave room for the See-all tile to complete a row at 3 columns (mobile) and 4 (desktop).
 */
const SHOWS_ON_HOME = 11
const visibleShows = computed(() => shows.value.slice(0, SHOWS_ON_HOME))
const epArt = episodeArtwork

/**
 * Topic chips under the hero search field (#1964 follow-up, UXS-012 §103).
 *
 * The hero's kicker is `topic`-toned by spec, but it was the only topic-coloured thing on the
 * screen — so the colour read as decoration rather than as "this is topic territory". These chips
 * give it siblings AND make the hero answerable: it says "ask across every episode" and then
 * offered an empty box you had to already know what to type into.
 *
 * Reuses `getTrendingTopics()`, which is memoised and already fetched for the momentum rail, so
 * this costs no extra request. Silent on failure — a hero that renders without chips is fine; one
 * that renders an error where its examples should be is not.
 */
const heroTopics = ref<Array<{ id: string; label: string }>>([])

async function loadHeroTopics(): Promise<void> {
  try {
    const res = await getTrendingTopics()
    heroTopics.value = (res.topics ?? [])
      .slice(0, 4)
      .map((t) => ({ id: t.topic_id, label: t.topic_label || t.topic_id.split(":").pop() || "" }))
      .filter((t) => t.label)
  } catch {
    heroTopics.value = []
  }
}

function goSearch(q: string): void {
  const term = q.trim()
  if (term) void router.push({ name: "search", query: { q: term } })
}

onMounted(async () => {
  void loadHeroTopics()
  try {
    interestsDismissed.value = localStorage.getItem(INTERESTS_DISMISSED_KEY) === "1"
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
  // Completed set drives the Continue filter (PL.6); fire-and-forget so it doesn't gate first paint.
  if (auth.isAuthenticated) void completed.ensureLoaded()
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
  if (!(auth.isAuthenticated || !auth.loaded)) {
    continueSection.phase.value = "ready" // signed out: nothing to resume is the truth, not a gap
    return
  }
  // First activation (nothing loaded yet) → a real load with its skeleton. Every RETURN after that
  // refreshes in place with no loading flicker — the kept-alive list stays on screen (operator: the
  // reload glitch on returning to Home was the complaint).
  // One path, not two. `refreshContinueQuietly` existed to avoid the skeleton a full reload used
  // to flash on return — but `useSectionState` revalidates in place now and only shows `loading`
  // when it has nothing, so the flicker it guarded against cannot happen either way.
  //
  // Keeping it had become actively wrong: it wrote `data` directly, bypassing the section, so a
  // FAILED quiet refresh left the rail looking current — no stale flag, and therefore no notice
  // and no retry. The section's own failure handling is exactly what should run here.
  await loadContinue()
  // Recommended = peers of the most-recent play (v1 heuristic; PRD-041 supersedes). Only compute it
  // when we don't already have it, so returning to Home doesn't re-flicker it either.
  if (continueItems.value[0] && !recSection.isReady.value) await loadRecommended()
})

type ContinueItem = { detail: EpisodeDetail; position: number }

/**
 * Continue-listening rebuilt from what THIS DEVICE recorded (#1909 follow-up).
 *
 * The rail is built from `GET /playback`, so with no network it disappeared — on a device that had
 * written every one of those positions itself and, for a downloaded episode, holds the title, show
 * and artwork too. The operator's case is the whole point: five episodes downloaded, on a plane,
 * wanting to carry on where they left off.
 *
 * Only episodes we can describe are listed. A position for an episode that was never downloaded has
 * no title on this device, and a row reading "ep-7f3a" is worse than no row.
 */
async function localContinue(): Promise<ContinueItem[]> {
  const downloads = useDownloadsStore()
  const items: ContinueItem[] = []
  for (const p of allPositions()) {
    if (p.finished || p.seconds <= 1) continue
    const entry = downloads.entry(p.slug)
    if (!entry || entry.state !== "downloaded") continue
    const known = await localKnowledgeFor(p.slug)
    const detail =
      known?.detail ??
      ({
        slug: p.slug,
        title: entry.title ?? p.slug,
        feed_id: entry.feedId ?? "",
        podcast_title: entry.showTitle ?? null,
        publish_date: null,
        duration_seconds: entry.durationSeconds ?? null,
        episode_image_url: null,
        feed_image_url: null,
        artwork_url: localArtworkFor(p.slug),
        summary_title: null,
        summary_bullets: [],
        summary_text: null,
        has_transcript: !!entry.transcriptPath,
        has_summary: false,
        has_gi: false,
        has_kg: false,
        has_bridge: false,
      } as EpisodeDetail)
    items.push({ detail, position: p.seconds })
    if (items.length >= 6) break
  }
  return items
}

async function fetchContinue(): Promise<ContinueItem[]> {
  // A failure here must NOT collapse to "nothing in progress" — that would blank the resume hero
  // (the top now shows an error skeleton, not a fabricated hero) and drop Recommended, with no sign
  // anything went wrong.
  let positions
  try {
    positions = await getPlaybackList()
  } catch (err) {
    // The device knows where you are. Falling back to it beats an empty rail, and beats a cached
    // copy of the server's answer — this is the record, not a copy of one.
    const local = await localContinue()
    if (local.length) return local
    throw err
  }
  // `finished` episodes are not in progress. Without it, an episode you heard to the end sat here
  // forever — the last cadence save left it parked seconds from its end — and reopening it resumed
  // at end-epsilon and immediately auto-advanced away again.
  const inProgress = positions.filter((p) => p.position_seconds > 1 && !p.finished).slice(0, 6)
  const hydrated = await Promise.all(
    inProgress.map(
      (p) =>
        getEpisode(p.slug)
          .then((detail) => ({ detail, position: p.position_seconds }))
          .catch(() => null) // one unreadable episode is not an outage
    )
  )
  return hydrated.filter((x): x is ContinueItem => !!x)
}

/** Extracted so the error state can offer a real retry rather than a dead end. */
async function loadContinue(): Promise<void> {
  await continueSection.load(fetchContinue)
}
</script>

<template>
  <section>
    <!-- One page-level statement, above the rails that are showing it (#1909). The rails keep their
         content; this says why it may be out of date, and carries the retry they no longer have. -->
    <StaleNotice v-if="anyStale" :busy="retrying" @retry="retryStale" />

    <!-- Adaptive hero -->
    <!-- The hero must not lie about your history. A failed playback fetch used to collapse to []
         and silently blank the resume hero, so a user mid-episode lost their place (#1591, S7); it
         now shows the error skeleton below instead. (The old "discover hero" fallback in this slot
         was removed when search moved down — H.3; the top is resume-or-error now.) -->
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
      <img
        v-if="resumeArt(resumeTop.detail)"
        :src="resumeArt(resumeTop.detail)!"
        alt=""
        class="absolute inset-0 h-full w-full object-cover opacity-30"
      />
      <div class="relative p-5">
        <span class="lp-kicker text-grounded">{{ t("home.continue") }}</span>
        <h1 class="mt-1 font-display text-2xl font-extrabold leading-tight tracking-tight">
          {{ resumeTop.detail.title }}
        </h1>
        <p class="mt-1 text-sm text-muted">{{ resumeTop.detail.podcast_title }}</p>
        <div class="mt-3 h-1 rounded bg-overlay">
          <div
            class="h-1 rounded bg-accent"
            :style="{
              width:
                Math.min(
                  100,
                  (resumeTop.position / (resumeTop.detail.duration_seconds || 1)) * 100
                ) + '%',
            }"
          />
        </div>
        <RouterLink
          :to="{ name: 'player', params: { slug: resumeTop.detail.slug } }"
          data-testid="home-resume"
          class="mt-3 inline-flex h-11 items-center gap-2 rounded-full bg-accent px-5 font-bold text-accent-foreground no-underline"
        >
          ► {{ t("home.resume") }} · {{ formatTime(resumeTop.position) }}
        </RouterLink>
      </div>
    </div>
    <!-- The "Ask across every episode" title + the search box + its topic chips are ONE unit and
         moved together, LOWER on the page (H.3) — see the Search section below Your Week. The top of
         Home is now the resume hero (when resuming) then Jump-back-in + the trending rails (H.4). -->

    <!-- Set-your-interests card (first visit; dismissible) — opens the cluster picker -->
    <!-- One quiet line, not a bordered accent card (#1964).
         As a card it was the third pitch before any content, and the worst-composed object on the
         page: a 1px orange stroke fighting the solid orange Search button ~40px above it, a title
         wrapping in a column with 200px of unused width, and "Not now" aligned to neither the
         button's left nor its centre. It is an offer, not an announcement — so it gets a line. -->
    <section v-if="showInterestsCard" class="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1">
      <span class="min-w-0">
        <span class="block text-sm text-muted">{{ t("interests.cardTitle") }}</span>
      </span>
      <!--
        The two controls are ONE stacked group, not two siblings of the text.
        Side by side they were both `shrink-0` on the same row, so together they claimed the width
        the copy needed: on a 390px screen the title wrapped to two lines and the body to four, in a
        card whose whole job is a one-line ask. Stacking "Not now" under the primary button returns
        that width to the left column and puts the dismiss where it reads as secondary — beneath the
        action it declines, rather than competing beside it.
      -->
      <span class="flex shrink-0 items-center gap-4">
        <button type="button" class="text-sm font-bold text-accent" @click="pickerOpen = true">
          {{ t("interests.cardCta") }}
        </button>
        <button type="button" class="text-sm text-muted" @click="dismissInterests">
          {{ t("interests.dismiss") }}
        </button>
      </span>
    </section>

    <!-- Your Week — the personal digest, in-app (#1412). The first curated, personalized block.
         Self-hides when signed-out. Compact/full is a synced per-user preference.

         ORDERED BY WHETHER IT HAS ANYTHING TO SAY (#1978). It used to sit unconditionally above the
         editorial sections, which is right once it is delivering. For a brand-new account it is
         not: measured, it renders 373px with zero episode links — four rows of "will land here" —
         between the hero and "What's new", the app's most distinctive component. That is ~44% of
         the first viewport spent promising future value to the one audience with no history, which
         is every beta tester on their first run.
         #1591 decided this must TEACH rather than self-hide, and that stands — so it is not hidden
         and not reordered (a `v-if` on "has content" is a chicken-and-egg: the component that
         reports the state is the one being unmounted). It TEACHES IN ONE LINE instead, exactly as
         the set-your-interests offer above it does since #1964: an explanation is a line, not an
         announcement. Populated, it renders in full as before. -->
    <!-- Jump back in (H.5): every OTHER in-progress listen beyond the resume hero, so more than one
         active episode is reachable, not just the most recent. -->
    <section v-if="jumpBackIn.length" class="mt-7" data-testid="home-jump-back-in">
      <h2 class="lp-section mb-3">{{ t("home.jumpBackIn") }}</h2>
      <ul class="flex gap-3 overflow-x-auto pb-1">
        <li v-for="it in jumpBackIn" :key="it.detail.slug" class="w-40 shrink-0">
          <RouterLink
            :to="{ name: 'player', params: { slug: it.detail.slug } }"
            class="block no-underline text-canvas-foreground"
          >
            <img
              v-if="resumeArt(it.detail)"
              :src="resumeArt(it.detail)!"
              alt=""
              loading="lazy"
              class="aspect-square w-full rounded-xl bg-elevated object-cover"
            />
            <div v-else class="aspect-square w-full rounded-xl bg-elevated" />
            <div class="mt-2 h-1 rounded bg-overlay">
              <div
                class="h-1 rounded bg-accent"
                :style="{
                  width:
                    Math.min(100, (it.position / (it.detail.duration_seconds || 1)) * 100) + '%',
                }"
              />
            </div>
            <div class="mt-1 line-clamp-2 text-sm font-bold leading-tight">
              {{ it.detail.title }}
            </div>
            <div class="lp-kicker mt-0.5">{{ it.detail.podcast_title }}</div>
          </RouterLink>
        </li>
      </ul>
    </section>

    <!-- Key voices (wave-G): the people most present in your corpus. Self-hides when empty. -->
    <KeyVoicesRail v-if="auth.isAuthenticated" />

    <!-- Discovery moved UP (H.4): the "what's hot" tabs (Rising / Trending / Storylines) surface
         right after Jump-back-in, before the personal digest, instead of folded low on the page. -->
    <section class="mt-7" data-testid="home-discovery">
      <Tabs
        v-model="discoveryTab"
        :tabs="discoveryTabs"
        :label="t('home.discoveryTabs')"
        id-prefix="discovery"
        variant="pill"
        class="mb-3"
      />

      <div v-show="discoveryTab === 'rising'" v-bind="panelAttrs('discovery', 'rising')">
        <MomentumRail
          kind="topic"
          :title="t('home.risingNow')"
          hide-heading
          @open="cardTarget = { kind: 'topic', id: $event.entity_id }"
        />
      </div>
      <div v-show="discoveryTab === 'trending'" v-bind="panelAttrs('discovery', 'trending')">
        <TrendingTopics
          :key="railKey"
          hide-heading
          @open="cardTarget = { kind: 'topic', id: $event }"
        />
      </div>
      <div v-show="discoveryTab === 'storylines'" v-bind="panelAttrs('discovery', 'storylines')">
        <Storylines :key="railKey" hide-heading @open="openStoryline" />
      </div>
    </section>

    <YourWeek :key="railKey" />

    <!-- A one-line look BACK, pointing at the recap in Profile (#1914). Placed under Your Week so
         the forward-looking digest ("what to play") comes first and this is the quieter follow-up.
         Self-hides when there is nothing to look back on. -->
    <RecapPrompt />

    <!-- Search (H.3): the "Ask across every episode" title + box moved DOWN here together from under
         the hero, so the top of Home leads with the resume hero + the trending rails. Topic chips are
         the tappable entry points. testids unchanged across the move. -->
    <section class="mt-7" data-testid="home-search-section">
      <span class="lp-kicker text-topic">{{ t("home.askKicker") }}</span>
      <h2 class="mt-2 font-display text-2xl font-extrabold leading-none tracking-tight">
        {{ t("home.askTitle") }}
      </h2>
      <!-- Cap the ask box: full-bleed on a wide desktop flung the Search button to the far right
           with an oversized input between (mobile-first layout, unbounded wide). -->
      <form class="mt-3 flex max-w-2xl gap-2" @submit.prevent="goSearch(query)">
        <label class="sr-only" for="home-search">{{ t("home.askKicker") }}</label>
        <input
          id="home-search"
          v-model="query"
          type="search"
          :placeholder="t('home.askPlaceholder')"
          data-testid="home-search-input"
          class="h-11 min-w-0 flex-1 rounded-full border border-border bg-surface px-4 text-sm"
        />
        <button
          type="submit"
          data-testid="home-search-submit"
          class="h-11 shrink-0 rounded-full bg-accent px-5 font-bold text-accent-foreground"
        >
          {{ t("search.title") }}
        </button>
      </form>
      <div
        v-if="heroTopics.length"
        data-testid="home-topic-chips"
        class="mt-3 flex flex-wrap gap-2"
      >
        <button
          v-for="tp in heroTopics"
          :key="tp.id"
          type="button"
          data-testid="home-topic-chip"
          class="rounded-full border border-topic/40 px-3 py-1.5 text-sm font-semibold text-topic transition hover:bg-overlay"
          @click="goSearch(tp.label)"
        >
          {{ tp.label }}
        </button>
      </div>
    </section>

    <!-- What's new — editorial ranked: a featured #1 + ranked rows, all on screen, NO scroll.
         Renders while loading and on error too (#1591): the section header is the thing that tells
         you this content exists, so hiding it on failure made an outage indistinguishable from a
         cold corpus. Only a successful-but-empty load hides — the system has nothing to show and
         there is no action the user can take. -->
    <section v-if="wnFeatured || !whatsNew.isReady.value" class="mt-7">
      <div class="mb-3 flex items-baseline justify-between">
        <h2 class="lp-section">{{ t("home.whatsNew") }}</h2>
        <RouterLink
          :to="{ name: 'browse', query: { tab: 'episodes' } }"
          class="text-sm font-bold text-accent no-underline"
        >
          {{ t("home.browseAll") }} →
        </RouterLink>
      </div>

      <SectionStatus :phase="whatsNew.phase.value" :rows="3" @retry="loadWhatsNew" />

      <template v-if="wnFeatured">
        <!-- Featured #01 — capped width: full-bleed on a wide desktop stretched the background artwork
           (opacity-30 cover) across the whole page and it visibly lost resolution. -->
        <div class="relative max-w-3xl">
          <!-- Action row (favourite/download/queue) in the artwork's upper-right; sibling of the link,
           not nested in the <a>. -->
          <EpisodeActions :slug="wnFeatured.slug" class="absolute right-3 top-3 z-30" />
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
              >01</span
            >
            <div
              class="relative flex min-h-[12rem] flex-col justify-end p-5 sm:min-h-[16rem] sm:p-6"
            >
              <span class="lp-kicker text-grounded">{{ wnFeatured.podcast_title }}</span>
              <h3 class="mt-1 font-display text-2xl font-extrabold leading-tight tracking-tight">
                {{ wnFeatured.title }}
              </h3>
              <p class="mt-2 flex items-center gap-2 text-sm text-muted">
                <span v-if="formatDuration(wnFeatured.duration_seconds)">{{
                  formatDuration(wnFeatured.duration_seconds)
                }}</span>
                <span v-if="wnFeatured.has_gi" class="text-grounded"
                  >● {{ t("catalog.insightsBadge") }}</span
                >
              </p>
            </div>
          </RouterLink>
        </div>

        <!-- Ranked rows 02–06 — same capped column as the featured card above, so they line up. -->
        <ul class="mt-2 max-w-3xl">
          <li v-for="(ep, i) in wnRows" :key="ep.slug" class="flex items-center gap-2">
            <RouterLink
              :to="{ name: 'player', params: { slug: ep.slug } }"
              class="group flex min-w-0 flex-1 items-center gap-3 rounded-xl px-2 py-2.5 no-underline text-canvas-foreground hover:bg-overlay"
              @click="recordDiscoverClick(ep.slug, i + 1)"
            >
              <span
                class="w-6 shrink-0 text-center font-display text-xl font-extrabold tracking-tight text-disabled"
                aria-hidden="true"
                >{{ rank(i) }}</span
              >
              <!-- #15 — small square artwork beside the rank so rows 02–06 aren't text-only; falls back
                 to a plain tile when the episode has no art (never a broken image). -->
              <img
                v-if="epArt(ep)"
                :src="epArt(ep)!"
                alt=""
                loading="lazy"
                class="h-11 w-11 shrink-0 rounded-lg bg-elevated object-cover"
              />
              <span v-else class="h-11 w-11 shrink-0 rounded-lg bg-elevated" aria-hidden="true" />
              <span class="min-w-0 flex-1">
                <span class="block font-bold leading-tight">{{ ep.title }}</span>
                <span class="lp-kicker mt-0.5 block">{{ ep.podcast_title }}</span>
              </span>
              <span
                class="shrink-0 text-muted transition group-hover:text-accent"
                aria-hidden="true"
                >▶</span
              >
            </RouterLink>
            <EpisodeActions :slug="ep.slug" class="mr-1" />
          </li>
        </ul>
      </template>
    </section>

    <!-- #1261-9: browse-all entry points. Compact two-link strip into the Browse hub (the trending
         rails now sit higher, after Jump-back-in — H.4).

         The original comment here claimed this strip was what kept the standalone
         /browse/topics and /browse/people routes from being dead code. It never did: both links
         below point at `{ name: 'browse', query: { tab } }` — the HUB — and those two standalone
         routes still have zero links anywhere in the app (audit, #2013). They are reachable only
         by typing the URL. Left in place deliberately as deep-link targets, but nothing in the UI
         leads to them, and a reader should not be told otherwise. -->
    <nav
      class="mt-6 flex flex-wrap gap-2 text-sm font-semibold"
      :aria-label="t('home.browseNavLabel')"
      data-testid="home-browse-nav"
    >
      <RouterLink
        :to="{ name: 'browse', query: { tab: 'topics' } }"
        class="rounded-full border border-border bg-surface px-3 py-1.5 text-canvas-foreground no-underline transition hover:bg-overlay"
      >
        {{ t("home.browseTopics") }} →
      </RouterLink>
      <RouterLink
        :to="{ name: 'browse', query: { tab: 'people' } }"
        class="rounded-full border border-border bg-surface px-3 py-1.5 text-canvas-foreground no-underline transition hover:bg-overlay"
      >
        {{ t("home.browsePeople") }} →
      </RouterLink>
    </nav>

    <!-- Trending shows (RFC-103 §show): cover-art carousel with the cadence sparkline over the art;
         cards link to the show page. Artwork joined from the loaded podcasts list by feed_id. -->
    <!-- The CATALOGUE, not `shows`: this rail shows what is trending across the corpus, which is
         mostly shows the user does not follow. `shows` would resolve almost none of their art. -->
    <TrendingShowsRail :key="railKey" :title="t('home.trendingShows')" :podcasts="catalogue" />

    <!-- Recommended — no-scroll responsive grid -->
    <section v-if="recommended.length || (resumeState && !recSection.isReady.value)" class="mt-7">
      <h2 class="lp-section mb-3">{{ t("home.recommended") }}</h2>
      <SectionStatus :phase="recSection.phase.value" :rows="2" @retry="loadRecommended" />
      <ul class="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
        <li v-for="ep in recommended.slice(0, 8)" :key="ep.slug" class="relative h-full">
          <EpisodeActions :slug="ep.slug" class="absolute right-2 top-2 z-10" />
          <RouterLink
            :to="{ name: 'player', params: { slug: ep.slug } }"
            class="flex h-full flex-col no-underline text-canvas-foreground"
          >
            <img
              v-if="epArt(ep)"
              :src="epArt(ep)!"
              alt=""
              class="aspect-square w-full rounded-xl object-cover bg-elevated"
            />
            <div v-else class="aspect-square w-full rounded-xl bg-elevated" />
            <!--
              Neither the title nor the show name is clipped (#2004 items 3/3b).

              The title clamped at two lines with a reserved height and the show name truncated to
              one, on the reasoning that a 1-line title beside a 2-line one leaves rows ragged
              (#1584). The requirement is real; the method cost the ends of long names, and in the
              Recommended grid the clamped title actually overflowed INTO the show name — an
              ellipsis at line two AND a visible third line, because the clamp computed but the
              overflow still painted.

              Rows are now even because the CARD is even: the link is a flex column filling its grid
              cell, the artwork is fixed, and the text block takes the rest. Both lines wrap freely.
            -->
            <div class="mt-2 text-sm font-bold leading-tight">{{ ep.title }}</div>
            <div class="lp-kicker mt-0.5">{{ ep.podcast_title }}</div>
          </RouterLink>
        </li>
      </ul>
    </section>

    <InterestsPicker v-if="pickerOpen" @close="pickerOpen = false" @saved="onInterestsSaved" />

    <!-- Your shows — the shows you FOLLOW (UXS-014:102), not the corpus catalogue.
         Shown to any signed-in user, empty or not: a signed-in listener following nothing needs to
         learn the capability exists, and a section that silently vanishes can't teach it. -->
    <section v-if="auth.isAuthenticated" class="mt-7">
      <h2 class="lp-section mb-3">{{ t("home.shows") }}</h2>
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
        <p class="text-sm text-muted">{{ t("home.showsEmpty") }}</p>
        <ul v-if="suggestedShows.length" class="mt-3 grid grid-cols-3 gap-3 sm:grid-cols-6">
          <li v-for="p in suggestedShows" :key="p.feed_id">
            <ShowTile :show="p" followable />
          </li>
        </ul>
        <RouterLink
          :to="{ name: 'catalog' }"
          class="mt-3 inline-block text-xs font-bold text-accent no-underline"
        >
          {{ t("home.showsBrowse") }}
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
            {{ t("home.seeAllShows", { count: shows.length }) }}
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
  </section>
</template>
