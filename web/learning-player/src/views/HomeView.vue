<script setup lang="ts">
/**
 * Home — the Learning Hub (PRD-042 / UXS-012). Adaptive hero: resume-state (Continue) when
 * signed-in with in-progress history, else discover-state ("Ask your library" + Featured).
 * Corpus search is prominent in both states. Sections (What's new / Recommended / Your shows)
 * hide cleanly when empty. Login-first (RFC-120): this view is authed-only. All data from the
 * real /api/app/* surface.
 */
import { computed, onActivated, onMounted, ref, watch } from "vue"
import { useI18n } from "vue-i18n"
defineOptions({ name: "HomeView" }) // stable name for <keep-alive :include> (App.vue)
import { RouterLink, useRouter } from "vue-router"
import {
  getDiscover,
  getEpisode,
  getPlaybackList,
  getPodcasts,
  getRelated,
  getTrendingTopics,
  recordDiscoverClick,
} from "../services/api"
import type { EpisodeDetail, EpisodeSummary, Podcast } from "../services/types"
import { formatTime } from "../player/transcriptSync"
import { formatDuration } from "../utils/format"
import { formatPublishDate } from '../utils/format'
import { episodeArtwork } from "../utils/episode"
import { useAuthStore } from "../stores/auth"
import { useLibraryStore } from "../stores/library"
import { allPositions } from "../services/playbackPositions"
import { localArtworkFor, localKnowledgeFor } from "../services/downloads"
import { useDownloadsStore } from "../stores/downloads"
import { anyStale, useSectionState } from "../composables/useSectionState"
import { useTrendingScope } from "../composables/useTrendingScope"
import StaleNotice from "../components/StaleNotice.vue"
import { useUserPreferencesStore } from "../stores/userPreferences"
import { useInterestsStore } from "../stores/interests"
import { useCompletedStore } from "../stores/completed"
import EntityCard from "../components/EntityCard.vue"
import InterestsPicker from "../components/InterestsPicker.vue"
import KeyVoicesRail from "../components/KeyVoicesRail.vue"
import DiscoveryExplorer from "../components/DiscoveryExplorer.vue"
import CollectionsTeaser from "../components/CollectionsTeaser.vue"
import RevisitRail from "../components/RevisitRail.vue"
import SectionHeading from "../components/SectionHeading.vue"
import { useIsDesktop } from "../composables/useMediaQuery"
import TrendingShowsRail from "../components/TrendingShowsRail.vue"
import EpisodeActions from "../components/EpisodeActions.vue"
import EpisodeTile from "../components/EpisodeTile.vue"
import QueueButton from "../components/QueueButton.vue"
import SectionStatus from "../components/SectionStatus.vue"
import StorylineCard from "../components/StorylineCard.vue"
import RecapPrompt from "../components/RecapPrompt.vue"
import YourWeek from "../components/YourWeek.vue"

const INTERESTS_DISMISSED_KEY = "lp.interests.dismissed"

const { t, locale } = useI18n()
const isDesktop = useIsDesktop()
const router = useRouter()
const auth = useAuthStore()
const library = useLibraryStore()
const userPrefs = useUserPreferencesStore()
const interests = useInterestsStore()
const completed = useCompletedStore()

// #2030 — the app-level trending lens (Corpus ⇄ My listening). Home owns the toggle; every
// trending surface reads the same stored preference, so one choice governs the rails and the
// topic/storyline card momentum badges. Auth-gated: signed out, it is forced to corpus.
const { scope: trendingScope } = useTrendingScope()

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
// Four visible, then four more per tap (operator 2026-09-19) — one grid row at both breakpoints.
// The fetch asks for 12 rather than the api default of 6, so "show more" is worth the tap; the
// route's own ceiling is 25 and the similarity merge caps at 50, so 12 is well inside both.
const RECOMMENDED_PAGE = 4
const RECOMMENDED_FETCH = 12
const recommendedShown = ref(RECOMMENDED_PAGE)
const visibleRecommended = computed(() => recommended.value.slice(0, recommendedShown.value))
// Re-collapse when the set itself changes — it is keyed off the most recent play, so it changes
// under you as you listen, and leaving it expanded would silently grow the page.
watch(recommended, () => {
  recommendedShown.value = RECOMMENDED_PAGE
})
// An episode the user marked played is finished — it drops out of Continue (PL.6). Reactive: it
// disappears the moment mark-as-played toggles, no refetch.
const continueItems = computed(() =>
  continueSection.data.value.filter((x) => !completed.has(x.detail.slug))
)
const query = ref("")

// Trending-topic chip → open the topic entity card (overlay), same surface as Search.
const cardTarget = ref<{ kind: "person" | "topic"; id: string } | null>(null)
// Discovery = one shared list (DiscoveryList) over three ENTITY KINDS — the tabs pick WHAT (topics /
// storylines / people), and two little switches pick HOW: sort (Rising = by velocity, Trending = by
// volume) and scope (Corpus ⇄ Mine). All three kinds come from one `/trending` endpoint carrying
// both signals, so Rising⇄Trending is a client-side re-sort. This replaced three bespoke rails
// (MomentumRail / TrendingTopics / Storylines) so every tab reads identically (operator 2026-09-14).
// A tapped discovery row opens the entity: topics/people → the entity card overlay; storylines → the
// storyline overlay (its id is the anchor topic, resolved inside DiscoveryList). The tabs + sort +
// scope controls live in the shared DiscoveryExplorer now (operator 2026-09-14).
function onDiscoveryOpen(p: { kind: "topic" | "storyline" | "person"; id: string }): void {
  if (p.kind === "storyline") storylineTarget.value = p.id
  else cardTarget.value = { kind: p.kind, id: p.id }
}
// #9 / F4.5 — a tapped storyline opens as a dismissible OVERLAY card (StorylineCard), the same
// lightweight-and-in-context pattern a topic uses, rather than navigating to a full page (operator
// 2026-09-14: the two must feel the same). The `/storyline/:id` route stays for deep-links/sharing;
// StorylineCard adds a `?storyline=` history entry so hardware Back closes the card, not the page.
const storylineTarget = ref<string | null>(null)

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
  return recSection.load(async () => (await getRelated(top.detail.slug, RECOMMENDED_FETCH)).items)
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
// "What's new": a featured #01 hero, then rows 02–05 as a numbered chart (operator 2026-09-14).
// Five items, not six — the chart ends at 05 (operator 2026-09-17). The section now shares a
// desktop row with Trending shows, and a top five is a rounder thing to end on than a top six.
const wnFeatured = computed(() => latest.value[0] ?? null)
/**
 * "Since <date>" for What's new — the publish date of the newest episode ALREADY on screen.
 *
 * No fetch and no invented metric: it is the date of `wnFeatured`, which this section renders
 * anyway. Blank when the episode carries no date, so the kicker disappears rather than reading
 * "Since —".
 */
const whatsNewSince = computed(() =>
  wnFeatured.value?.publish_date
    ? formatPublishDate(wnFeatured.value.publish_date, locale.value)
    : "",
)
const wnRows = computed(() => latest.value.slice(1, 5))
// Ranked "chart" rows 02–05 beneath the #01 hero (operator 2026-09-14): the numbered leaderboard
// look is the point. wnRows starts at latest[1], so row i is rank i+2.
const rank = (i: number): string => String(i + 2).padStart(2, "0")
// The row's discover-position telemetry must count a click on THIS EPISODE only. The wrapping <li>
// catches every bubbled click inside the card — action buttons, the "Read more" toggle, and the
// show-name link that navigates AWAY to the podcast — so record only when the clicked anchor is one
// of the episode's own links (artwork/title → player), identified by the slug in its href.
function onWnRowClick(e: MouseEvent, slug: string, position: number): void {
  const href = (e.target as HTMLElement | null)?.closest("a")?.getAttribute("href")
  if (href && href.includes(slug)) recordDiscoverClick(slug, position)
}
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
    <!-- FULL WIDTH on desktop (operator 2026-09-18). It was `max-w-3xl` to match the "What's new"
         featured card, but What's new became a half-width column later and its featured card is now
         166px — so the pairing that justified 766px has not existed for a while, and the hero was
         the only thing on the page matching neither the 542px column nor the 1114px full row.

         Full rather than half because nothing sits beside it: at half width the top-right of the
         page would simply be empty, which is a worse first impression than a wide card. -->
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
        <!-- `play=1`: Resume means RESUME (operator 2026-09-18). This reads as a play control and
             behaved as a link — it opened the episode paused at the saved position, so continuing
             took a second tap on a transport further down the page. -->
        <div class="mt-3 flex items-center gap-2">
          <RouterLink
            :to="{ name: 'player', params: { slug: resumeTop.detail.slug }, query: { play: '1' } }"
            data-testid="home-resume"
            class="inline-flex h-11 items-center gap-2 rounded-full bg-accent px-5 font-bold text-accent-foreground no-underline"
          >
            ▶ {{ t("home.resume") }} · {{ formatTime(resumeTop.position) }}
          </RouterLink>
          <!-- The queue, reachable without playing something first (operator 2026-09-19). Its only
               entrances were the full player and the mini-player, and the mini-player is only there
               while something is loaded — so with nothing playing the queue could not be opened at
               all. This is the one place on Home that is already about "what I am listening to", so
               it is where the way in belongs.

               Same list glyph `QueueButton` draws, deliberately: that button ADDS to the queue and
               this one OPENS it, and they are the same object.

               Plated the way `ShowRow` plates its over-artwork controls, not left as a quiet muted
               outline: this hero sits ON the episode artwork, so a `border-border text-muted`
               circle disappeared into whatever the cover happened to be. Resuming is still the
               primary action — that is the filled accent pill — and this reads as secondary
               without depending on the image behind it being calm. -->

          <RouterLink
            :to="{ name: 'queue' }"
            data-testid="home-open-queue"
            class="lp-tap inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-full border border-white/25 bg-black/55 text-white no-underline shadow-lg backdrop-blur-sm transition hover:text-white"
            :aria-label="t('queue.title')"
            :title="t('queue.title')"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
              class="h-5 w-5"
              aria-hidden="true"
            >
              <path d="M13 6H3" /><path d="M13 12H3" /><path d="M13 18H3" /><path d="M15 16l2 2 4-4" />
            </svg>
          </RouterLink>
        </div>
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
    <!-- Kicker carries the QUANTITY, title says what the section is (operator 2026-09-18).
         A kicker that names the category ("IN PROGRESS" over "Jump back in") says the same thing
         twice, which is the redundancy the ask block had. A number does not. -->
    <section v-if="jumpBackIn.length" class="mt-7" data-testid="home-jump-back-in">
      <SectionHeading
        :title="t('home.jumpBackIn')"
        :kicker="t('home.jumpBackInCount', jumpBackIn.length, { named: { count: jumpBackIn.length } })"
      />
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

    <!-- Your Week — the personal digest LEADS the content, right under Continue / Jump-back-in
         (operator review): the forward-looking "what to play next" is the reason to open Home. -->
    <YourWeek :key="railKey" />

    <!-- Discovery: the shared tabbed DiscoveryExplorer (Topics / Storylines / People + sort/scope),
         after the digest. Capped at 5 rows here; Discover uses the same section capped at 10.

         TITLED, with the same heading Discover gives it (operator 2026-09-17). Untitled it sat
         directly under Your Week's first-run line, so on an empty digest it read as Your Week's own
         content arriving without a heading — when in fact Your Week had rendered nothing and this is
         the next section entirely.

         Half width from `lg`, matching Discover: a trend row is a short label against a sparkline +
         multiplier + follow, and across the full column those two clusters sit ~500px apart. -->
    <!-- Discovery and Revisit share the row on `lg` (operator 2026-09-18). Discovery has been
         half-width since it was titled, which left the right half of the column EMPTY on desktop —
         this is the gap the rail was asked to fill, so the two sit side by side rather than the
         rail pushing everything below it down a screen. Stacked on phones, where there is one
         column and no gap to fill. -->
    <div class="lg:flex lg:items-start lg:gap-8">
      <section class="mt-7 lg:w-1/2 lg:pr-4" data-testid="home-discovery">
        <!-- Five rows on desktop, three on a phone (operator 2026-09-18). Trends sits beside the
             revisit rail, which is taller, so at three rows the column ended ~110px short and left
             a hole under it. A prop cannot be set by a media query in CSS, hence `useIsDesktop`. -->
        <DiscoveryExplorer
          :collapsed="isDesktop ? 5 : 3"
          :title="t('browse.trendsTitle')"
          @open="onDiscoveryOpen"
        />
      </section>
      <div class="lg:w-1/2">
        <RevisitRail />
      </div>
    </div>

    <!-- A one-line look BACK, pointing at the recap in Profile (#1914). Placed under Your Week so
         the forward-looking digest ("what to play") comes first and this is the quieter follow-up.
         Self-hides when there is nothing to look back on. -->
    <RecapPrompt />

    <!-- Search (H.3): the "Ask across every episode" title + box moved DOWN here together from under
         the hero, so the top of Home leads with the resume hero + the trending rails. Topic chips are
         the tappable entry points. testids unchanged across the move. -->
    <!-- The ask box and the boards teaser share the row on `lg` (operator 2026-09-18). The ask box
         is deliberately capped (a full-bleed input flung the Search button to the far right), so
         the right of this row was empty on desktop — the same gap the revisit rail filled beside
         Trends. Stacked on phones. -->
    <div class="lg:flex lg:items-start lg:gap-8">
    <section class="mt-7 lg:w-1/2" data-testid="home-search-section">
      <!-- The same heading tier as every other section. It was `font-display text-2xl` — a third
           title size on one page — and its kicker ("Ask across every episode") restated the title
           beneath it. No kicker: the pattern is a count or a date, and this section has neither. -->
      <SectionHeading :title="t('home.askTitle')" />
      <!-- Cap the ask box: full-bleed on a wide desktop flung the Search button to the far right
           with an oversized input between (mobile-first layout, unbounded wide). -->
      <form class="lp-search mt-3 flex gap-2" @submit.prevent="goSearch(query)">
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
      <!-- Just TWO examples (operator): a single row that always fits, rather than scrolling a
           longer list that got visibly clipped at the screen edge. -->
      <div
        v-if="heroTopics.length"
        data-testid="home-topic-chips"
        class="mt-3 flex flex-wrap gap-2"
      >
        <button
          v-for="tp in heroTopics.slice(0, 2)"
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
      <div class="lg:w-1/2">
        <CollectionsTeaser />
      </div>
    </div>

    <!-- What's new and Trending shows SHARE a desktop row, half each (operator 2026-09-17). Both are
         narrow-by-nature lists — a ranked chart and a stack of show bands — that were each stretched
         across the full column, so the page became a single tall stack of half-empty rows. On a phone
         they go back to one over the other, unchanged.

         `items-start` so the shorter of the two does not stretch to match the taller. -->
    <div class="lg:flex lg:items-start lg:gap-6">
      <!-- What's new — editorial ranked: a featured #1 + ranked rows, all on screen, NO scroll.
           Renders while loading and on error too (#1591): the section header is the thing that tells
           you this content exists, so hiding it on failure made an outage indistinguishable from a
           cold corpus. Only a successful-but-empty load hides — the system has nothing to show and
           there is no action the user can take. -->
      <section v-if="wnFeatured || !whatsNew.isReady.value" class="mt-7 min-w-0 lg:w-1/2">
      <SectionHeading
        :title="t('home.whatsNew')"
        :kicker="whatsNewSince ? t('home.whatsNewSince', { date: whatsNewSince }) : null"
      >
        <template #action>
          <RouterLink
            :to="{ name: 'browse', query: { tab: 'episodes' } }"
            class="text-sm font-bold text-accent no-underline"
          >
            {{ t("home.browseAll") }} →
          </RouterLink>
        </template>
      </SectionHeading>

      <SectionStatus :phase="whatsNew.phase.value" :rows="3" @retry="loadWhatsNew" />

      <template v-if="wnFeatured">
        <!-- Featured #01 — capped width: full-bleed on a wide desktop stretched the background artwork
           (opacity-30 cover) across the whole page and it visibly lost resolution. -->
        <div class="relative max-w-3xl">
          <!-- Shared EpisodeActions row (favourite/queue/download/collect) in the artwork's upper-right;
           sibling of the link, not nested in the <a>. The featured card is wide (max-w-3xl) so the
           four icons fit without wrapping. -->
          <EpisodeActions :slug="wnFeatured.slug" overlay class="absolute right-3 top-3 z-30" />
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

        <!-- Ranked rows 02–06 (operator 2026-09-14): the numbered chart look restored, same capped
             column as the featured card so they line up. The per-row EpisodeActions cluster that
             crushed the title into one-word-per-line on a phone is DROPPED — the "package it" trade
             (favourite/queue/download/collect live on the player and the #01 hero, not on every row).
             The wrapping <li> keeps the discover-position telemetry (fires only for the episode's
             own links, not a stray bubbled click). -->
        <ul class="mt-2 max-w-3xl">
          <li
            v-for="(ep, i) in wnRows"
            :key="ep.slug"
            class="flex items-center gap-1"
            @click="onWnRowClick($event, ep.slug, i + 1)"
          >
            <RouterLink
              :to="{ name: 'player', params: { slug: ep.slug } }"
              class="group flex min-w-0 flex-1 items-center gap-3 rounded-xl px-2 py-2.5 no-underline text-canvas-foreground hover:bg-overlay"
            >
              <span
                class="w-6 shrink-0 text-center font-display text-xl font-extrabold tracking-tight text-disabled"
                aria-hidden="true"
                >{{ rank(i) }}</span
              >
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
            <!-- Queue sits OUTSIDE the link — never an interactive inside an interactive (the same
                 rule EpisodeRow's `#trailing` slot follows). It is the one action worth carrying on
                 a row you are only scanning, because it is the only one you would take WITHOUT
                 opening the episode, which is what a "what's new" list is for. The rest (favourite /
                 download / collect) still live on the player and the #01 hero, where you have
                 already committed to the episode — so this narrows the earlier "no actions on every
                 row" decision rather than undoing it (operator 2026-09-16). -->
            <QueueButton :slug="ep.slug" class="shrink-0" />
          </li>
        </ul>
      </template>
      </section>

      <!-- Trending shows (RFC-103 §show): cover-art bands with the cadence sparkline woven over the
           art; each links to the show page. Artwork is joined from the loaded podcasts list by
           feed_id.

           The CATALOGUE, not `shows`: this rail shows what is trending across the corpus, which is
           mostly shows the user does not follow. `shows` would resolve almost none of their art. -->
      <div class="min-w-0 lg:w-1/2">
        <TrendingShowsRail
          :key="railKey"
          :title="t('home.trendingShows')"
          :podcasts="catalogue"
          :scope="trendingScope"
        />
      </div>
    </div>

    <!-- Discover entry points (operator 2026-09-14): a compact one-line strip — a "Discover" lead-in
         + three chips deep-linking into Browse's Trends section on the matching kind.

         These pointed at a separate /trends page, which was a second, thinner copy of a section
         Browse already renders — tapping a chip left the hub for a page with the same three tabs and
         less around them. The operator called it "small pages that should not exist" (2026-09-18).
         /trends is deleted; `?trends=<kind>` selects the kind and scrolls it into view. -->
    <nav
      class="mt-6 flex flex-wrap items-center gap-2 text-sm"
      :aria-label="t('home.browseNavLabel')"
      data-testid="home-browse-nav"
    >
      <span class="font-bold text-muted">{{ t("home.discoverLabel") }}</span>
      <RouterLink
        :to="{ name: 'browse', query: { trends: 'topic' } }"
        data-testid="home-discover-topics"
        class="rounded-full border border-border bg-surface px-3 py-1 font-semibold text-canvas-foreground no-underline transition hover:bg-overlay"
      >
        {{ t("home.tabTopics") }}
      </RouterLink>
      <RouterLink
        :to="{ name: 'browse', query: { trends: 'storyline' } }"
        data-testid="home-discover-storylines"
        class="rounded-full border border-border bg-surface px-3 py-1 font-semibold text-canvas-foreground no-underline transition hover:bg-overlay"
      >
        {{ t("home.storylines") }}
      </RouterLink>
      <RouterLink
        :to="{ name: 'browse', query: { trends: 'person' } }"
        data-testid="home-discover-people"
        class="rounded-full border border-border bg-surface px-3 py-1 font-semibold text-canvas-foreground no-underline transition hover:bg-overlay"
      >
        {{ t("home.tabPeople") }}
      </RouterLink>
    </nav>

    <!-- Key voices (wave-G): the people most present in your corpus. Moved up to sit right after
         Trending shows and before Recommended (operator review) — a quiet discovery rail. Self-hides
         when empty. -->
    <KeyVoicesRail v-if="auth.isAuthenticated" />

    <!-- Recommended — no-scroll responsive grid -->
    <section v-if="recommended.length || (resumeState && !recSection.isReady.value)" class="mt-7">
      <SectionHeading :title="t('home.recommended')" />
      <SectionStatus :phase="recSection.phase.value" :rows="2" @retry="loadRecommended" />
      <!-- The SAME tile the Discover grid uses (operator 2026-09-17), not a second copy of it.
           This grid was hand-rolled here: square artwork, overlaid actions, show name and title —
           EpisodeTile's shape, re-implemented. Keeping two of them is how they drifted apart in the
           first place (title-above-show here, show-above-title there; clamped there, unclamped
           here). One component, so a change to the tile reaches every grid that uses it. -->
      <!-- 2 on a phone, 4 on desktop (operator 2026-09-17). Deliberately NOT the browse grids' 3/4:
           Recommended is a short curated set on the home screen, so its tiles stay large enough to
           read at a glance rather than matching a dense catalogue.

           FOUR to begin with — one row on both breakpoints — then a control that reveals the rest
           in place (operator 2026-09-19). There is no "see all" here because there is nowhere for
           it to go: nothing in the app is a recommendations page, and inventing a route to satisfy
           the shape of a link would be worse than expanding where you already are. It sits at the
           very bottom of Home, so an expansion pushes nothing else down. -->
      <ul class="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <li v-for="ep in visibleRecommended" :key="ep.slug" class="h-full">
          <EpisodeTile :episode="ep" />
        </li>
      </ul>
      <button
        v-if="recommended.length > visibleRecommended.length"
        type="button"
        class="mt-4 w-full rounded-xl border border-border py-2.5 text-sm font-bold text-accent transition hover:bg-overlay"
        data-testid="home-recommended-more"
        @click="recommendedShown += RECOMMENDED_PAGE"
      >
        {{ t("ec.moreEpisodes", { count: recommended.length - visibleRecommended.length }) }}
      </button>
    </section>

    <InterestsPicker v-if="pickerOpen" @close="pickerOpen = false" @saved="onInterestsSaved" />

    <EntityCard
      v-if="cardTarget"
      :kind="cardTarget.kind"
      :id="cardTarget.id"
      @close="cardTarget = null"
    />
    <!-- A tapped storyline opens ON TOP as a dismissible overlay (operator 2026-09-14), the same
         wrapper a topic uses — not a full-page navigation. -->
    <StorylineCard
      v-if="storylineTarget"
      :id="storylineTarget"
      @close="storylineTarget = null"
    />
  </section>
</template>
