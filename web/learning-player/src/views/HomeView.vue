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
import { RouterLink } from "vue-router"
import {
  getWhatsNew,
  getRecommended,
  getEpisode,
  getPlaybackList,
  getRelated,
} from "../services/api"
import type { EpisodeDetail, EpisodeSummary } from "../services/types"
import { formatTime } from "../player/transcriptSync"
import { formatDuration } from "../utils/format"
import { formatPublishDate } from '../utils/format'
import { episodeArtwork } from "../utils/episode"
import { kindPill, type InterestKind } from "../utils/interests"
import { GUIDED_START_PREF, useGuidedStart } from "../composables/useGuidedStart"
import { useAuthStore } from "../stores/auth"
import GuidedStart from "../components/GuidedStart.vue"
import { useLibraryStore } from "../stores/library"
import { allPositions } from "../services/playbackPositions"
import { localArtworkFor, localKnowledgeFor } from "../services/downloads"
import { useDownloadsStore } from "../stores/downloads"
import { anyStale, useSectionState } from "../composables/useSectionState"
import StaleNotice from "../components/StaleNotice.vue"
import { useUserPreferencesStore } from "../stores/userPreferences"
import { useInterestsStore } from "../stores/interests"
import { useCompletedStore } from "../stores/completed"
import { usePlayed } from "../composables/usePlayed"
import InterestsPicker from "../components/InterestsPicker.vue"
import KeyVoicesRail from "../components/KeyVoicesRail.vue"
import SearchSection from "../components/SearchSection.vue"
import CollectionsTeaser from "../components/CollectionsTeaser.vue"
import RevisitRail from "../components/RevisitRail.vue"
import SectionHeading from "../components/SectionHeading.vue"
import EpisodeActions from "../components/EpisodeActions.vue"
import EpisodeTile from "../components/EpisodeTile.vue"
import CardRail from "../components/CardRail.vue"
import SectionStatus from "../components/SectionStatus.vue"
import RecapPrompt from "../components/RecapPrompt.vue"
import YourWeek from "../components/YourWeek.vue"

const { t, locale } = useI18n()
const auth = useAuthStore()
const library = useLibraryStore()
const userPrefs = useUserPreferencesStore()
const interests = useInterestsStore()
const completed = useCompletedStore()
const { isPlayed } = usePlayed()

// What's new is the listener's own world, newest first (operator 2026-10-07) — `scope` says
// whether it is (`yours`) or the every-show fallback (`all`), and the kicker says the same.
const whatsNew = useSectionState<{ items: EpisodeSummary[]; scope: "yours" | "all" }>(
  { items: [], scope: "all" },
  { cacheKey: "home.whatsnew.v2" },
)
const latest = computed(() => whatsNew.data.value.items)
/**
 * Continue gets the same contract as every other section (#1591, S7): a playback outage must not
 * silently swap the resume hero for the discover hero. An outage that looks like a new account is
 * the exact defect #1591 exists to kill.
 */
const continueSection = useSectionState<{ detail: EpisodeDetail; position: number }[]>([], {
  cacheKey: "home.continue",
})
/**
 * Recommended, from the best basis there is (operator 2026-10-07): "more like" the latest in-progress
 * listen, else picks from what the listener follows (`/recommended`), else nothing — the section does
 * not render without a basis. It never repeats What's new: an episode already there is dropped here.
 */
const recSection = useSectionState<{ items: EpisodeSummary[]; basis: "listening" | "interests" | "none" }>(
  { items: [], basis: "none" },
  { cacheKey: "home.recommended.v2" },
)
const recBasis = computed(() => recSection.data.value.basis)
const recommended = computed(() => {
  const inWhatsNew = new Set(latest.value.map((e) => e.slug))
  return recSection.data.value.items.filter((e) => !inWhatsNew.has(e.slug))
})
// One rail of up to 12 (operator 2026-10-08: a horizontal rail, like every other Home row). The
// fetch asks for 12 rather than the api default of 6; the route's own ceiling is 25 and the
// similarity merge caps at 50, so 12 is well inside both.
const RECOMMENDED_FETCH = 12
// A played episode is finished — it drops out of Continue (PL.6). Reactive: it disappears the
// moment mark-as-played toggles, no refetch.
// `isPlayed`, not `completed.has`: the server rail still offers an episode whose last position was
// the end of it, and reading the hand-marked list alone left it sitting in "Jump back in" with
// nothing left to jump back into.
const continueItems = computed(() =>
  continueSection.data.value.filter((x) => !isPlayed(x.detail.slug))
)

/** Home's Discover strip: one chip per Trends kind, in the Interests page's order. */
const DISCOVER_CHIPS: { kind: InterestKind; label: string; testid: string }[] = [
  { kind: "topic", label: "home.tabTopics", testid: "home-discover-topics" },
  { kind: "person", label: "home.tabPeople", testid: "home-discover-people" },
  { kind: "theme", label: "home.themes", testid: "home-discover-themes" },
  { kind: "storyline", label: "home.storylines", testid: "home-discover-storylines" },
]

// First-Home dismissible "set your interests" card → opens the picker (PRD-043 FR4 / 3.5).
const pickerOpen = ref(false)
// Only offer the "choose interests" card to users who have NOT already picked any — the bug was it
// showed even to users with a full interest set. Gate on the store being loaded so it never flashes
// before we know, and it hides the instant interests exist.
/**
 * The guided start (operator 2026-10-07) — for a NEW listener (no interests yet), and for as long as
 * a run they started is not finished, so following through step 1 does not make step 2 vanish.
 * State lives in `useGuidedStart` (synced): `active` once shown, `done` once finished, `restart` when
 * Settings asked for it again (then it runs from step 1 whatever the listener already has). "Not now"
 * snoozes it for three days (operator 2026-10-08). An account that already chose interests before
 * this existed never sees it unless it is restarted.
 */
const guided = useGuidedStart()
const guidedState = guided.state
const showInterestsCard = computed(
  () =>
    auth.isAuthenticated &&
    interests.loaded &&
    !guided.isSnoozed() &&
    guidedState.value !== "done" &&
    (interests.ids.length === 0 || guidedState.value === "active" || guidedState.value === "restart")
)
watch(showInterestsCard, (on) => {
  if (on && guidedState.value === undefined) void userPrefs.set(GUIDED_START_PREF, "active")
})
/** Close the flow and rebuild Home from what was just chosen. */
async function finishGuidedStart(): Promise<void> {
  await userPrefs.set(GUIDED_START_PREF, "done")
  await Promise.all([loadWhatsNew(), loadRecommended()])
}

/**
 * The first name for the welcome, or null. An email-link account's `name` can be the address
 * itself, and "Welcome, marko@…" reads worse than no name at all.
 */
const welcomeName = computed<string | null>(() => {
  const raw = auth.user?.name?.trim() ?? ""
  if (!raw || raw.includes("@")) return null
  return raw.split(/\s+/)[0] ?? null
})

/** "Not now": snooze the guide (synced across devices), it comes back in three days. */
function dismissInterests(): void {
  guided.snooze()
}

async function onInterestsSaved(): Promise<void> {
  // NO `dismissInterests()` here (2026-09-25). The picker now writes the saved set into the
  // interests store, so `showInterestsCard` goes false on its own — the card hides because the user
  // HAS interests, which is the real reason, not because we marked the offer as declined.
  //
  // It was actively wrong twice over. `dismissInterests` persists to localStorage AND write-syncs
  // the preference across devices, so choosing interests permanently suppressed the card: clear
  // them again later and it would never come back. And it masked the bug — saving from Home LOOKED
  // right while the store stayed empty, so the defect only showed on the Profile path, where
  // nothing set the flag. That is what `PersonalisationTests.test10` hit.
  //
  // Declining still works: "Not now" calls `dismissInterests` directly, which is the only place
  // that should.
  //
  // Re-pull discovery so a personalized order (when the flag is on) takes effect immediately.
  await loadWhatsNew()
  // Interests are a basis for Recommended: it can appear the moment they are saved.
  await loadRecommended()
}

/**
 * What's New load (#1591). Failure is NOT collapsed into emptiness — that collapse is why a total
 * API outage rendered the same page as a brand-new account. A rejection lands in the error phase,
 * which renders a message and a retry.
 */
function loadWhatsNew(): Promise<void> {
  return whatsNew.load(() => getWhatsNew(5))
}

/** #1591 — Recommended, same contract: a rejection is an error phase, not an empty list. */
function loadRecommended(): Promise<void> {
  if (!auth.isAuthenticated) return Promise.resolve()
  const top = continueItems.value[0]
  return recSection.load(async () => {
    if (top) {
      return { items: (await getRelated(top.detail.slug, RECOMMENDED_FETCH)).items, basis: "listening" as const }
    }
    const r = await getRecommended(RECOMMENDED_FETCH)
    return { items: r.items, basis: r.basis }
  })
}

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
    await Promise.all([loadWhatsNew(), loadContinue()])
    await loadRecommended()
  } finally {
    retrying.value = false
  }
}
const resumeState = computed(() => auth.isAuthenticated && continueItems.value.length > 0)
// "What's new": a featured #01 hero, then rows 02–05 as a numbered chart (operator 2026-09-14).
// Five items, not six — the chart ends at 05 (operator 2026-09-17).
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
/**
 * The What's new kicker says WHOSE new it is (operator 2026-10-07): the listener's shows and topics,
 * or — the fallback when they follow nothing that matches — every show. Recommended's kicker says
 * the other half: picked from what they listen to.
 */
const whatsNewKicker = computed(() => {
  const scope =
    whatsNew.data.value.scope === "yours" ? t("home.whatsNewYours") : t("home.whatsNewAll")
  return whatsNewSince.value
    ? t("home.whatsNewKicker", { scope, date: whatsNewSince.value })
    : scope
})
const resumeTop = computed(() => continueItems.value[0] ?? null)
// "Jump back in" (H.5): every OTHER in-progress listen beyond the resume hero, so multiple active
// episodes are all reachable (cap a handful for the rail).
const jumpBackIn = computed(() => continueItems.value.slice(1, 8))
const resumeArt = episodeArtwork
const epArt = episodeArtwork

onMounted(async () => {
  // Load the user's chosen interests so the "choose interests" card only shows when there are none
  // (fire-and-forget: the card stays hidden until this resolves, then appears only if empty).
  if (auth.isAuthenticated) void interests.ensureLoaded()
  // Completed set drives the Continue filter (PL.6); fire-and-forget so it doesn't gate first paint.
  if (auth.isAuthenticated) void completed.ensureLoaded()
  await loadWhatsNew()
  // Follow / favourite state on the tiles reads the library store.
  if (auth.isAuthenticated) void library.ensureLoaded()
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
  // What's new follows what you follow, so a return to Home re-reads it (operator 2026-10-07): a
  // listener sent to Discover by the welcome card's "Follow shows" comes back to their own shows,
  // not the every-show fallback they left. In place — only the first load shows a skeleton.
  if (whatsNew.isReady.value) void loadWhatsNew()
  // Recommended = peers of the most-recent play (v1 heuristic; PRD-041 supersedes). Only compute it
  // when we don't already have it, so returning to Home doesn't re-flicker it either.
  // Every return re-reads it, in place: interests chosen or an episode started elsewhere changes the
  // basis, and the section must appear (or move to "more like") without a reload.
  await loadRecommended()
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
        <p class="lp-show-name mt-1 text-sm text-muted" :title="resumeTop.detail.podcast_title ?? undefined">{{ resumeTop.detail.podcast_title }}</p>
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
        <!-- Resume alone. The queue link that sat beside it is GONE (operator 2026-09-27): the
             masthead carries a queue entry at every width and on every screen, which is strictly
             more reach than a control that only exists while something is in progress — the exact
             gap it was added for in the first place (2026-09-19). Two ways in, one of them
             conditional, is one too many. -->
        <div class="mt-3 flex items-center gap-2">
          <RouterLink
            :to="{ name: 'player', params: { slug: resumeTop.detail.slug }, query: { play: '1' } }"
            data-testid="home-resume"
            class="inline-flex h-11 items-center gap-2 rounded-full bg-accent px-5 font-bold text-accent-foreground no-underline"
          >
            ▶ {{ t("home.resume") }} · {{ formatTime(resumeTop.position) }}
          </RouterLink>
        </div>
      </div>
    </div>
    <!-- The "Ask across every episode" title + the search box + its topic chips are ONE unit and
         moved together, LOWER on the page (H.3) — see the Search section below Your Week. The top of
         Home is now the resume hero (when resuming) then Jump-back-in + the trending rails (H.4). -->

    <!-- Welcome + set-your-interests card (first visit, until interests exist or "Not now").
         Operator 2026-10-07, from a beta user who did not know what "Choose interests" / "Not now"
         did: it was one muted line with two text links. It is now a WELCOME — the person's name,
         what choosing interests changes, and two real buttons. (It was reduced to a line in #1964
         for competing with the Search button; a designed card that leads the page is the answer to
         that, not a quieter line.) Button labels are unchanged: device journeys find them by text. -->
    <GuidedStart
      v-if="showInterestsCard"
      :welcome-name="welcomeName"
      :restart="guidedState === 'restart'"
      @choose-interests="pickerOpen = true"
      @dismiss="dismissInterests"
      @finish="finishGuidedStart"
    />

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
      <!-- The standard rail and the standard episode tile (operator 2026-10-05: every rail looks the
           same on every page). This was the one episode rail that hand-rolled its tile — square art,
           2-line title, show name UNDER the title, no actions — and drifted from every other rail. -->
      <CardRail>
        <li v-for="it in jumpBackIn" :key="it.detail.slug" class="lp-rail-item">
          <EpisodeTile
            :episode="it.detail"
            :progress="it.position / (it.detail.duration_seconds || 1)"
          />
        </li>
      </CardRail>
    </section>

    <!-- Your Week — the personal digest LEADS the content, right under Continue / Jump-back-in
         (operator review): the forward-looking "what to play next" is the reason to open Home.
         It renders only once the week has something to review (operator 2026-10-08). -->
    <YourWeek :key="railKey" />

    <!-- Search — the same section Discover renders (operator 2026-10-05: one screen family). Trends
         left Home (operator 2026-10-07) and lives on Discover only. Left half on `lg`, with the
         revisit rail and the boards teaser stacked in the right half; stacked on phones. -->
    <div class="lg:flex lg:items-start lg:gap-8">
      <div class="lg:w-1/2 lg:pr-4">
        <SearchSection prefix="home" />
      </div>
      <div class="lg:w-1/2">
        <RevisitRail />
        <CollectionsTeaser />
      </div>
    </div>

    <!-- A one-line look BACK, pointing at the recap in Profile (#1914). Self-hides when there is
         nothing to look back on. -->
    <RecapPrompt />

    <!-- Recommended — up here, right after the look back, and What's new at the bottom (operator
         2026-10-07): picks for YOU lead, the corpus-wide feed follows. A horizontal rail of the same
         EpisodeTile every rail uses (operator 2026-10-08): the 2-column grid with "Show 4 more"
         spent two screens on a phone, and the rail keeps it to one row. -->
    <section v-if="recommended.length" class="mt-7" data-testid="home-recommended">
      <SectionHeading
        :title="t('home.recommended')"
        :kicker="recBasis === 'listening' ? t('home.recommendedKicker') : t('home.recommendedFromFollows')"
      />
      <SectionStatus :phase="recSection.phase.value" :rows="2" @retry="loadRecommended" />
      <CardRail>
        <li v-for="ep in recommended" :key="ep.slug" class="lp-rail-item">
          <EpisodeTile :episode="ep" />
        </li>
      </CardRail>
    </section>

    <!-- Discover entry points: a small title line over one row of four chips, each deep-linking
         into Browse's Trends section on its kind (`?trends=<kind>` selects and scrolls to it).

         Operator 2026-10-08: the inline "Discover" lead-in pushed the fourth chip onto a second
         row, and plain grey chips said nothing about the kind. The lead-in is now a title above, so
         the chips get the whole width, and each chip wears its kind's colour from `kindPill` — the
         same pill the Interests page and episode notes use, so the colour already means "topic" or
         "storyline" wherever the listener meets it. -->
    <nav class="mt-6" :aria-label="t('home.browseNavLabel')" data-testid="home-browse-nav">
      <p class="lp-kicker mb-2">{{ t("home.discoverLabel") }}</p>
      <div class="flex flex-nowrap items-center gap-2 overflow-x-auto text-sm [scrollbar-width:none]">
        <RouterLink
          v-for="chip in DISCOVER_CHIPS"
          :key="chip.kind"
          :to="{ name: 'browse', query: { trends: chip.kind } }"
          :data-testid="chip.testid"
          class="shrink-0 whitespace-nowrap rounded-full px-3.5 py-1.5 font-semibold no-underline transition hover:brightness-125"
          :class="kindPill(chip.kind)"
        >
          {{ t(chip.label) }}
        </RouterLink>
      </div>
    </nav>

    <!-- Key voices (wave-G): the people most present in your corpus. Moved up to sit right after
         Trending shows and before Recommended (operator review) — a quiet discovery rail. Self-hides
         when empty. -->
    <KeyVoicesRail v-if="auth.isAuthenticated" />


    <!-- What's new keeps HALF the desktop row (operator 2026-09-17): a ranked chart stretched across
         the full column read as a half-empty row. It shared the row with Trending shows until that
         left Home (2026-10-05); the half width stays because the chart is still narrow by nature.

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
        :kicker="whatsNewKicker"
      >
        <template #action>
          <RouterLink
            :to="{ name: 'browse', query: { tab: 'episodes' }, hash: '#catalog' }"
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
          <!-- Tapping the card OPENS the episode, paused, like every episode card. Only Resume starts
               playback (operator 2026-10-05). -->
          <RouterLink
            :to="{ name: 'player', params: { slug: wnFeatured.slug } }"
            class="relative block overflow-hidden rounded-2xl border border-border no-underline text-canvas-foreground"
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
              <span class="lp-kicker lp-show-name text-grounded" :title="wnFeatured.podcast_title ?? undefined">{{ wnFeatured.podcast_title }}</span>
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

        <!-- Ranked rows 02–06 (operator 2026-09-14): the numbered chart look, same capped column as
             the featured card so they line up. Each row carries the #01 card's actions, stacked in a
             column (operator 2026-10-05) — the side-by-side cluster is what once crushed the title.
             Tapping a row opens it. -->
        <ul class="mt-2 max-w-3xl">
          <li
            v-for="(ep, i) in wnRows"
            :key="ep.slug"
            class="flex items-center gap-1 border-b border-border py-3 last:border-b-0"
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
                <span class="lp-kicker lp-show-name mt-0.5" :title="ep.podcast_title ?? undefined">{{ ep.podcast_title }}</span>
              </span>
            </RouterLink>
            <!-- The row's actions, STACKED in one column on the right (operator 2026-10-05): queue and
                 ⋯, with the heart inside the ⋯ (operator 2026-10-07) — three stacked targets made each
                 row taller than its content. The row opens the episode (paused, like every card — only
                 Resume plays). Outside the link — never an interactive inside an interactive. A
                 vertical stack costs height, not width, so the title keeps the row's width; side by
                 side, a cluster like this once crushed it to one word per line. -->
            <EpisodeActions :slug="ep.slug" hide-favorite class="shrink-0 flex-col" />
          </li>
        </ul>
      </template>
      </section>

      <!-- Trending shows is NOT on Home (operator 2026-10-05): it lives on Discover, as the same
           tile rail as every other. Home's version was a different shape — full-width cover bands with
           a sparkline — for a section with the same name. -->
    </div>

    <InterestsPicker v-if="pickerOpen" trigger="home_prompt" @close="pickerOpen = false" @saved="onInterestsSaved" />

  </section>
</template>
