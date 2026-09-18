<script setup lang="ts">
/**
 * Your Library (UXS-014) — the hub for everything per-user, tabbed: Saved (favourited things, in
 * per-kind sections — episodes, insights, …) · Highlights · Revisit · Queue · Recent. One place,
 * tabbed; the Saved tab grows a new section as new favourite kinds arrive. Auth-gated.
 */
import { computed, onActivated, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
defineOptions({ name: 'LibraryView' }) // stable name for <keep-alive :include> (App.vue)
import { RouterLink, useRoute } from 'vue-router'
import Tabs from '../components/Tabs.vue'
import { panelAttrs, type TabSpec } from '../components/tabs'
import { useCaptureStore } from '../stores/capture'
import { useLibraryStore } from '../stores/library'
import { useCollectionsStore } from '../stores/collections'
import StaleNotice from '../components/StaleNotice.vue'
import { useResurfacingStore } from '../stores/resurfacing'
import { useFavoritesStore } from '../stores/favorites'
import FavoriteButton from '../components/FavoriteButton.vue'
import type { FavoriteEntity } from '../services/types'
import { useSavedQueriesStore } from '../stores/savedQueries'
import { useUserPreferencesStore } from '../stores/userPreferences'
import { useFollowedShows } from '../composables/useFollowedShows'
import ShowRow from '../components/ShowRow.vue'
import FollowButton from '../components/FollowButton.vue'
import { useSignInGate } from '../composables/useSignInGate'
import { useInterestsStore } from '../stores/interests'
import { useSectionState } from '../composables/useSectionState'
import EpisodeCard from '../components/EpisodeCard.vue'
import DownloadedList from '../components/DownloadedList.vue'
import SectionStatus from '../components/SectionStatus.vue'
import ShowTile from '../components/ShowTile.vue'
import FollowedInterests from '../components/FollowedInterests.vue'
import HighlightsView from './HighlightsView.vue'
import ResurfacingInbox from './ResurfacingInbox.vue'
import CollectionsView from './CollectionsView.vue'
import SavedFilterBar from '../components/SavedFilterBar.vue'
import SavedColorControl from '../components/SavedColorControl.vue'
import ShowAllToggle from '../components/ShowAllToggle.vue'
import { useCappedSections } from '../composables/useCappedSections'
import { matchesQuery } from '../utils/textFilter'

const { t } = useI18n()
const favorites = useFavoritesStore()

/** Route to a saved entity's page, or null (storyline has no standalone page). */
function entityRoute(e: FavoriteEntity): { name: string; params: Record<string, string> } | null {
  if (e.kind === 'topic') return { name: 'topic', params: { id: e.ref } }
  if (e.kind === 'person') return { name: 'person', params: { id: e.ref } }
  if (e.kind === 'show') return { name: 'podcast', params: { feedId: e.ref } }
  return null
}
const capture = useCaptureStore()

/**
 * Every per-account store here already tracked `stale` — favourites, library, collections and
 * captures all set it when they fall back to their cached copy — and NONE of them said so. Home
 * got a notice; Library, which is the tab you actually open to find things you kept, showed a
 * cached list as though it were current.
 *
 * One notice for the tab, not one per section, for the same reason Home has one: four sections
 * each announcing their own staleness is the wall of repeated text this arc removed.
 */
const libraryStale = computed(
  () =>
    favorites.stale ||
    capture.stale ||
    useLibraryStore().stale ||
    useCollectionsStore().stale,
)
const libraryRetrying = ref(false)
async function retryLibrary(): Promise<void> {
  if (libraryRetrying.value) return
  libraryRetrying.value = true
  try {
    // `load`, not `ensureLoaded`: the point of the retry is to go back to the network for lists
    // that are already loaded — from cache — and would otherwise be considered done.
    await Promise.allSettled([
      favorites.load(),
      capture.load(),
      useLibraryStore().load(),
      useCollectionsStore().load(),
    ])
  } finally {
    libraryRetrying.value = false
  }
}

/**
 * Is there anything at all in Saved? (#1967 follow-up)
 *
 * Saved holds THREE things — favourited episodes, kept insights, and marked moments. Each section
 * was `v-if`'d on its own contents, so a new account saw exactly one heading ("Highlights") under
 * a tab of near-identical meaning, and it read as a redundant hierarchy. A design critic reviewing
 * an empty account concluded the two-level structure was a mistake; with any content it is
 * correct, and #1141 deliberately merged insights into Saved rather than giving them a tab.
 *
 * So the fix is the empty state, not the hierarchy: one honest empty state for the whole tab that
 * names all three things it holds, instead of one orphan heading standing for all of them.
 */
/**
 * Genuinely empty.
 *
 * The "is the library merely UNKNOWN" question is answered one place only — the `v-if` on
 * `capture.unavailable` that precedes this in the template. Repeating it here read as a second
 * guard while being unreachable behind the first, which is the kind of defensive-looking dead code
 * that makes the real guard hard to find. Offline, a user with highlights used to be shown
 * "Episodes you favourite, insights you keep, and moments you mark all live here": emptiness is a
 * claim about the ACCOUNT, and that one was a claim about the network.
 */
const savedIsEmpty = computed(() => !favorites.episodes.length && !capture.count)
const savedQueries = useSavedQueriesStore()
const userPrefs = useUserPreferencesStore()

/**
 * The Saved filter bar (RFC-121 ph. 3 / #2042), lifted to govern every section at once: which
 * kinds show (type chips, none = all), a colour filter, and a sort. The colour filter used to live
 * inside the Highlights list; it now covers episodes + entities too, since every saved item can
 * carry a colour (phase B). Sort is passed to Highlights (its per-episode grouping is the default);
 * for episodes/entities it only reorders the "by colour" view — they are otherwise newest-first.
 */
const savedTypes = ref<string[]>([])
const savedColor = ref<string | null>(null)
// One sort model across both tabs (#2042): 'recent' (default, as-stored newest-first) or 'title'
// (A–Z). Colour is a FILTER now, not a sort; per-episode grouping of highlights is structural.
const savedSort = ref<string>('recent')
// Type-to-filter search + per-section caps keep the hub scannable at 100+ items (#2042 follow-up).
// A non-empty query lifts every cap so a match is never hidden behind "Show all".
/** Muted filter (operator 2026-09-18) — highlights only; every other Saved section ignores it. */
const savedMutedOnly = ref(false)
const savedSearch = ref('')
const savedSearchActive = computed(() => savedSearch.value.trim() !== '')
// Saved's per-type sections page 10 at a time (operator 2026-09-18) rather than the default 6 with
// an all-or-nothing expand: at a hundred saved episodes "Show all" produces a scroll with no
// landmarks, so each press adds another ten.
const savedCaps = useCappedSections(10, 10)

/** A highlight matches the search on its own text (quote / speaker) — episode titles are findable
 *  through the Episodes section. Shared predicate so the count here and HighlightsView agree. */
function highlightMatches(h: { quote_text?: string | null; speaker?: string | null }): boolean {
  return matchesQuery(h.quote_text, savedSearch.value) || matchesQuery(h.speaker, savedSearch.value)
}

/** Apply the unified sort: A–Z by the given key, or 'recent' (leave the as-stored newest-first). */
function arrangeSaved<T>(items: T[], key: (i: T) => string): T[] {
  return savedSort.value === 'title'
    ? [...items].sort((a, b) => key(a).localeCompare(key(b)))
    : items
}

/** Type chips render only for kinds that actually have items (the #1962 presence rule). */
const availableTypes = computed<{ key: string; label: string }[]>(() => {
  const out: { key: string; label: string }[] = []
  if (savedQueries.count) out.push({ key: 'searches', label: t('library.savedTypeSearches') })
  if (favorites.episodes.length) out.push({ key: 'episodes', label: t('library.savedTypeEpisodes') })
  if (capture.count) out.push({ key: 'highlights', label: t('library.savedTypeHighlights') })
  const kinds = new Set(favorites.entities.map((e) => e.kind))
  if (kinds.has('show')) out.push({ key: 'shows', label: t('library.savedTypeShows') })
  if (kinds.has('topic')) out.push({ key: 'topics', label: t('library.savedTypeTopics') })
  if (kinds.has('person')) out.push({ key: 'people', label: t('library.savedTypePeople') })
  if (kinds.has('storyline'))
    out.push({ key: 'storylines', label: t('library.savedTypeStorylines') })
  return out
})

function typeVisible(key: string): boolean {
  return savedTypes.value.length === 0 || savedTypes.value.includes(key)
}
/** The entity kind → its type-chip key. */
const ENTITY_TYPE_KEY: Record<string, string> = {
  show: 'shows',
  topic: 'topics',
  person: 'people',
  storyline: 'storylines',
}

const filteredEpisodes = computed(() => {
  const eps = favorites.episodes.filter(
    (e) =>
      (!savedColor.value || e.color === savedColor.value) &&
      (matchesQuery(e.title, savedSearch.value) ||
        matchesQuery(e.podcast_title, savedSearch.value)),
  )
  return arrangeSaved(eps, (e) => e.title)
})
const filteredEntities = computed(() => {
  const ents = favorites.entities.filter(
    (e) =>
      (!savedColor.value || e.color === savedColor.value) &&
      typeVisible(ENTITY_TYPE_KEY[e.kind] ?? 'entities') &&
      matchesQuery(e.label, savedSearch.value),
  )
  return arrangeSaved(ents, (e) => e.label)
})
// Saved searches match on the query text; a colour filter hides them (searches carry no colour).
const filteredSearches = computed(() =>
  savedColor.value ? [] : savedQueries.list.filter((q) => matchesQuery(q.q, savedSearch.value)),
)

// Cap each section to the top N (lifted while searching), with a "Show all" expand in place.
const visibleEpisodes = computed(() =>
  savedCaps.visible('episodes', filteredEpisodes.value, savedSearchActive.value),
)
/**
 * Saved entities, split ONE SECTION PER KIND (operator 2026-09-17).
 *
 * They used to share a single "Shows, topics & people" list where every row opened with a kind
 * kicker — a mix that reads as one heap of unlike things, and made a saved SHOW a 48px line of text
 * on the one hand and a 176px artwork tile over in Following on the other.
 *
 * Split, each kind can be presented as itself: shows get artwork like every other show surface,
 * while topics / storylines / people stay as text rows, which is all they have.
 */
const savedByKind = (kind: string) => computed(() => filteredEntities.value.filter((e) => e.kind === kind))
const savedShowEntities = savedByKind('show')
const savedTopicEntities = savedByKind('topic')
const savedStorylineEntities = savedByKind('storyline')
const savedPersonEntities = savedByKind('person')

/**
 * Saved shows resolved to full catalogue records, so the section can render the SAME row Discover's
 * Shows list uses — 44px artwork, title, episode count — rather than a bare label. A show that has
 * left the corpus still renders from its saved label instead of vanishing, the same fallback
 * `useFollowedShows` makes.
 */
/**
 * The non-show kinds, in the order they render: topics, then storylines, then PEOPLE LAST
 * (operator 2026-09-17). Shows are not here — they render above Episodes with artwork.
 */
const savedEntityGroups = computed(() => [
  { kind: 'topic', labelKey: 'library.savedTypeTopics', items: savedTopicEntities.value },
  { kind: 'storyline', labelKey: 'library.savedTypeStorylines', items: savedStorylineEntities.value },
  { kind: 'person', labelKey: 'library.savedTypePeople', items: savedPersonEntities.value },
])

const savedShowPodcasts = computed(() => {
  const byId = new Map(catalogue.value.map((p) => [p.feed_id, p]))
  return savedShowEntities.value.map((e) => ({
    entity: e,
    show: byId.get(e.ref) ?? {
      feed_id: e.ref,
      title: e.label,
      artwork_url: null,
      image_url: null,
      description: null,
      episode_count: 0,
    },
  }))
})

/**
 * Colour-, muted- and search-filtered highlight count, so the Highlights section hides when empty.
 *
 * Must apply EVERY filter `HighlightsView` applies. Adding the muted filter to the list but not
 * here put "Highlights 6" directly above a list of 2 — the same class of disagreement the count
 * was made filter-aware to fix in the first place.
 */
const visibleHighlightCount = computed(
  () =>
    capture.highlights.filter(
      (h) =>
        (!savedColor.value || h.color === savedColor.value) &&
        (!savedMutedOnly.value || h.retired) &&
        highlightMatches(h),
    ).length,
)

/** Anything to show at all under the current filters? Drives the "no match" note. */
const nothingMatchesFilter = computed(
  () =>
    !savedIsEmpty.value &&
    (savedTypes.value.length > 0 || savedColor.value !== null || savedSearchActive.value) &&
    !(typeVisible('searches') && filteredSearches.value.length) &&
    !(typeVisible('episodes') && filteredEpisodes.value.length) &&
    !(typeVisible('highlights') && visibleHighlightCount.value) &&
    !filteredEntities.value.length,
)

// Tabs: Shows (the feeds you follow) · Saved · Revisit · Queue · Recent — five fit a phone row with
// no scroll. Highlights + Collections are now SECTIONS inside Saved (everything you deliberately kept
// under one tab), which is what kept the strip short. Shows is the follow-management home.
// Queue + Recent moved to the player surface (#1838) — reachable from the mini/full player's queue
// button, not Library — which frees the tab strip (and the slot the Collections tab will take).
type Tab = 'shows' | 'saved' | 'collections' | 'revisit'
/**
 * The tab strip is {@link ../components/Tabs.vue} now (#1594 item 7) — this view had `role`-less
 * buttons, so the strip did not announce as tabs at all and the arrow keys did nothing.
 *
 * Labels are translated HERE rather than inside `Tabs`: a shared control that reaches into i18n
 * would force every caller onto one key namespace, and two of the seven call sites pass strings
 * that are not i18n keys at all.
 */
const TAB_KEYS: { key: Tab; labelKey: string }[] = [
  { key: 'shows', labelKey: 'library.following' },
  { key: 'saved', labelKey: 'library.saved' },
  { key: 'collections', labelKey: 'library.collections' },
  { key: 'revisit', labelKey: 'library.revisit' },
]
const tabs = computed<TabSpec<Tab>[]>(() =>
  TAB_KEYS.map((tb) => ({ key: tb.key, label: t(tb.labelKey), testid: `library-tab-${tb.key}` })),
)
// Home's "See all N shows →" deep-links here with ?tab=shows so it lands on the follows, not Saved.
const route = useRoute()
const initialTab = String(route.query.tab || '')
const tab = ref<Tab>(TAB_KEYS.some((tb) => tb.key === initialTab) ? (initialTab as Tab) : 'saved')

// Followed shows — the same derivation Home's "Your shows" rail uses (shared so they can't drift).
// Section-state so a catalogue/library outage renders error+retry, never a fake "you follow nothing".
const {
  shows: followedShows,
  suggested: suggestedShows,
  load: loadFollows,
  catalogue,
} = useFollowedShows()

// Unfollowing from the Following list. `ShowTile` owned this internally; `ShowRow` takes its
// controls from the host, so the host owns the toggle. `gated()` wraps a ZERO-argument action, so
// the row is closed over per call rather than passed through it.
const library = useLibraryStore()
const { isGated, gated } = useSignInGate()
const busyFollow = ref<string | null>(null)
function toggleFollow(p: { feed_id: string; title?: string | null }): void {
  gated(async () => {
    busyFollow.value = p.feed_id
    try {
      await library.toggle(p.feed_id, { title: p.title ?? p.feed_id })
    } finally {
      busyFollow.value = null
    }
  })()
}

/**
 * The Following tab's own filter bar (#2042 follow-up) — same shape as Saved's, minus colour:
 * search + type chips (shows / topics / people / storylines present) + sort (recent / A–Z). Each
 * section caps to the top N with "Show all"; a search lifts the caps. FollowedInterests applies the
 * same search / sort / type-visibility to its three sections.
 */
const followingInterests = useInterestsStore()
const followingSearch = ref('')
const followingSort = ref('recent')
const followingTypes = ref<string[]>([])
const followingSearchActive = computed(() => followingSearch.value.trim() !== '')
const followingCaps = useCappedSections()

const followingAvailableTypes = computed<{ key: string; label: string }[]>(() => {
  const out: { key: string; label: string }[] = []
  if (followedShows.value.length) out.push({ key: 'shows', label: t('library.savedTypeShows') })
  const ids = followingInterests.ids
  if (ids.some((i) => i.startsWith('topic:')))
    out.push({ key: 'topics', label: t('library.savedTypeTopics') })
  if (ids.some((i) => i.startsWith('person:')))
    out.push({ key: 'people', label: t('library.savedTypePeople') })
  if (ids.some((i) => i.startsWith('thc:') || i.startsWith('tc:')))
    out.push({ key: 'storylines', label: t('library.savedTypeStorylines') })
  return out
})
function followingTypeVisible(key: string): boolean {
  return followingTypes.value.length === 0 || followingTypes.value.includes(key)
}
/** The interest-kind subset of the active type filter, handed to FollowedInterests. */
const interestVisibleTypes = computed(() =>
  followingTypes.value.filter((k) => k !== 'shows'),
)
/**
 * Whether the interests block renders at all.
 *
 * `FollowedInterests` reads an EMPTY `visibleTypes` as "no filter — show every kind", which is right
 * when nothing is selected and wrong when the selection is Shows-only: filtering the shows key out
 * of `['shows']` also produces `[]`, so picking Shows revealed topics, people and storylines
 * instead of hiding them (operator 2026-09-17). One sentinel, two meanings.
 *
 * Deciding it HERE keeps that ambiguity out of the child: this view knows whether a filter is
 * active, the child only knows the list it was handed.
 */
const showFollowedInterests = computed(
  () => followingTypes.value.length === 0 || interestVisibleTypes.value.length > 0,
)
const filteredShows = computed(() => {
  const shows = followedShows.value.filter((s) => matchesQuery(s.title, followingSearch.value))
  return followingSort.value === 'title'
    ? [...shows].sort((a, b) => (a.title ?? '').localeCompare(b.title ?? ''))
    : shows
})
const visibleShows = computed(() =>
  followingCaps.visible('shows', filteredShows.value, followingSearchActive.value),
)

const showsSection = useSectionState<null>(null)
function loadFollowedShows(): Promise<void> {
  return showsSection.load(async () => {
    await loadFollows()
    return null
  })
}

// Library is in KEEP_ALIVE_TABS, so `onMounted` fires once per session — a badge refreshed only
// there goes stale the moment you review anything (#2004 item 14 follow-up).
onActivated(() => {
  void useResurfacingStore().load()
  // Follows load in `onMounted`, which fires ONCE for a kept-alive tab. So a failed library fetch —
  // or an account switch, which resets the store without reloading it — left this tab showing
  // "you're not following any shows yet" plus six suggestions for the rest of the session, with no
  // way to recover short of a full reload. Retry only when there is nothing good to show, so a
  // healthy tab does not refetch on every visit.
  if (showsSection.phase.value === 'error' || !useLibraryStore().loaded) {
    void loadFollowedShows()
  }
})

onMounted(async () => {
  // The Revisit tab is one tap away, so the nav badge must not disagree with what the user is
  // about to see. Fire-and-forget: the badge is ambient, and nothing on this page waits on it.
  void useResurfacingStore().load()
  await favorites.ensureLoaded()
  // The Highlights section and the tab's empty state both gate on `capture.count`, so this tab has
  // to hydrate the store itself rather than trust App.vue's sign-in load to have finished. Without
  // it the gate is a chicken-and-egg: an unloaded store reads as zero captures, the section never
  // mounts, and the view that would have loaded them never runs.
  void capture.ensureLoaded().catch(() => {})
  // #1261-8: fire-and-forget the USERPREFS-1 hydrate so the saved-queries store picks up the
  // cross-device list (the preferences endpoint being offline shouldn't gate the tab).
  void userPrefs.hydrate()
  void loadFollowedShows()
})
</script>

<template>
  <!-- Same container as Discover — `mx-auto max-w-3xl px-4 pb-8` (operator 2026-09-17). Library was
       a bare <section> inheriting the app shell's wider `max-w-6xl px-5`, so the SAME tile markup
       rendered at 270px here and 176px on Browse, and 118px against 108px on a phone. The column
       count was never the cause; the container was. Discover sets the standard, so Library adopts
       it and the grids agree by construction instead of by compensating arithmetic. -->
  <section class="mx-auto max-w-3xl px-4 pb-8">
    <h1 class="mb-4 font-display text-3xl font-extrabold tracking-tight">{{ t('library.title') }}</h1>

    <!-- Standalone, never chained into a neighbouring v-if. -->
    <StaleNotice v-if="libraryStale" :busy="libraryRetrying" @retry="retryLibrary" />

    <!-- `equal-width` keeps all four on ONE phone row (they used to wrap at px-3/text-sm). -->
    <Tabs
      v-model="tab"
      :tabs="tabs"
      :label="t('nav.library')"
      id-prefix="library"
      equal-width
      class="mb-6"
    />

    <!-- Following — everything you follow: shows (feeds) plus the topics / people / storylines you
         followed via +. The follow-management home: Home's "See all N shows →" deep-links here
         (?tab=shows). Sectioned by kind, like Saved.

         `v-show`, so the panel keeps its scroll position and expanded rows across tab switches.

         This was briefly `v-if` to fix a "Read more" that appeared on one-line show descriptions.
         That was the wrong diagnosis: mounting under `display: none` is harmless on its own, because
         ResizeObserver fires across `display: none` → visible. The real fault was in ShowRow, which
         attached its observer in `onMounted` and so attached NOTHING when its two-phase `show` prop
         had no description on the first render. `v-if` only masked it by delaying mount until both
         stores had loaded — and it did nothing for the identical rows in the Saved panel. Fixed in
         the component; this stays `v-show`. -->
    <div v-show="tab === 'shows'" v-bind="panelAttrs('library', 'shows')">
      <!-- Following's own filter bar (search + type + sort), same shape as Saved's minus colour. -->
      <SavedFilterBar
        v-if="tab === 'shows' && followingAvailableTypes.length"
        v-model:types="followingTypes"
        v-model:sort="followingSort"
        v-model:search="followingSearch"
        :available-types="followingAvailableTypes"
        :show-colors="false"
        :search-placeholder="t('library.searchFollowing')"
      />
      <section v-if="followingTypeVisible('shows')" class="mb-6">
        <!-- Same heading as every other Library section (operator 2026-09-18): `lp-section` with
             the count as a muted kicker beside it. Following used a small uppercase kicker while
             Saved used this, so two tabs of the same hub labelled their sections two ways. -->
        <h2 class="lp-section mb-2">
          {{ t('library.followingShows') }}
          <span v-if="filteredShows.length" class="lp-kicker ml-1 font-normal">{{
            filteredShows.length
          }}</span>
        </h2>
        <SectionStatus :phase="showsSection.phase.value" :rows="2" @retry="loadFollowedShows" />
        <div
          v-if="showsSection.isReady.value && !followedShows.length"
          class="rounded-xl border border-dashed border-border p-4"
        >
          <p class="text-sm text-muted">{{ t('library.showsEmpty') }}</p>
          <!--
            The grid below is SUGGESTIONS, and it used to sit under the "Shows" kicker in tiles
            identical to the followed ones — so the only thing saying "these are not yours" was a
            dashed border. Reading it as "here are your shows, and they all say Follow" is the
            obvious misreading, and it is the one that got reported.
          -->
          <h4 v-if="suggestedShows.length" class="lp-kicker mt-3">{{ t('library.showsSuggested') }}</h4>
          <ul v-if="suggestedShows.length" class="mt-2 grid grid-cols-3 gap-3 sm:grid-cols-6" data-testid="library-shows-suggested">
            <li v-for="p in suggestedShows" :key="p.feed_id"><ShowTile :show="p" followable /></li>
          </ul>
          <RouterLink
            :to="{ name: 'catalog' }"
            class="mt-3 inline-block text-xs font-bold text-accent no-underline"
          >{{ t('library.showsBrowse') }}</RouterLink>
        </div>
        <template v-else-if="followedShows.length">
          <!-- The shared ShowRow, not a tile grid (operator 2026-09-17) — the third surface to take
               it, after Discover → Shows and Library → Saved. Following is a column of lists
               (shows, topics, storylines, people), so a block of tiles in the middle of them broke
               that column; the row keeps the section reading as one more list and shows the
               description, which a tile has no room for. -->
          <ul class="flex flex-col" data-testid="library-shows-list">
            <li v-for="p in visibleShows" :key="p.feed_id">
              <ShowRow :show="p">
                <!-- Bare controls: ShowRow stacks and plates them over the artwork. -->
                <template #actions>
                  <FollowButton
                    variant="overlay"
                    :following="library.has(p.feed_id)"
                    :busy="busyFollow === p.feed_id"
                    :gated="isGated"
                    @toggle="toggleFollow(p)"
                  />
                  <FavoriteButton
                    :item="{ kind: 'show', ref: p.feed_id, label: p.title ?? p.feed_id }"
                  />
                </template>
              </ShowRow>
            </li>
          </ul>
          <ShowAllToggle
            v-if="followingCaps.overflows(filteredShows.length, followingSearchActive)"
            :expanded="followingCaps.expanded.has('shows')"
            :count="filteredShows.length"
            @toggle="followingCaps.toggle('shows')"
          />
        </template>
      </section>

      <!-- Topics / People / Storylines you follow (previously invisible — the interests profile). -->
      <FollowedInterests
        v-if="showFollowedInterests"
        :search="followingSearch"
        :sort="followingSort"
        :visible-types="interestVisibleTypes"
      />
    </div>

    <!-- Saved — everything you deliberately kept, one section per kind: searches, episodes, insights,
         plus Highlights and Collections (folded in from their old tabs to keep the strip to five).
         Each section owns its own presence/empty state, so there is no separate "nothing saved" line. -->
    <div v-show="tab === 'saved'" v-bind="panelAttrs('library', 'saved')">
        <!-- The filter bar governs every section below (type · colour · sort), RFC-121 ph. 3. Only
             shown once there is something to filter. -->
        <!-- Gated on the ACTIVE tab (not just v-show) so only ONE SavedFilterBar is ever mounted —
             the two tabs share its static testids, and the bar is presentational (state lives in
             LibraryView refs), so unmounting the inactive one loses nothing (#2042). -->
        <SavedFilterBar
          v-if="tab === 'saved' && !savedIsEmpty && !capture.unavailable"
          v-model:types="savedTypes"
          v-model:color="savedColor"
          v-model:sort="savedSort"
          v-model:search="savedSearch"
          v-model:muted-only="savedMutedOnly"
          :available-types="availableTypes"
          show-muted
        />
        <!-- #1261-8: Saved searches — power-listener persistent queries.
             Tap the query to re-run the search; ×  removes it. Searches carry no colour, so a
             colour filter hides them. -->
        <section
          v-if="typeVisible('searches') && filteredSearches.length"
          class="mb-6"
          data-testid="saved-searches-section"
        >
          <h2 class="lp-section mb-2">{{ t('library.savedSearches') }}</h2>
          <ul class="flex flex-col">
            <li
              v-for="q in filteredSearches"
              :key="q.scope + '|' + q.q"
              class="flex items-center gap-2 border-b border-border py-2"
            >
              <RouterLink
                :to="{ name: 'search', query: { q: q.q, scope: q.scope } }"
                class="min-w-0 flex-1 text-sm font-semibold leading-snug text-canvas-foreground no-underline"
              >
                {{ q.q }}
                <span class="lp-kicker ml-1 font-normal">
                  {{ q.scope === 'mine' ? t('search.scopeMine') : t('search.scopeAll') }}
                </span>
              </RouterLink>
              <button
                type="button"
                class="shrink-0 rounded-full border border-border px-2 py-1 text-xs font-semibold text-muted transition hover:text-canvas-foreground"
                :aria-label="t('library.savedSearchRemove', { q: q.q })"
                @click="savedQueries.remove(q.q, q.scope)"
              >
                ×
              </button>
            </li>
          </ul>
        </section>

        <!-- SHOWS lead the saved content, before episodes (operator 2026-09-17), as a list row with
             44px artwork — the shape Discover's Shows LIST uses.

             A tile grid was tried and rejected here: Saved is a column of lists (searches, episodes,
             topics, storylines, people), and a block of artwork tiles in the middle of them breaks
             that column. The row keeps the section reading as one more list while still showing the
             cover, which is how a show is actually recognised. -->
        <section v-if="savedShowPodcasts.length" class="mb-6">
          <h2 class="lp-section mb-2">
            {{ t('library.savedTypeShows') }}
            <span class="lp-kicker ml-1 font-normal">{{ savedShowPodcasts.length }}</span>
          </h2>
          <ul class="flex flex-col" data-testid="saved-shows-list">
            <li
              v-for="{ entity, show } in savedShowPodcasts"
              :key="show.feed_id"
              data-testid="saved-entity"
            >
              <ShowRow :show="show">
                <!-- Bare controls: ShowRow stacks and plates them over the artwork. -->
                <template #actions>
                  <SavedColorControl
                    :color="entity.color"
                    @pick="favorites.setColor('show', entity.ref, $event)"
                  />
                  <FavoriteButton :item="{ kind: 'show', ref: entity.ref, label: entity.label }" />
                </template>
              </ShowRow>
            </li>
          </ul>
        </section>

        <!-- Episodes — each carries the shared colour control (phase B) in the card's action row.
             Capped to the top N with "Show all" (#2042 follow-up); search lifts the cap. -->
        <section v-if="typeVisible('episodes') && filteredEpisodes.length" class="mb-6">
          <h2 class="lp-section mb-2">
            {{ t('library.savedEpisodes') }}
            <span class="lp-kicker ml-1 font-normal">{{ filteredEpisodes.length }}</span>
          </h2>
          <div class="flex flex-col">
            <!-- `hide-favorite`: every row here IS saved, so the heart restates what the surface
                 already says — while spending one of three slots in the 128px column, which wrapped
                 the colour control onto a second row (operator 2026-09-16). It moves into the ⋯
                 (still the only way to unsave) and the colour control takes its place via
                 `lead-action`, so the row is three wide again and nothing wraps. -->
            <EpisodeCard v-for="e in visibleEpisodes" :key="e.slug" :episode="e" hide-favorite>
              <template #lead-action>
                <SavedColorControl
                  :color="e.color"
                  @pick="favorites.setColor('episode', e.slug, $event)"
                />
              </template>
            </EpisodeCard>
          </div>
          <ShowAllToggle
            v-if="savedCaps.overflows(filteredEpisodes.length, savedSearchActive, 'episodes')"
            :expanded="savedCaps.remaining('episodes', filteredEpisodes.length) === 0"
            :count="filteredEpisodes.length"
            :remaining="savedCaps.remaining('episodes', filteredEpisodes.length)"
            @toggle="savedCaps.toggle('episodes', filteredEpisodes.length)"
          />
        </section>
        <!-- Topics, then storylines, then people — each its own section (operator 2026-09-17).
             These three carry no imagery, so they stay text rows; only the heading changes, which is
             the point: the kind is named once at the top instead of on every row. -->
        <!-- `<template v-for>` with the v-if INSIDE: in Vue 3 `v-if` wins on the same element and
             would be evaluated before `grp` exists. -->
        <template v-for="grp in savedEntityGroups" :key="grp.kind">
          <section v-if="grp.items.length" class="mb-6">
            <h2 class="lp-section mb-2">
              {{ t(grp.labelKey) }}
              <span class="lp-kicker ml-1 font-normal">{{ grp.items.length }}</span>
            </h2>
            <ul class="flex flex-col">
              <li
                v-for="e in grp.items"
                :key="e.kind + ':' + e.ref"
                class="flex items-center gap-2 border-b border-border py-2"
                data-testid="saved-entity"
              >
                <RouterLink
                  v-if="entityRoute(e)"
                  :to="entityRoute(e)!"
                  class="min-w-0 flex-1 truncate text-sm font-semibold text-canvas-foreground no-underline"
                >{{ e.label }}</RouterLink>
                <span v-else class="min-w-0 flex-1 truncate text-sm font-semibold">{{ e.label }}</span>
                <SavedColorControl :color="e.color" @pick="favorites.setColor(e.kind, e.ref, $event)" />
                <FavoriteButton :item="{ kind: e.kind, ref: e.ref, label: e.label }" />
              </li>
            </ul>
          </section>
        </template>
        <!-- Insights are NOT favorites — they save via the highlights path and render in the
             Highlights section below (RFC-121 / #1593). -->
        <!-- Highlights — captured moments / spans / saved insights, grouped by episode, with notes.
             Folded in from its old tab (#1141). Conditional like its two siblings now: when it was
             the only unconditional section, an empty account saw one orphan heading standing for a
             tab that actually holds three things. -->
        <section v-if="typeVisible('highlights') && visibleHighlightCount" class="mb-6">
          <h2 class="lp-section mb-2">
            {{ t('library.highlights') }}
            <span class="lp-kicker ml-1 font-normal">{{ visibleHighlightCount }}</span>
          </h2>
          <HighlightsView
            :filter-color="savedColor"
            :sort="savedSort"
            :search="savedSearch"
            :muted-only="savedMutedOnly"
          />
        </section>

        <!-- Downloaded (#1905) — device-local, native only, renders with no API calls.
             LAST of the Saved sections (operator 2026-09-18). It sat second, above everything the
             user actually saved, so a native build opened Saved on a device-storage list rather
             than on their own captures. It is a utility view of what happens to be on this phone,
             not a thing they curated — so it reads last. -->
        <DownloadedList />

        <!-- Filters can empty every section while the account is NOT empty — say so, rather than
             show a blank tab that reads as a bug. -->
        <p
          v-if="nothingMatchesFilter"
          class="text-muted"
          data-testid="saved-no-match"
        >{{ t('library.savedEmptyFiltered') }}</p>

        <!-- ONE empty state for the whole tab, naming all three things it holds. A new account now
             learns what Saved is FOR, instead of meeting a lone "Highlights" heading and inferring
             the tab is redundant. The ghost card shows the shape of what will live here; the action
             is the only thing a person can actually do about being empty. -->
        <p
          v-if="capture.unavailable"
          class="text-muted"
          data-testid="saved-unavailable"
        >{{ t('library.savedUnavailable') }}</p>

        <div v-else-if="savedIsEmpty">
          <p class="text-muted">{{ t('library.savedEmpty') }}</p>
          <div class="mt-4 rounded-2xl border border-border p-4 opacity-40" aria-hidden="true">
            <span class="lp-kicker block">{{ t('library.highlights') }}</span>
            <span class="mt-2 block h-3 w-3/4 rounded bg-overlay"></span>
            <span class="mt-2 block h-3 w-1/2 rounded bg-overlay"></span>
          </div>
          <RouterLink
            :to="{ name: 'catalog' }"
            class="mt-4 inline-block text-sm font-bold text-accent no-underline"
          >
            {{ t('highlights.emptyCta') }}
          </RouterLink>
        </div>
    </div>

    <!-- Collections — the Pinterest-style curation boards, now a first-class tab (RFC-119). -->
    <div v-show="tab === 'collections'" v-bind="panelAttrs('library', 'collections')">
      <CollectionsView />
    </div>

    <!-- Revisit — spaced resurfacing of past highlights with reflection prompts. -->
    <div v-show="tab === 'revisit'" v-bind="panelAttrs('library', 'revisit')">
      <ResurfacingInbox />
    </div>
  </section>
</template>
