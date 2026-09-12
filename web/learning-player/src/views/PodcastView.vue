<script setup lang="ts">
/**
 * Per-podcast catalog view (PRD-038 FR2): one show's episodes, newest-first, paginated.
 * Header derives the show title + total from the first page (no separate feed endpoint in
 * the MVP). Cards reuse EpisodeCard.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import AddToCollectionButton from '../components/AddToCollectionButton.vue'
import FavoriteButton from '../components/FavoriteButton.vue'
import EntityCard from '../components/EntityCard.vue'
import EpisodeCard from '../components/EpisodeCard.vue'
import PodcastSignalsBand from '../components/PodcastSignalsBand.vue'
import ShowActivityChart from '../components/ShowActivityChart.vue'
import NoteComposer from '../components/NoteComposer.vue'
import SectionStatus from '../components/SectionStatus.vue'
import FollowButton from '../components/FollowButton.vue'
import ShareMenu from '../components/ShareMenu.vue'
import { accentForKind, type EntityCardModel } from '../composables/entityShareCard'
import { formatPublishDate } from '../utils/format'
import { getPodcasts, listPodcastEpisodes } from '../services/api'
import { useAuthStore } from '../stores/auth'
import { useLibraryStore } from '../stores/library'
import { useCompletedStore } from '../stores/completed'
import { useFavoritesStore } from '../stores/favorites'
import { useSignInGate } from '../composables/useSignInGate'
import { showArtwork } from '../utils/episode'
import { formatDuration } from '../utils/format'
import type { EpisodeSummary, Podcast } from '../services/types'

const PAGE_SIZE = 20
const props = defineProps<{ feedId: string }>()
const { t, locale } = useI18n()
const router = useRouter()

// Back = return to wherever you came from (Home, an entity card, the player kicker, Browse…), not a
// hardcoded destination the user may never have visited. Mirrors the player's back (falls back to
// Browse on a cold deep-link with no in-app history).
function goBack(): void {
  if (window.history.length > 1) router.back()
  else void router.push({ name: 'catalog' })
}

const episodes = ref<EpisodeSummary[]>([])
const total = ref(0)
const page = ref(0)
const hasMore = ref(false)
const loading = ref(false)
const error = ref(false)
const show = ref<Podcast | null>(null)
const descExpanded = ref(false)
// Hide-played toggle (SD.9) — reads the completed set from PL.6.
const completed = useCompletedStore()
const favorites = useFavoritesStore()
const hidePlayed = ref(false)
const visibleEpisodes = computed(() =>
  hidePlayed.value ? episodes.value.filter((e) => !completed.has(e.slug)) : episodes.value,
)

// Update cadence (SD.6) — the median gap between the loaded (recent) episodes' publish dates,
// bucketed into a one-word rhythm. Needs ≥3 dated episodes to be meaningful.
const cadence = computed<string | null>(() => {
  const days = episodes.value
    .map((e) => (e.publish_date ? Date.parse(e.publish_date) : NaN))
    .filter((n) => !Number.isNaN(n))
    .sort((a, b) => b - a)
  if (days.length < 3) return null
  const gaps: number[] = []
  for (let i = 0; i < days.length - 1; i++) gaps.push((days[i] - days[i + 1]) / 86_400_000)
  gaps.sort((a, b) => a - b)
  const median = gaps[Math.floor(gaps.length / 2)]
  if (median <= 0) return null
  if (median < 2) return 'daily'
  if (median < 10) return 'weekly'
  if (median < 20) return 'biweekly'
  if (median < 45) return 'monthly'
  return 'irregular'
})
// Typical episode length (SD.7) — the MEDIAN duration of the loaded episodes, so one 3-hour special
// doesn't skew a show of 20-minute episodes. Approximate: it's over the loaded pages, not the whole
// feed, so it reads "~48 min". Needs ≥3 dated durations to be worth showing.
const typicalLength = computed<string | null>(() => {
  const secs = episodes.value
    .map((e) => e.duration_seconds ?? 0)
    .filter((s) => s > 0)
    .sort((a, b) => a - b)
  if (secs.length < 3) return null
  return formatDuration(secs[Math.floor(secs.length / 2)])
})
const cardTarget = ref<{ kind: 'person' | 'topic'; id: string } | null>(null)

// #2036 — the shareable card for this show: title + episode count + a canonical link. Clean (no
// quote/byline) — the feed description is marketing copy, not a signature take. Brand-cyan accent
// (shows own no theme token).
const shareModel = computed<EntityCardModel>(() => ({
  kicker: t('share.kickerShow'),
  title: showTitle.value || props.feedId,
  stats: total.value ? `${total.value} ${total.value === 1 ? 'episode' : 'episodes'}` : null,
  accent: accentForKind('show'),
  url:
    typeof window !== 'undefined' ? `${window.location.origin}/podcast/${props.feedId}` : null,
}))

const showArt = showArtwork
/**
 * Has the show lookup finished? Drives the title fallback.
 *
 * The heading used to fall straight through to the raw `feedId`, so the page painted "p05" as its
 * title for as long as the lookup took — an internal identifier presented to a listener as the name
 * of the show. It is only an acceptable last resort once we KNOW no name is coming.
 */
const showResolved = ref(false)
async function loadShow(): Promise<void> {
  try {
    const all = await getPodcasts().catch(() => [] as Podcast[])
    show.value = all.find((p) => p.feed_id === props.feedId) ?? null
  } finally {
    showResolved.value = true
  }
}

/** The show's name, or null while we are still finding out (never the raw feed id mid-flight). */
const showTitle = computed(
  () => show.value?.title ?? episodes.value[0]?.podcast_title ?? (showResolved.value ? props.feedId : null),
)

/** Feed last-updated, formatted for display (#2043); null when the channel carried none. */
const feedUpdated = computed(() => formatPublishDate(show.value?.last_updated ?? null, locale.value))

// Follow this show → a feed subscription (/api/app/library), which is what fills the "new in your
// follows" section of Your Week. Distinct from the interest tokens followed on entity cards.
const auth = useAuthStore()
const library = useLibraryStore()
const { isGated, gated } = useSignInGate()
// Auth may resolve after this mounts, so load follow-state on the transition, not just onMounted.
watch(
  () => auth.isAuthenticated,
  (authed) => {
    if (authed) void library.ensureLoaded().catch(() => {})
  },
  { immediate: true },
)

const following = computed(() => library.has(props.feedId))
const togglingFollow = ref(false)
/** Signed-out follows route to sign-in rather than firing a 401 the store silently reverts (#1590). */
const toggleFollow = gated(async () => {
  togglingFollow.value = true
  try {
    await library.toggle(props.feedId, { title: show.value?.title ?? episodes.value[0]?.podcast_title })
  } finally {
    togglingFollow.value = false
  }
})

async function loadMore(): Promise<void> {
  loading.value = true
  error.value = false
  try {
    const next = page.value + 1
    const res = await listPodcastEpisodes(props.feedId, { page: next, pageSize: PAGE_SIZE })
    episodes.value.push(...res.items)
    page.value = next
    total.value = res.total
    hasMore.value = res.has_more
  } catch {
    error.value = true
  } finally {
    loading.value = false
  }
}

function reset(): void {
  episodes.value = []
  page.value = 0
  total.value = 0
  hasMore.value = false
  show.value = null
  descExpanded.value = false
  void loadShow()
  void loadMore()
}

onMounted(() => {
  void loadShow()
  void loadMore()
  if (auth.isAuthenticated) {
    void completed.ensureLoaded().catch(() => {})
    void favorites.ensureLoaded().catch(() => {})
  }
})
watch(() => props.feedId, reset)
</script>

<template>
  <section>
    <button type="button" class="lp-nav" @click="goBack">‹ {{ t('nav.back') }}</button>

    <header class="mb-6 mt-2 flex gap-4 sm:gap-5">
      <!--
        LEFT COLUMN: artwork, then the show's actions (#2004 item 5).

        Structurally the same problem as the browse row: the artwork sat alone at 80px (112 at `sm`)
        while the text column carried the title, the episode count, the description, the expand
        toggle AND Follow + collection. The space beside and below the artwork was dead.

        Bigger artwork is a PREREQUISITE here, not an independent tweak — a "+ Follow show" pill does
        not fit under an 80px column. At 144px it does.
      -->
      <!-- Capped to the artwork width (w-36): without it the action row below sets the column's
           width, so a wide row of pills pushed the column past 144px and squeezed the text column
           to a third of the row (title wrapping to 3 lines). The actions wrap WITHIN 144px instead. -->
      <div class="flex w-36 shrink-0 flex-col gap-3">
      <!-- Placeholder so the column keeps its width when a show has no artwork — otherwise the
           actions beneath it are squeezed against a zero-width gap (same bug as EpisodeCard). -->
      <div
        v-if="!(show && showArt(show))"
        class="h-36 w-36 rounded-xl bg-elevated"
        aria-hidden="true"
      />
      <img
        v-if="show && showArt(show)"
        :src="showArt(show)!"
        :alt="show.title ?? ''"
        class="h-36 w-36 rounded-xl bg-elevated object-cover"
      />
        <!-- Two aligned rows under the 144px artwork: the primary Follow pill full-width on top,
             the secondary actions as an even icon row beneath (two pills can't share a 144px row,
             so Collection uses its compact icon variant here rather than the wide pill). -->
        <div class="flex flex-col gap-2">
          <!-- The shared show-follow pill (F2.4), inline variant — full width of the column. -->
          <FollowButton
            :following="following"
            :busy="togglingFollow"
            :gated="isGated"
            class="w-full justify-center"
            @toggle="toggleFollow"
          />
          <div class="flex items-center justify-between">
            <!-- Save the show (heart) — the ONE save affordance, distinct from Follow (SD.1 / F2.2). -->
            <FavoriteButton :item="{ kind: 'show', ref: feedId, label: show?.title ?? feedId }" />
            <!-- Pin this show into a collection (RFC-119). -->
            <AddToCollectionButton :item="{ kind: 'show', ref: feedId }" />
            <!-- Share (card / link / text) — #2036. -->
            <ShareMenu :model="shareModel" />
          </div>
        </div>
      </div>
      <div class="min-w-0 flex-1">
        <h1 class="font-display text-2xl font-extrabold leading-tight tracking-tight sm:text-3xl">
          <template v-if="showTitle">{{ showTitle }}</template>
          <!-- Placeholder, not the feed id: same height as the real heading so nothing jumps when
               the name lands. `aria-hidden` keeps a screen reader from announcing a shimmer bar. -->
          <span
            v-else
            class="block h-7 w-2/3 animate-pulse rounded bg-elevated sm:h-9"
            aria-hidden="true"
            data-testid="podcast-title-skeleton"
          />
        </h1>
        <!-- Feed by-line (#2043): host/author names straight from the RSS channel. Text for now —
             linking each to its person card is the entity-resolution follow-up (#2044). -->
        <p v-if="show?.authors?.length" class="mt-1 text-sm text-muted" data-testid="podcast-byline">
          {{ t('podcast.byline', { authors: show.authors.join(', ') }) }}
        </p>
        <p v-if="total" class="mt-1 text-sm text-muted">
          {{ t('podcast.episodeCount', { count: total }, total)
          }}<template v-if="cadence"> · {{ t(`podcast.cadence.${cadence}`) }}</template
          ><template v-if="typicalLength"> · {{ t('podcast.typicalLength', { len: typicalLength }) }}</template>
        </p>
        <!-- Feed language + last-updated (#2043), when the channel carried them. -->
        <p
          v-if="show?.language || feedUpdated"
          class="mt-1 flex flex-wrap items-center gap-x-2 text-xs text-muted"
          data-testid="podcast-feed-meta"
        >
          <span v-if="show?.language" class="rounded-full bg-overlay px-2 py-0.5 uppercase">{{ show.language }}</span>
          <span v-if="feedUpdated">{{ t('podcast.updated', { date: feedUpdated }) }}</span>
        </p>
        <p
          v-if="show?.description"
          class="mt-2 text-sm leading-relaxed text-muted"
          :class="descExpanded ? '' : 'line-clamp-[8]'"
        >
          {{ show.description }}
        </p>
        <!-- Collapsed shows ~8 lines (SD.2): enough to read what the show is before deciding to
             expand. The toggle only appears for descriptions long enough to actually be clamped at
             8 lines (~400+ chars), so medium ones that already fit show no redundant "show more". -->
        <button
          v-if="show?.description && show.description.length > 400"
          type="button"
          class="mt-1 text-xs font-bold text-accent"
          @click="descExpanded = !descExpanded"
        >
          {{ descExpanded ? t('podcast.showLess') : t('podcast.showMore') }}
        </button>

        <!-- Follow → feed subscription; its unheard episodes surface in Your Week. Rendered for
             signed-out visitors too (#1590) — the tap routes to sign-in. This is the primary follow
             surface, so hiding it hid the capability from everyone deciding whether to sign up. -->
      </div>
    </header>

    <!-- Activity first (SD.4): the publishing rhythm up top, right under the header, before the
         topic/people signals. -->
    <ShowActivityChart :episodes="episodes" />

    <!-- Show-level signals: what this show's about + who's on it (taps open the entity card). -->
    <PodcastSignalsBand :feed-id="feedId" @open="cardTarget = $event" />

    <!-- F1.3/F1.4: reserve the episode list's shape while loading (no jump when it fills) and offer
         a retry on failure, instead of a bare "Loading…"/error line. -->
    <SectionStatus
      v-if="episodes.length === 0 && (loading || error)"
      :phase="loading ? 'loading' : 'error'"
      :rows="4"
      @retry="loadMore"
    />
    <p v-else-if="episodes.length === 0" class="text-muted">{{ t('catalog.empty') }}</p>

    <div v-else>
      <!-- Hide-played toggle (SD.9): reads the completed set (mark-as-played). -->
      <label class="mb-3 flex w-fit items-center gap-2 text-sm font-semibold text-muted">
        <input v-model="hidePlayed" type="checkbox" data-testid="hide-played" class="accent-accent" />
        {{ t('podcast.hidePlayed') }}
      </label>
      <p v-if="visibleEpisodes.length === 0" class="text-muted">{{ t('podcast.allPlayed') }}</p>
      <!-- Highlight the latest episode at the top (SD.8): the list is newest-first, so the first is
           the newest — label it, then the rest follow. -->
      <template v-else>
        <span class="lp-kicker mb-1 block text-accent" data-testid="latest-label">{{ t('podcast.latest') }}</span>
        <EpisodeCard :episode="visibleEpisodes[0]" />
        <EpisodeCard v-for="ep in visibleEpisodes.slice(1)" :key="ep.slug" :episode="ep" />
      </template>
      <div class="mt-6 flex justify-center">
        <button
          v-if="hasMore"
          type="button"
          :disabled="loading"
          class="rounded-full border border-border px-5 py-2 font-bold disabled:opacity-50"
          @click="loadMore"
        >
          {{ loading ? t('catalog.loading') : t('catalog.loadMore') }}
        </button>
      </div>
    </div>

    <!-- Notes on this show (NT.1). -->
    <NoteComposer target="show" :target-id="feedId" />

    <EntityCard
      v-if="cardTarget"
      :kind="cardTarget.kind"
      :id="cardTarget.id"
      @close="cardTarget = null"
    />
  </section>
</template>
