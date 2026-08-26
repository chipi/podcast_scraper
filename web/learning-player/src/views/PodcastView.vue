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
import EntityCard from '../components/EntityCard.vue'
import EpisodeCard from '../components/EpisodeCard.vue'
import PodcastSignalsBand from '../components/PodcastSignalsBand.vue'
import ShowActivityChart from '../components/ShowActivityChart.vue'
import { getPodcasts, listPodcastEpisodes } from '../services/api'
import { useAuthStore } from '../stores/auth'
import { useLibraryStore } from '../stores/library'
import { useSignInGate } from '../composables/useSignInGate'
import { showArtwork } from '../utils/episode'
import type { EpisodeSummary, Podcast } from '../services/types'

const PAGE_SIZE = 20
const props = defineProps<{ feedId: string }>()
const { t } = useI18n()
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
const cardTarget = ref<{ kind: 'person' | 'topic'; id: string } | null>(null)

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
})
watch(() => props.feedId, reset)
</script>

<template>
  <section>
    <button type="button" class="lp-nav" @click="goBack">‹ {{ t('nav.back') }}</button>

    <header class="mb-6 mt-2 flex gap-4 sm:gap-5">
      <img
        v-if="show && showArt(show)"
        :src="showArt(show)!"
        :alt="show.title ?? ''"
        class="h-20 w-20 shrink-0 rounded-xl bg-elevated object-cover sm:h-28 sm:w-28"
      />
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
        <p v-if="total" class="mt-1 text-sm text-muted">
          {{ t('podcast.episodeCount', { count: total }, total) }}
        </p>
        <p
          v-if="show?.description"
          class="mt-2 text-sm leading-relaxed text-muted"
          :class="descExpanded ? '' : 'line-clamp-3'"
        >
          {{ show.description }}
        </p>
        <button
          v-if="show?.description && show.description.length > 180"
          type="button"
          class="mt-1 text-xs font-bold text-accent"
          @click="descExpanded = !descExpanded"
        >
          {{ descExpanded ? t('podcast.showLess') : t('podcast.showMore') }}
        </button>

        <!-- Follow → feed subscription; its unheard episodes surface in Your Week. Rendered for
             signed-out visitors too (#1590) — the tap routes to sign-in. This is the primary follow
             surface, so hiding it hid the capability from everyone deciding whether to sign up. -->
        <div class="mt-3 flex items-center gap-2">
          <button
            type="button"
            data-testid="follow-show"
            class="inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-bold transition disabled:opacity-50"
            :class="
              following ? 'bg-accent text-accent-foreground' : 'bg-overlay text-canvas-foreground hover:bg-elevated'
            "
            :aria-pressed="isGated ? undefined : following"
            :disabled="togglingFollow"
            :title="isGated ? t('auth.signInToFollow') : t('podcast.followHint')"
            :aria-label="isGated ? t('auth.signInToFollow') : undefined"
            @click="toggleFollow"
          >
            <span aria-hidden="true">{{ following ? '✓' : '+' }}</span>
            {{ following ? t('podcast.following') : t('podcast.follow') }}
          </button>
          <!-- Pin this show into a collection (RFC-119). -->
          <AddToCollectionButton :item="{ kind: 'show', ref: feedId }" />
        </div>
      </div>
    </header>

    <!-- Show-level signals: what this show's about + who's on it (taps open the entity card). -->
    <PodcastSignalsBand :feed-id="feedId" @open="cardTarget = $event" />

    <!-- Publishing cadence over time (from the loaded episodes' dates). -->
    <ShowActivityChart :episodes="episodes" />

    <p v-if="loading && episodes.length === 0" class="text-muted">{{ t('catalog.loading') }}</p>
    <p v-else-if="error && episodes.length === 0" class="text-danger">{{ t('catalog.loadError') }}</p>
    <p v-else-if="episodes.length === 0" class="text-muted">{{ t('catalog.empty') }}</p>

    <div v-else>
      <EpisodeCard v-for="ep in episodes" :key="ep.slug" :episode="ep" />
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

    <EntityCard
      v-if="cardTarget"
      :kind="cardTarget.kind"
      :id="cardTarget.id"
      @close="cardTarget = null"
    />
  </section>
</template>
