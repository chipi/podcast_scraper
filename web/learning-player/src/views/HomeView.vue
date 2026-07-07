<script setup lang="ts">
/**
 * Home — the Learning Hub (PRD-042 / UXS-012). Adaptive hero: resume-state (Continue) when
 * signed-in with in-progress history, else discover-state ("Ask your library" + Featured).
 * Corpus search is prominent in both states. Sections (What's new / Recommended / Your shows)
 * hide cleanly when empty or signed-out. All data from the real /api/app/* surface.
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRouter } from 'vue-router'
import {
  getDiscover,
  getEpisode,
  getPlaybackList,
  getPodcasts,
  getRelated,
  recordDiscoverClick,
} from '../services/api'
import type { EpisodeDetail, EpisodeSummary, Podcast } from '../services/types'
import { formatTime } from '../player/transcriptSync'
import { formatDuration } from '../utils/format'
import { episodeArtwork, showArtwork } from '../utils/episode'
import { useAuthStore } from '../stores/auth'
import EntityCard from '../components/EntityCard.vue'
import InterestsPicker from '../components/InterestsPicker.vue'
import QueueButton from '../components/QueueButton.vue'
import Storylines from '../components/Storylines.vue'
import TrendingTopics from '../components/TrendingTopics.vue'

const INTERESTS_DISMISSED_KEY = 'lp.interests.dismissed'

const { t } = useI18n()
const router = useRouter()
const auth = useAuthStore()

const latest = ref<EpisodeSummary[]>([])
const shows = ref<Podcast[]>([])
const recommended = ref<EpisodeSummary[]>([])
const continueItems = ref<{ detail: EpisodeDetail; position: number }[]>([])
const query = ref('')

// Trending-topic chip → open the topic entity card (overlay), same surface as Search.
const cardTarget = ref<{ kind: 'person' | 'topic'; id: string } | null>(null)

// First-Home dismissible "set your interests" card → opens the picker (PRD-043 FR4 / 3.5).
const interestsDismissed = ref(false)
const pickerOpen = ref(false)
const showInterestsCard = computed(() => auth.isAuthenticated && !interestsDismissed.value)

function dismissInterests(): void {
  interestsDismissed.value = true
  try {
    localStorage.setItem(INTERESTS_DISMISSED_KEY, '1')
  } catch {
    /* private mode / storage disabled — the card just reappears next load */
  }
}

async function onInterestsSaved(): Promise<void> {
  dismissInterests()
  // Re-pull discovery so a personalized order (when the flag is on) takes effect immediately.
  latest.value = (await getDiscover(8).catch(() => null))?.items ?? latest.value
}

const resumeState = computed(() => auth.isAuthenticated && continueItems.value.length > 0)
// Editorial ranked "What's new": a featured #1 + ranked rows — all on screen, no scroll.
const wnFeatured = computed(() => latest.value[0] ?? null)
const wnRows = computed(() => latest.value.slice(1, 6))
const rank = (i: number) => String(i + 2).padStart(2, '0')
const resumeTop = computed(() => continueItems.value[0] ?? null)
const resumeArt = episodeArtwork
const showArt = showArtwork
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
  latest.value = (await getDiscover(8).catch(() => null))?.items ?? []
  getPodcasts()
    .then((s) => (shows.value = s))
    .catch(() => (shows.value = []))

  if (auth.isAuthenticated || !auth.loaded) {
    const positions = await getPlaybackList().catch(() => [])
    const inProgress = positions.filter((p) => p.position_seconds > 1).slice(0, 6)
    const hydrated = await Promise.all(
      inProgress.map((p) =>
        getEpisode(p.slug)
          .then((detail) => ({ detail, position: p.position_seconds }))
          .catch(() => null),
      ),
    )
    continueItems.value = hydrated.filter((x): x is { detail: EpisodeDetail; position: number } => !!x)
    // Recommended = peers of the most-recent play (v1 heuristic; PRD-041 supersedes).
    if (continueItems.value[0]) {
      recommended.value =
        (await getRelated(continueItems.value[0].detail.slug).catch(() => null))?.items ?? []
    }
  }
})
</script>

<template>
  <section>
    <!-- Adaptive hero -->
    <div v-if="resumeState && resumeTop" class="relative overflow-hidden rounded-2xl border border-border">
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
      <button
        type="button"
        class="shrink-0 rounded-full bg-accent px-4 py-2 text-sm font-bold text-accent-foreground"
        @click="pickerOpen = true"
      >
        {{ t('interests.cardCta') }}
      </button>
      <button type="button" class="shrink-0 text-sm text-muted" @click="dismissInterests">
        {{ t('interests.dismiss') }}
      </button>
    </section>

    <!-- What's new — editorial ranked: a featured #1 + ranked rows, all on screen, NO scroll -->
    <section v-if="wnFeatured" class="mt-7">
      <div class="mb-3 flex items-baseline justify-between">
        <h2 class="lp-section">{{ t('home.whatsNew') }}</h2>
        <RouterLink :to="{ name: 'catalog' }" class="text-sm font-bold text-accent no-underline">
          {{ t('home.browseAll') }} →
        </RouterLink>
      </div>

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
    </section>

    <!-- Trending topics (Plan B): corpus-wide "heating up" from temporal_velocity. -->
    <TrendingTopics @open="cardTarget = { kind: 'topic', id: $event }" />

    <!-- Storylines (B): theme clusters — topics discussed together. Opens the anchor topic card. -->
    <Storylines @open="cardTarget = { kind: 'topic', id: $event }" />

    <!-- Recommended — no-scroll responsive grid -->
    <section v-if="recommended.length" class="mt-7">
      <h2 class="lp-section mb-3">{{ t('home.recommended') }}</h2>
      <ul class="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
        <li v-for="ep in recommended.slice(0, 8)" :key="ep.slug" class="relative">
          <QueueButton :slug="ep.slug" class="absolute right-2 top-2 z-10 bg-canvas/70 backdrop-blur" />
          <RouterLink :to="{ name: 'player', params: { slug: ep.slug } }" class="block no-underline text-canvas-foreground">
            <img v-if="epArt(ep)" :src="epArt(ep)!" alt="" class="aspect-square w-full rounded-xl object-cover bg-elevated" />
            <div v-else class="aspect-square w-full rounded-xl bg-elevated" />
            <div class="mt-2 line-clamp-2 text-sm font-bold leading-tight">{{ ep.title }}</div>
            <div class="lp-kicker mt-0.5">{{ ep.podcast_title }}</div>
          </RouterLink>
        </li>
      </ul>
    </section>

    <InterestsPicker v-if="pickerOpen" @close="pickerOpen = false" @saved="onInterestsSaved" />

    <!-- Your shows -->
    <section v-if="shows.length" class="mt-7">
      <h2 class="lp-section mb-3">{{ t('home.shows') }}</h2>
      <ul class="grid grid-cols-3 gap-3 sm:grid-cols-4">
        <li v-for="p in shows" :key="p.feed_id">
          <RouterLink :to="{ name: 'podcast', params: { feedId: p.feed_id } }" class="block no-underline text-canvas-foreground">
            <img v-if="showArt(p)" :src="showArt(p)!" alt="" class="aspect-square w-full rounded-xl object-cover bg-elevated" />
            <div v-else class="aspect-square w-full rounded-xl bg-elevated" />
            <div class="mt-1 text-xs font-bold">{{ p.title ?? p.feed_id }}</div>
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
