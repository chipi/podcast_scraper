<script setup lang="ts">
/**
 * Your Library (UXS-014) — the hub for everything per-user: Saved (polymorphic favorites, grouped
 * by kind), Queue, and Shows. One place, tabbed; scales as new favorite kinds arrive. Auth-gated.
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { getEpisode, getPlaybackList } from '../services/api'
import type { EpisodeDetail } from '../services/types'
import { useFavoritesStore } from '../stores/favorites'
import { formatTime } from '../player/transcriptSync'
import EpisodeCard from '../components/EpisodeCard.vue'
import QueueView from './QueueView.vue'

const { t } = useI18n()
const favorites = useFavoritesStore()

// Tabs: Saved (favorites) · Queue (what's next) · Recent (what you've heard). "Shows" returns once
// subscriptions are user-curated — today there's no way to curate them, so we don't fake it.
type Tab = 'saved' | 'queue' | 'recent'
const tab = ref<Tab>('saved')
const tabs: { key: Tab; label: string }[] = [
  { key: 'saved', label: 'library.saved' },
  { key: 'queue', label: 'library.queue' },
  { key: 'recent', label: 'library.recent' },
]

const hasSaved = computed(() => favorites.count > 0)

// Recent listens = the per-user playback history, newest-played first (same source as Home's
// "Continue"); hydrate slugs to titles/artwork (small histories).
const recent = ref<{ detail: EpisodeDetail; position: number }[]>([])
const art = (d: EpisodeDetail) => d.artwork_url || d.episode_image_url || d.feed_image_url

async function loadRecent(): Promise<void> {
  const positions = await getPlaybackList().catch(() => [])
  const hydrated = await Promise.all(
    positions.slice(0, 30).map((p) =>
      getEpisode(p.slug)
        .then((detail) => ({ detail, position: p.position_seconds }))
        .catch(() => null),
    ),
  )
  recent.value = hydrated.filter((x): x is { detail: EpisodeDetail; position: number } => !!x)
}

onMounted(async () => {
  await favorites.ensureLoaded()
  await loadRecent()
})
</script>

<template>
  <section>
    <h1 class="mb-4 font-display text-3xl font-extrabold tracking-tight">{{ t('library.title') }}</h1>

    <!-- Tabs -->
    <div class="mb-6 flex gap-1 border-b border-border">
      <button
        v-for="tb in tabs"
        :key="tb.key"
        type="button"
        class="-mb-px border-b-2 px-4 py-2 text-sm font-bold transition"
        :class="tab === tb.key ? 'border-accent text-canvas-foreground' : 'border-transparent text-muted hover:text-canvas-foreground'"
        @click="tab = tb.key"
      >{{ t(tb.label) }}</button>
    </div>

    <!-- Saved (favorites) -->
    <div v-show="tab === 'saved'">
      <p v-if="!hasSaved" class="text-muted">{{ t('library.savedEmpty') }}</p>
      <template v-else>
        <section v-if="favorites.episodes.length" class="mb-8">
          <h2 class="lp-kicker mb-2">{{ t('library.episodes') }}</h2>
          <EpisodeCard v-for="e in favorites.episodes" :key="e.slug" :episode="e" />
        </section>
        <section v-if="favorites.insights.length">
          <h2 class="lp-kicker mb-2">{{ t('library.insights') }}</h2>
          <ul class="flex flex-col">
            <li v-for="ins in favorites.insights" :key="ins.ref" class="border-b border-border py-3">
              <RouterLink
                v-if="ins.episode_slug"
                :to="{ name: 'player', params: { slug: ins.episode_slug }, query: ins.start_ms != null ? { t: String(Math.floor(ins.start_ms / 1000)) } : {} }"
                class="block no-underline text-canvas-foreground"
              >
                <p class="text-sm font-semibold leading-snug">{{ ins.text }}</p>
                <p class="lp-kicker mt-1">
                  {{ ins.podcast_title
                  }}<template v-if="ins.podcast_title && ins.start_ms != null"> · </template
                  ><span v-if="ins.start_ms != null" class="text-accent">▶ {{ formatTime(ins.start_ms / 1000) }}</span>
                </p>
              </RouterLink>
              <p v-else class="text-sm font-semibold leading-snug text-muted">{{ ins.text }}</p>
            </li>
          </ul>
        </section>
      </template>
    </div>

    <!-- Queue (embeds the existing view, sans its heading) -->
    <div v-show="tab === 'queue'">
      <QueueView hide-title />
    </div>

    <!-- Recent listens (playback history, newest first) -->
    <div v-show="tab === 'recent'">
      <p v-if="!recent.length" class="text-muted">{{ t('library.recentEmpty') }}</p>
      <ul v-else class="flex flex-col">
        <li v-for="r in recent" :key="r.detail.slug" class="border-b border-border py-3">
          <RouterLink
            :to="{ name: 'player', params: { slug: r.detail.slug }, query: r.position > 1 ? { t: String(Math.floor(r.position)) } : {} }"
            class="flex items-center gap-3 no-underline text-canvas-foreground"
          >
            <img
              v-if="art(r.detail)"
              :src="art(r.detail)!"
              alt=""
              loading="lazy"
              class="h-12 w-12 shrink-0 rounded-md bg-elevated object-cover"
            />
            <div v-else class="h-12 w-12 shrink-0 rounded-md bg-elevated" />
            <span class="min-w-0 flex-1">
              <span class="block truncate font-display font-bold leading-snug">{{ r.detail.title }}</span>
              <span v-if="r.detail.podcast_title" class="lp-kicker block">{{ r.detail.podcast_title }}</span>
            </span>
            <span v-if="r.position > 1" class="shrink-0 font-mono text-xs text-muted">{{ formatTime(r.position) }}</span>
          </RouterLink>
        </li>
      </ul>
    </div>
  </section>
</template>
