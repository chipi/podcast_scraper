<script setup lang="ts">
/**
 * Your Library (UXS-014) — the hub for everything per-user, tabbed: Saved (favorited episodes) ·
 * Knowledge (saved insights) · Queue (what's next) · Recent (playback history). One place, tabbed;
 * scales as new favorite kinds arrive. Auth-gated.
 */
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import { getEpisode, getPlaybackList } from '../services/api'
import type { EpisodeDetail } from '../services/types'
import { useFavoritesStore } from '../stores/favorites'
import { formatTime } from '../player/transcriptSync'
import { summaryFromDetail } from '../utils/episode'
import EpisodeCard from '../components/EpisodeCard.vue'
import QueueView from './QueueView.vue'
import HighlightsView from './HighlightsView.vue'

const { t } = useI18n()
const favorites = useFavoritesStore()

// Tabs: Saved (episodes) · Highlights (captured moments/spans) · Knowledge (saved insights) ·
// Queue (what's next) · Recent (heard). "Shows" returns once subscriptions are user-curated.
type Tab = 'saved' | 'highlights' | 'knowledge' | 'queue' | 'recent'
const tab = ref<Tab>('saved')
const tabs: { key: Tab; label: string }[] = [
  { key: 'saved', label: 'library.saved' },
  { key: 'highlights', label: 'library.highlights' },
  { key: 'knowledge', label: 'library.knowledge' },
  { key: 'queue', label: 'library.queue' },
  { key: 'recent', label: 'library.recent' },
]

// Recent listens = the per-user playback history, newest-played first (same source as Home's
// "Continue"); hydrate slugs to full episodes so they showcase through the shared card. The player
// auto-resumes from the saved position, so the card needs no separate "resume at" affordance.
const recent = ref<EpisodeDetail[]>([])

async function loadRecent(): Promise<void> {
  const positions = await getPlaybackList().catch(() => [])
  const hydrated = await Promise.all(
    positions.slice(0, 30).map((p) => getEpisode(p.slug).catch(() => null)),
  )
  recent.value = hydrated.filter((d): d is EpisodeDetail => !!d)
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

    <!-- Saved — favorited episodes. -->
    <div v-show="tab === 'saved'">
      <p v-if="!favorites.episodes.length" class="text-muted">{{ t('library.savedEmpty') }}</p>
      <section v-else class="flex flex-col">
        <EpisodeCard v-for="e in favorites.episodes" :key="e.slug" :episode="e" />
      </section>
    </div>

    <!-- Highlights — captured moments / spans / saved insights, grouped by episode, with notes. -->
    <div v-show="tab === 'highlights'">
      <HighlightsView />
    </div>

    <!-- Knowledge — saved insights (snapshot text + jump-to-moment). -->
    <div v-show="tab === 'knowledge'">
      <p v-if="!favorites.insights.length" class="text-muted">{{ t('library.knowledgeEmpty') }}</p>
      <ul v-else class="flex flex-col">
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
    </div>

    <!-- Queue (embeds the existing view, sans its heading) -->
    <div v-show="tab === 'queue'">
      <QueueView hide-title />
    </div>

    <!-- Recent listens (playback history, newest first) — showcased through the shared card. -->
    <div v-show="tab === 'recent'">
      <p v-if="!recent.length" class="text-muted">{{ t('library.recentEmpty') }}</p>
      <div v-else class="flex flex-col">
        <EpisodeCard v-for="d in recent" :key="d.slug" :episode="summaryFromDetail(d)" />
      </div>
    </div>
  </section>
</template>
