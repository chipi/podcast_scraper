<script setup lang="ts">
/**
 * Editorial-bold episode card (UXS-011 / PRD-038 FR3–FR4). Degrades cleanly: absent fields
 * (artwork, summary, topics, GI) are omitted, never shown empty.
 *
 * Uses the "stretched link" pattern (no nested anchors): the title link's ::after overlay
 * covers the whole card → Player, while the podcast kicker link sits above it (z-10) so it
 * independently navigates to that podcast's catalog view.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import type { EpisodeSummary } from '../services/types'
import { formatDuration, formatPublishDate } from '../utils/format'

const props = defineProps<{ episode: EpisodeSummary }>()
const { t, locale } = useI18n()

const duration = computed(() => formatDuration(props.episode.duration_seconds))
const date = computed(() => formatPublishDate(props.episode.publish_date, locale.value))
const topics = computed(() => props.episode.topics.slice(0, 4))
// Prefer our locally-stored copy (artwork_url); fall back to the remote feed image URLs.
const artwork = computed(
  () =>
    props.episode.artwork_url ||
    props.episode.episode_image_url ||
    props.episode.feed_image_url,
)
</script>

<template>
  <article class="relative -mx-2 flex gap-3 rounded border-b border-border px-2 py-4 hover:bg-overlay">
    <img
      v-if="artwork"
      :src="artwork"
      :alt="episode.podcast_title ?? ''"
      loading="lazy"
      class="h-16 w-16 shrink-0 rounded-md bg-elevated object-cover"
    />
    <div class="min-w-0 flex-1">
      <div class="flex items-baseline justify-between gap-3">
        <RouterLink
          :to="{ name: 'player', params: { slug: episode.slug } }"
          class="font-display text-lg font-bold leading-snug text-canvas-foreground no-underline after:absolute after:inset-0"
        >
          {{ episode.title }}
        </RouterLink>
        <span
          class="shrink-0 text-xs"
          :class="episode.status === 'ready' ? 'text-grounded' : 'text-warning'"
        >
          {{ episode.status === 'ready' ? t('status.ready') : t('status.pending') }}
        </span>
      </div>
      <RouterLink
        v-if="episode.podcast_title"
        :to="{ name: 'podcast', params: { feedId: episode.feed_id } }"
        class="lp-kicker relative z-10 inline-block no-underline"
      >
        {{ episode.podcast_title }}
      </RouterLink>
      <p v-if="episode.summary_preview" class="mt-1 line-clamp-2 text-sm text-muted">
        {{ episode.summary_preview }}
      </p>
      <div class="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
        <span v-if="date">{{ date }}</span>
        <span v-if="duration">{{ duration }}</span>
        <span v-if="episode.has_gi" class="text-grounded">● {{ t('catalog.insightsBadge') }}</span>
        <span
          v-for="topic in topics"
          :key="topic"
          class="relative z-10 rounded-full bg-overlay px-2 py-0.5 text-topic"
        >
          {{ topic }}
        </span>
      </div>
    </div>
  </article>
</template>
