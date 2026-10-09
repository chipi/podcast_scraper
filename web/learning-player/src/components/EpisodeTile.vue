<script setup lang="ts">
/**
 * An episode as a TILE, for horizontal rails (#2004 follow-up).
 *
 * ## Why not `EpisodeCard compact`
 *
 * `EpisodeCard` is a horizontal card: artwork column on the left, text column on the right. That is
 * right in a vertical list (Podcast, Queue, the Queue panel's "recently played") and wrong in a
 * 224px rail slot, because the text column gets what is left — about 100px. In "More like this" a
 * real title wrapped to EIGHT lines, the card grew to roughly 800px tall, and the action row, which
 * is positioned against the card's top-right, ended up floating over the artwork.
 *
 * A rail slot is narrow and repeated. It wants a tile: artwork on top at the full width, everything
 * else stacked beneath it. `compact` stays as it is for the vertical lists where it works — this is
 * a different shape, not a variant of that one.
 *
 * ## What it deliberately does NOT show
 *
 * * **No summary.** There is no room for one at this width, and a two-line truncated fragment is
 *   not a summary — it is the shape of one. The title earns the space instead.
 * * **The shared action row — same set as the list card.** The tile shows `EpisodeActions` —
 *   favourite, queue, download, add-to-collection. It used to omit add-to-collection, which is
 *   exactly what made the grid show 3 icons while the list showed 4; the action count must not
 *   change with the type of view (operator 2026-09-13). Download self-hides on web.
 *
 * ## The actions OVERLAY the artwork
 *
 * They used to sit below it, on the reasoning that icons over episode art crowd it. Home's
 * "Recommended for you" is the same shape — a square-artwork tile in a 2/3/4-column grid — and it
 * overlays, so the two grids disagreed about where an episode's controls live (operator
 * 2026-09-17). Overlaying also gives the title back the vertical space the row cost.
 *
 * The crowding objection was really about WIDTH: an absolutely-positioned row sizes to max-content
 * and will not wrap, so four icons ran off a narrow 2-column phone tile. `max-w-[76px]` is the fix
 * Home already uses — the row wraps two-up in the corner instead of spilling.
 */
import { computed } from 'vue'
import { RouterLink } from 'vue-router'
import EpisodeActions from './EpisodeActions.vue'
import LanguageBadge from './LanguageBadge.vue'
import PlayedBadge from './PlayedBadge.vue'
import type { EpisodeSummary } from '../services/types'
import { episodeArtwork } from '../utils/episode'

const props = defineProps<{
  /** Only what the tile draws — Jump back in hands it an EpisodeDetail, rails an EpisodeSummary. */
  episode: Pick<
    EpisodeSummary,
    | 'slug'
    | 'title'
    | 'podcast_title'
    | 'artwork_url'
    | 'episode_image_url'
    | 'feed_image_url'
    | 'language'
  >
  /** 0..1 — how far the listener got. Drawn under the artwork (Jump back in). */
  progress?: number
}>()

// Absolutised via the shared helper — a raw relative URL 404s against `capacitor://localhost`.
const artwork = computed(() => episodeArtwork(props.episode))
</script>

<template>
  <article class="relative flex h-full flex-col gap-2">
    <!-- Capped to the artwork's corner so the icon row WRAPS two-up rather than spilling past a
         narrow 2-column phone tile — an absolutely-positioned row sizes to max-content and will not
         wrap unbounded. Same treatment as Home's "Recommended for you". -->
    <EpisodeActions
      :slug="episode.slug"
      overlay
      class="absolute right-2 top-2 z-10 max-w-[76px] justify-end"
    />
    <!-- Top-LEFT, opposite the actions, on the same dark plate so it reads over any artwork. -->
    <LanguageBadge
      :lang="episode.language"
      overlay
      class="pointer-events-none absolute left-2 top-2 z-10"
    />
    <RouterLink
      :to="{ name: 'player', params: { slug: episode.slug } }"
      class="block no-underline"
    >
      <img
        v-if="artwork"
        :src="artwork"
        :alt="episode.podcast_title ?? ''"
        loading="lazy"
        class="aspect-square w-full rounded-xl bg-elevated object-cover"
      />
      <div v-else class="aspect-square w-full rounded-xl bg-elevated" />
      <div v-if="progress != null" class="mt-2 h-1 rounded bg-overlay" data-testid="episode-tile-progress">
        <div class="h-1 rounded bg-accent" :style="{ width: Math.min(100, Math.max(0, progress * 100)) + '%' }" />
      </div>
    </RouterLink>

    <RouterLink
      :to="{ name: 'player', params: { slug: episode.slug } }"
      class="block no-underline"
    >
      <span v-if="episode.podcast_title" class="lp-kicker lp-show-name" :title="episode.podcast_title">{{ episode.podcast_title }}</span>
      <!-- Above the title, not below it: the title is clamped to three lines and a marker after it
           would sit at a different height on every tile in the row. -->
      <PlayedBadge :slug="episode.slug" class="mt-1" />
      <!--
        The title gets the tile's FULL width and three lines, RESERVED (`lp-tile-title`, the one rule
        every tile shares). It was getting ~100px beside the artwork, which is what turned one long
        name into eight lines.
      -->
      <span
        class="lp-tile-title mt-0.5 font-display text-sm font-bold text-canvas-foreground"
        :title="episode.title"
      >{{ episode.title }}</span>
    </RouterLink>

  </article>
</template>
