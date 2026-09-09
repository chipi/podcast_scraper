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
 * * **The standard minimum action row, not all four.** The tile shows `EpisodeActions` —
 *   favourite, download, queue (the app-wide minimum; download self-hides on web). Add-to-
 *   collection stays OFF here: it belongs on a detail surface where you have already committed to
 *   the episode. Three 32px targets at `gap-3` sit comfortably across the 176px slot.
 * * **No overlay.** The actions sit BELOW the artwork. `ShowTile` overlays its single follow button
 *   deliberately, which works for one; two icons over episode art is the crowding this replaces.
 */
import { computed } from 'vue'
import { RouterLink } from 'vue-router'
import EpisodeActions from './EpisodeActions.vue'
import type { EpisodeSummary } from '../services/types'

const props = defineProps<{ episode: EpisodeSummary }>()

const artwork = computed(
  () => props.episode.artwork_url ?? props.episode.episode_image_url ?? props.episode.feed_image_url,
)
</script>

<template>
  <article class="flex flex-col gap-2">
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
    </RouterLink>

    <!-- Below the artwork, never over it. The shared minimum action row (favourite/download/queue);
         `gap-3` keeps the 32px hit areas from overlapping. -->
    <EpisodeActions :slug="episode.slug" />

    <RouterLink
      :to="{ name: 'player', params: { slug: episode.slug } }"
      class="block no-underline"
    >
      <span v-if="episode.podcast_title" class="lp-kicker block">{{ episode.podcast_title }}</span>
      <!--
        The title gets the tile's FULL width and up to three lines. It was getting ~100px beside the
        artwork, which is what turned one long name into eight lines. Clamped rather than truncated
        at one line: three lines is enough for almost every real title, and the rail needs its slots
        to stay the same height.
      -->
      <span
        class="mt-0.5 line-clamp-3 block font-display text-sm font-bold leading-snug text-canvas-foreground"
      >{{ episode.title }}</span>
    </RouterLink>
  </article>
</template>
