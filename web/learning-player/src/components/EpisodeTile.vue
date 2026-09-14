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
  <article class="flex h-full flex-col gap-2">
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

    <!-- Actions at the BOTTOM (operator), matching the list card. `mt-auto` drops them to the foot
         of the tile so every tile lines its action row up regardless of how many lines its title
         took — but that only works because the article is `h-full` and the rail stretches each slot
         to the tallest tile; without `h-full` the article is content-height and the actions sit
         unevenly right under each title (operator 2026-09-14). The shared EpisodeActions set owns its
         own tap-target spacing. -->
    <EpisodeActions :slug="episode.slug" class="mt-auto pt-1" />
  </article>
</template>
