<script setup lang="ts">
/**
 * Compact episode LIST-ROW — thumbnail + title + show kicker, linking to the player. ONE idiom for
 * the dense episode lists on the entity card (topic/person episodes), the storyline sheet, and the
 * Knowledge Panel's "More like this". Was triplicated markup that had already drifted (border on the
 * link vs the li, items-start vs items-center). Top-aligned, per the image-left/text-right rule.
 *
 * `#trailing` is a sibling of the link (never nested — no interactive-in-interactive), for a row
 * action like the Knowledge Panel's play-next. `@navigate` fires on tap, for hosts that must close
 * (the entity card dismisses its overlay).
 */
import { RouterLink } from "vue-router"
import { episodeArtwork } from "../utils/episode"
import type { EpisodeSummary } from "../services/types"

defineProps<{ episode: EpisodeSummary }>()
const emit = defineEmits<{ (e: "navigate"): void }>()
const art = episodeArtwork
</script>

<template>
  <div class="flex items-start gap-1 border-b border-border" data-testid="episode-row">
    <RouterLink
      :to="{ name: 'player', params: { slug: episode.slug } }"
      class="flex min-w-0 flex-1 items-start gap-3 py-2 no-underline text-canvas-foreground hover:bg-overlay"
      @click="emit('navigate')"
    >
      <img
        v-if="art(episode)"
        :src="art(episode)!"
        alt=""
        loading="lazy"
        class="h-10 w-10 shrink-0 rounded-md bg-elevated object-cover"
      />
      <div v-else class="h-10 w-10 shrink-0 rounded-md bg-elevated" />
      <span class="min-w-0 flex-1">
        <span class="block text-sm font-semibold">{{ episode.title }}</span>
        <span v-if="episode.podcast_title" class="lp-kicker block">{{
          episode.podcast_title
        }}</span>
      </span>
    </RouterLink>
    <slot name="trailing" />
  </div>
</template>
