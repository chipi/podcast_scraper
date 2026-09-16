<script setup lang="ts">
/**
 * Compact episode LIST-ROW — thumbnail + title + show kicker, linking to the player. ONE idiom for
 * the dense episode lists on the entity card (topic/person episodes), the storyline sheet, and the
 * Knowledge Panel's "More like this". Was triplicated markup that had already drifted (border on the
 * link vs the li, items-start vs items-center). Top-aligned, per the image-left/text-right rule.
 *
 * `#trailing` is a sibling of the link (never nested — no interactive-in-interactive), for a row
 * action like the Knowledge Panel's play-next.
 *
 * ## There is deliberately NO close hook on tap (2026-09-16)
 *
 * This used to emit `navigate` so a host sheet could dismiss itself, and the three entity sheets
 * wired it straight to `emit('close')`. That BROKE the link. Closing explicitly is the
 * "✕ / Escape / backdrop" path, which still owes a `router.back()` to pop the history entry the
 * sheet pushed on open — and it ran before the route settled, so the `back()` popped the push to
 * the player. Tapping an episode from a topic or person card landed on HOME instead of the episode.
 *
 * A host sheet does not need the hook: navigating changes the route, the query the sheet pushed
 * disappears, and `useModalSheet`'s watcher closes it via `closedByNavigation` — with nothing to
 * undo. `EntityCardBody`'s "Open in page" link had already learned exactly this and carries the
 * same warning; the lesson simply never reached these rows.
 */
import { RouterLink } from "vue-router"
import { episodeArtwork } from "../utils/episode"
import type { EpisodeSummary } from "../services/types"

defineProps<{ episode: EpisodeSummary }>()
const art = episodeArtwork
</script>

<template>
  <div class="flex items-start gap-1 border-b border-border" data-testid="episode-row">
    <RouterLink
      :to="{ name: 'player', params: { slug: episode.slug } }"
      class="flex min-w-0 flex-1 items-start gap-3 py-2 no-underline text-canvas-foreground hover:bg-overlay"
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
