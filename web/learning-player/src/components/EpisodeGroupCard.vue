<script setup lang="ts">
/**
 * An episode heading a group of things found IN it — Search's matching passages, Saved's captures,
 * Revisit's due moments. ONE header for all three (operator 2026-10-05).
 *
 * They had drifted to opposite failures. Search used the full `EpisodeCard`: 128px artwork with the
 * date, the match count and three action circles stacked under it, then a full-width "Hide matches
 * ▲" row — about 300px of mostly empty space before the first match. Saved and Revisit used
 * `EpisodeRow`: a 40px thumbnail beside an unclamped title that ran five lines, with the show name
 * under it.
 *
 * The balance:
 *
 * * **80px artwork** — the height of what sits beside it (show name + a three-line title), so
 *   neither column leaves a gap. The size the compact cards already use.
 * * **Show name ABOVE the title, one line**, as on every episode card and tile.
 * * **Title: 16px, at most three lines** — the cap every rail uses.
 * * **ONE muted line under it** (`#meta`): when / how many.
 * * **The fold is a chevron in the header**, not a row of its own; beside it, ONE ⋯ carrying the
 *   episode's actions (download, collection, favourite, queue) instead of three circles.
 * * **No frame around the group.** A divider ends the header; the items follow.
 *
 * Groups start EXPANDED: collapsing tidies a long page, it is not a default that hides what the
 * listener asked for. `v-model:expanded` lets a view own the state; unbound, the card keeps its own.
 */
import { computed } from "vue"
import { useI18n } from "vue-i18n"
import { RouterLink } from "vue-router"
import EpisodeActions from "./EpisodeActions.vue"
import { episodeArtwork } from "../utils/episode"
import type { EpisodeSummary } from "../services/types"

const props = defineProps<{
  episode: Pick<
    EpisodeSummary,
    "slug" | "title" | "podcast_title" | "artwork_url" | "episode_image_url" | "feed_image_url"
  >
  /** Rows in the body. No rows, no fold control — there is nothing to collapse. */
  itemCount: number
  testid?: string
  toggleTestid?: string
}>()
const expanded = defineModel<boolean>("expanded", { default: true })

const { t } = useI18n()
const art = computed(() => episodeArtwork(props.episode))
</script>

<template>
  <li class="list-none" :data-testid="testid ?? 'episode-group'">
    <div class="border-b border-border pb-2">
      <div class="flex items-start gap-2">
        <RouterLink
          :to="{ name: 'player', params: { slug: episode.slug } }"
          class="flex min-w-0 flex-1 items-start gap-3 no-underline text-canvas-foreground"
          data-testid="episode-group-link"
        >
          <img
            v-if="art"
            :src="art"
            alt=""
            loading="lazy"
            class="h-20 w-20 shrink-0 rounded-lg bg-elevated object-cover"
          />
          <div v-else class="h-20 w-20 shrink-0 rounded-lg bg-elevated" />
          <span class="min-w-0 flex-1">
            <!-- ONE line: the show is context; the title is what the row is for. -->
            <span
              v-if="episode.podcast_title"
              class="lp-kicker block truncate"
              :title="episode.podcast_title"
            >{{ episode.podcast_title }}</span>
            <span
              class="mt-0.5 line-clamp-3 font-display text-base font-bold leading-snug"
              :title="episode.title"
              data-testid="episode-group-title"
            >{{ episode.title }}</span>
            <slot name="extra" />
          </span>
        </RouterLink>
        <!-- Siblings of the link, never inside it: an interactive inside an interactive loses its
             accessible name (ShowTile's 2026-09-26 Android audit). -->
        <div class="flex shrink-0 flex-col items-center gap-1">
          <EpisodeActions :slug="episode.slug" hide-favorite hide-queue />
        </div>
      </div>
      <!-- The fold sits on the BOTTOM row, beside the count of what it folds (operator 2026-10-08):
           up in the side column, under the ⋯ menu and level with "Matched:", it read as part of the
           episode's actions rather than as "open / close these matches". -->
      <div v-if="$slots.meta || itemCount > 0" class="mt-1.5 flex items-center gap-2">
        <p v-if="$slots.meta" class="lp-kicker min-w-0 flex-1" data-testid="episode-group-meta">
          <slot name="meta" />
        </p>
        <button
          v-if="itemCount > 0"
          type="button"
          class="lp-tap ml-auto flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-bold text-accent"
          :aria-expanded="expanded"
          :aria-label="
            expanded
              ? t('highlights.collapseGroup', { title: episode.title })
              : t('highlights.expandGroup', { title: episode.title })
          "
          :data-testid="toggleTestid ?? 'episode-group-toggle'"
          @click="expanded = !expanded"
        >{{ expanded ? "▲" : "▼" }}</button>
      </div>
    </div>
    <!-- `v-show`, not `v-if`: collapsing keeps the state of what is inside (Search's folded clusters
         expand in place), and re-opening is instant. -->
    <div v-show="expanded" class="mt-2" data-testid="episode-group-body">
      <slot />
    </div>
  </li>
</template>
