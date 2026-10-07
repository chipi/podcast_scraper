<script setup lang="ts">
/**
 * A show as a LIST ROW, built to the same proportions as {@link EpisodeCard} (operator 2026-09-17).
 *
 * ## Why this exists
 *
 * Shows had two list representations and neither matched the episode list they sit beside: Discover's
 * Shows tab used a 44px thumbnail with a title and an episode count, and Library's Saved tab had a
 * bare line of text. Both read as a different KIND of thing from the episode rows directly above
 * them, when a show and an episode are the same kind of thing to a reader — cover art, a name, a
 * line about it, something to open.
 *
 * So this is EpisodeCard's shape with a show's content: 128px artwork in the left column with the
 * facts and the actions under it, the name and description filling the right. Defined once and used
 * by both surfaces, so they cannot drift apart again — which is exactly how they got here.
 *
 * ## The description clamp
 *
 * Uses the shared `lp-media-*` window (see style.css): the aside sets the row's height and the prose
 * fills it and clips, rather than a fixed `line-clamp-N` that leaves dead space beside the artwork on
 * one row and overshoots on the next.
 */
import { computed, ref } from 'vue'
import { useClampedProse } from '../composables/useClampedProse'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import type { Podcast } from '../services/types'
import { showArtwork } from '../utils/episode'
import { formatPublishDate } from '../utils/format'
import { borderClass } from '../utils/highlightColors'

const props = defineProps<{
  show: Podcast
  /**
   * Put the `#actions` controls in a row UNDER the artwork, where {@link EpisodeCard} puts its own,
   * instead of over it. For surfaces that list shows and episodes together (Library › Saved): one
   * page, one place for controls (operator 2026-10-05). The overlay stays the default where a show
   * list stands alone and carries a labelled Follow, which does not fit beside a heart at 128px.
   */
  actionsBelow?: boolean
  /**
   * The saved colour (Library › Saved): the same left bar {@link EpisodeCard} draws, so a coloured
   * show reads as coloured in the list and not only in its small swatch (operator 2026-10-07).
   */
  color?: string | null
}>()
const { t, locale } = useI18n()

const art = computed(() => showArtwork(props.show))
const title = computed(() => props.show.title ?? props.show.feed_id)
const description = computed(() => props.show.description?.trim() ?? '')

/**
 * Show metadata, split by WHERE it belongs rather than dumped in one line.
 *
 * Under the artwork go the two facts about the FEED as an object — how many episodes, when it last
 * moved. Beside the title goes what the show IS — who makes it, what it is filed under — because
 * that reads as part of the identity, above the description it introduces.
 *
 * Cadence and typical length, which the show PAGE also shows, are deliberately absent: both are
 * derived from the episode list, and a list row holds only the catalogue record. Faking them from
 * `episode_count` would be inventing data.
 */
const updated = computed(() => formatPublishDate(props.show.last_updated ?? null, locale.value))
const identity = computed<string[]>(() => {
  const out: string[] = []
  if (props.show.authors?.length)
    out.push(t('podcast.byline', { authors: props.show.authors.join(', ') }))
  if (props.show.category) out.push(props.show.category)
  return out
})

// Read more/less. The measurement is `useClampedProse`, shared with EpisodeCard,
// PersonCardContent and PodcastView — this file used to carry its own copy, which is how the
// half-cut last line survived here after being fixed elsewhere (operator 2026-09-23, Browse › Shows).
const descExpanded = ref(false)
const descEl = ref<HTMLElement | null>(null)
const { clipped: descClipped } = useClampedProse(descEl, descExpanded)

const canExpand = computed(() => !!description.value && (descClipped.value || descExpanded.value))
</script>

<template>
  <article
    class="lp-media-row group relative -mx-3 gap-4 rounded-xl border-b border-border px-3 py-5 transition-colors sm:gap-5"
    :class="color ? ['border-l-4', borderClass(color)] : ''"
    data-testid="show-row"
  >
    <!-- LEFT: artwork with the controls OVER it, the episode count beneath. -->
    <div class="lp-media-aside">
      <div class="relative">
        <img
          v-if="art"
          :src="art"
          :alt="title"
          loading="lazy"
          class="h-32 w-32 rounded-lg bg-elevated object-cover"
        />
        <!-- Without a fixed-width child the column collapses and squeezes what sits beneath it. -->
        <div v-else class="h-32 w-32 rounded-lg bg-elevated" aria-hidden="true" />
        <!-- The surface's controls OVER the artwork as one right-aligned column — the same L the
             grid tile makes (operator 2026-09-17). Under the artwork they cost a row of vertical
             space on every entry and, at 128px, a labelled Follow plus the heart did not fit side by
             side anyway.
             Stacked and flush right so the two edges read as one object. The plate classes match
             ShowTile's and EpisodeActions' `overlay`, so contrast never depends on whatever artwork
             happens to be underneath. `z-30` keeps them above the title's stretched card-link
             overlay; `.prevent.stop` so acting never also opens the show. -->
        <div
          v-if="$slots.actions && !actionsBelow"
          class="absolute right-1.5 top-1.5 z-30 flex flex-col items-end gap-1.5 [&>button]:border-white/25 [&>button]:bg-black/55 [&>button]:shadow-lg [&>button]:backdrop-blur-sm"
          @click.prevent.stop
        >
          <slot name="actions" />
        </div>
      </div>
      <!-- Facts about the feed as an object, under the artwork — the slot the episode card gives its
           date and duration. Stacked rather than joined with separators: the column is 128px, so one
           line would wrap anyway and wrap in the wrong places. -->
      <div v-if="show.episode_count || updated" class="text-xs font-medium leading-snug text-muted">
        <div v-if="show.episode_count">
          {{ t('podcast.episodeCount', { count: show.episode_count }, show.episode_count) }}
        </div>
        <div v-if="updated">{{ t('podcast.updated', { date: updated }) }}</div>
      </div>
      <!-- The controls UNDER the artwork (`actionsBelow`): the episode card's row — same width as
           the artwork, same gap, unplated — so a show and an episode on one page act alike.
           `relative z-30` keeps them above the title's stretched link; `.prevent.stop` so acting
           never also opens the show. -->
      <div
        v-if="$slots.actions && actionsBelow"
        class="relative z-30 flex w-32 flex-wrap items-center gap-[12px]"
        data-testid="show-row-actions"
        @click.prevent.stop
      >
        <slot name="actions" />
      </div>
      <!-- Surface-specific INFORMATION under the artwork (Discover's trending sparkline), kept apart
           from `#actions` so a readout never lands in the overlay's control column. -->
      <div v-if="$slots.meta" class="relative z-30">
        <slot name="meta" />
      </div>
    </div>

    <div class="lp-media-body">
      <!-- The name WRAPS (UXS-014:70) — the width is elastic here, so there is no reserved-height
           row for a clamp to protect. The stretched ::after makes the whole row open the show. -->
      <!-- `#menu`: a ⋯ to the RIGHT of the name, where Search's episode cards put theirs (operator
           2026-10-05), so shows and episodes on one page carry their actions in the same place.
           `relative z-30` keeps it above the name's stretched link. -->
      <div class="flex items-start gap-2">
        <RouterLink
          :to="{ name: 'podcast', params: { feedId: show.feed_id } }"
          class="lp-show-name lp-show-name--3 min-w-0 flex-1 font-display text-lg font-bold leading-snug text-canvas-foreground no-underline after:absolute after:inset-0 sm:text-xl"
          :title="title"
          >{{ title }}</RouterLink
        >
        <div v-if="$slots.menu" class="relative z-30 shrink-0" data-testid="show-row-menu">
          <slot name="menu" />
        </div>
      </div>
      <!-- Who makes it and what it is filed under, between the title and the description: it belongs
           to the show's identity, so it reads above the blurb it introduces rather than below it
           (operator 2026-09-17). Joined, so separators fall only between values actually PRESENT —
           a template-level "· unless first" has to know what preceded it and gets it wrong the
           moment a field is missing. The row has the width for one line here; the artwork column
           does not, which is why the feed facts are stacked over there instead. -->
      <p v-if="identity.length" class="lp-kicker mt-0.5" data-testid="show-row-identity">
        {{ identity.join(' · ') }}
      </p>
      <div
        v-if="description"
        ref="descEl"
        class="lp-media-clip mt-2"
        :class="descExpanded ? 'lp-media-clip--open' : ''"
      >
        <p class="text-sm leading-relaxed text-muted">{{ description }}</p>
      </div>
      <button
        v-if="canExpand"
        type="button"
        class="lp-media-foot relative z-30 mt-1 w-fit text-xs font-bold text-accent transition hover:opacity-80"
        data-testid="show-row-read-more"
        :aria-expanded="descExpanded"
        @click="descExpanded = !descExpanded"
      >
        {{ descExpanded ? t('card.readLess') : t('card.readMore') }}
      </button>
    </div>

  </article>
</template>
