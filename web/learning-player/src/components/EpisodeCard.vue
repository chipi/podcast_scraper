<script setup lang="ts">
/**
 * Editorial-bold episode card (UXS-011 / PRD-038 FR3–FR4). A clean two-line lede, with grounded
 * insights **one tap away, expanding in place** — the same gesture on touch and pointer.
 *
 * Uses the "stretched link" pattern (no nested anchors): the title link's ::after overlay covers
 * the whole card → Player; the podcast kicker, queue toggle and insights control sit above it
 * (relative z-30) so they stay independently interactive.
 *
 * ## Why there is no hover reveal (#1583)
 *
 * This card previously carried TWO reveal mechanisms for the same content: a sparkle popover of
 * `summary_bullets`, and a whole-card hover overlay rendering the FULL `summary_text` while the
 * title, kicker, lede and meta all faded to `opacity-0`. Both are gone, and none of it should come
 * back, because:
 *
 * - the overlay rendered unbounded text in a fixed-height, `overflow-hidden` box, so long summaries
 *   were sliced mid-sentence with no ellipsis and no scroll — the "doesn't fit" complaint;
 * - it erased the card's own identity, leaving an anonymous pull-quote you couldn't attribute;
 * - `group-hover` is not a gesture on touch, the app's primary platform;
 * - with no hover intent, moving a pointer down a list strobed every card in turn;
 * - the two mechanisms gated differently (`has_gi && bullets.length` vs any summary text) and
 *   stacked, rendering the popover on top of the already-revealed overlay;
 * - `opacity-0` does not remove content from the accessibility tree, so every card read its whole
 *   summary to screen readers — 20 per catalogue page;
 * - in the queue, reaching for the reorder controls erased the title you were trying to move.
 *
 * The full prose lives on the player page (`KnowledgePanel`), which has room to scroll. Rule of
 * thumb: a list card shows a bounded preview and links out; it never hosts unbounded text.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import type { EpisodeSummary, FavoriteAdd } from '../services/types'
import { formatDuration, formatPublishDate } from '../utils/format'
import { episodeArtwork } from '../utils/episode'
import FavoriteButton from './FavoriteButton.vue'
import QueueButton from './QueueButton.vue'
import DownloadButton from './DownloadButton.vue'
import AddToCollectionButton from './AddToCollectionButton.vue'

const props = defineProps<{
  episode: EpisodeSummary
  /**
   * NARROW containers — the "More like this" rail (`w-56`, 224px) and the queue's recent list.
   *
   * The full card assumes a wide row: 128px of artwork plus a text column with room for a
   * multi-line title. In 224px the full card leaves ~80px of text, which is worse than what it
   * replaced. Compact keeps the identity (artwork, show, title) and drops the meta column.
   */
  compact?: boolean
}>()
const { t, locale } = useI18n()

const duration = computed(() => formatDuration(props.episode.duration_seconds))
const date = computed(() => formatPublishDate(props.episode.publish_date, locale.value))
const bullets = computed(() => props.episode.summary_bullets ?? [])
// The TRUE key-point count — `summary_bullets` is capped for card size, so its length pinned at 8
// for every richly-summarised episode ("every episode has 8 key points"). Fall back to the visible
// bullets when the server didn't send a count.
const keyPointCount = computed(() => props.episode.summary_bullet_count ?? bullets.value.length)

// Show the insights affordance only when there's grounded summary content to reveal.
/**
 * The badge counts KEY POINTS — `summary_bullets` — and now says so.
 *
 * It read "N insights" while counting bullets. Insights are a different thing: the timestamped
 * claims and observations in the Knowledge Panel, each anchored to a moment. The card cannot show a
 * true insight count — the server deliberately does not compute one per row, because it would cost
 * an artifact load per card (`schemas.py:104`) — so the honest fix is to name what is actually
 * being counted, using the same word the Insights panel uses for the same field.
 *
 * The `has_gi` gate went with it: that flag means the episode has generated insights, which is not
 * what this badge is about. Bullets come from the summary. If there are bullets, there is a count.
 */
const hasKeyPoints = computed(() => bullets.value.length > 0)
// Prefer our locally-stored copy (artwork_url); fall back to the remote feed image URLs.
const artwork = computed(() => episodeArtwork(props.episode))


const favItem = computed<FavoriteAdd>(() => ({
  kind: 'episode',
  ref: props.episode.slug,
  label: props.episode.title,
  sublabel: props.episode.podcast_title ?? undefined,
  slug: props.episode.slug,
}))
</script>

<template>
  <article
    data-testid="episode-card"
    class="group relative -mx-3 flex gap-4 rounded-xl border-b border-border px-3 py-5 transition-colors sm:gap-5"
  >
    <!--
      LEFT COLUMN: artwork, then the facts about the episode (#2004 item 4).

      The artwork used to sit alone at 80px (96 at `sm`) while the text column carried the show
      name, the title, the lede, the insights control AND the date/duration — five things against
      one. Moving the facts under the artwork uses space that was dead and gives the summary room
      to be read.
    -->
    <div class="flex shrink-0 flex-col gap-2">
    <img
      v-if="artwork"
      :src="artwork"
      :alt="episode.podcast_title ?? ''"
      loading="lazy"
      :class="compact ? 'h-20 w-20 rounded-lg bg-elevated object-cover' : 'h-32 w-32 rounded-lg bg-elevated object-cover'"
    />
      <!-- Without this the column has no fixed-width child and collapses, squeezing the facts
           beneath it. An episode with no artwork must still hold the same shape. -->
      <div
        v-else
        class="rounded-lg bg-elevated"
        :class="compact ? 'h-20 w-20' : 'h-32 w-32'"
        aria-hidden="true"
      />
      <div v-if="!compact && (date || duration)" class="flex items-center gap-1.5 text-xs font-medium text-muted">
        <span v-if="date">{{ date }}</span>
        <span v-if="date && duration" aria-hidden="true">·</span>
        <span v-if="duration">{{ duration }}</span>
      </div>
      <!-- A COUNT, not a toggle: the card does not render the bullets themselves, so there is
           nothing to expand. It stays because "how much is in here" is worth knowing at a glance —
           and it says KEY POINTS, which is what it counts. -->
      <div
        v-if="!compact && hasKeyPoints"
        data-testid="card-key-point-count"
        class="inline-flex w-fit items-center gap-1.5 rounded-full bg-overlay px-2.5 py-1 text-xs font-bold text-canvas-foreground"
      >
        <svg viewBox="0 0 24 24" class="h-3.5 w-3.5" fill="currentColor" aria-hidden="true">
          <path d="M12 2.5l1.9 4.6 4.6 1.9-4.6 1.9L12 15.5l-1.9-4.6L5.5 9l4.6-1.9L12 2.5z" />
        </svg>
        {{ t('card.keyPointCount', { count: keyPointCount }, keyPointCount) }}
      </div>
    </div>
    <div class="flex min-w-0 flex-1 flex-col">
      <!--
        The action row owns its own line, and the show name owns the next one (#2004 item 4).

        They used to share a flex row: the buttons `shrink-0`, the name `min-w-0`. So the name
        absorbed every pixel of squeeze and stacked vertically — "COMPLEX SYSTEMS WITH PATRICK
        MCKENZIE (PATIO11)" over six lines — and it was worst in the `w-56` "More like this" rail,
        where the same four buttons compete inside 224px.
      -->
      <div class="flex items-start justify-end gap-3">
        <div class="flex shrink-0 items-center gap-[12px]">
          <span
            v-if="episode.status !== 'ready'"
            class="relative z-30 rounded-full bg-overlay px-2 py-0.5 text-xs font-semibold text-warning"
          >
            {{ t('status.pending') }}
          </span>

          <FavoriteButton :item="favItem" class="z-30" />

          <QueueButton :slug="episode.slug" />

          <DownloadButton :slug="episode.slug" />

          <AddToCollectionButton :item="{ kind: 'episode', ref: episode.slug }" />

          <!-- Optional extra actions in the same icon row (e.g. the queue's reorder ↑/↓). -->
          <slot name="actions" />
        </div>
      </div>

      <!-- The show name, full width, with the whole column to wrap into. -->
      <RouterLink
        v-if="episode.podcast_title"
        :to="{ name: 'podcast', params: { feedId: episode.feed_id } }"
        class="lp-kicker relative z-30 mt-1 block no-underline"
      >
        {{ episode.podcast_title }}
      </RouterLink>

      <!-- Title (stretched link → Player). Never fades: card identity stays visible in every state. -->
      <RouterLink
        :to="{ name: 'player', params: { slug: episode.slug } }"
        class="mt-1 font-display text-lg font-bold leading-snug text-canvas-foreground no-underline transition-opacity duration-200 after:absolute after:inset-0 sm:text-xl"
      >
        {{ episode.title }}
      </RouterLink>

      <!-- Clean one-line lede (never the bullets jammed together) -->
      <p
        v-if="episode.summary_preview"
        class="mt-2 line-clamp-2 text-sm leading-relaxed text-muted"
      >
        {{ episode.summary_preview }}
      </p>

    </div>

  </article>
</template>
