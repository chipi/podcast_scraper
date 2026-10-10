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
 * The full prose also lives on the player page (`KnowledgePanel`), which has room to scroll.
 *
 * ## Read more (BE.2) — an explicit toggle, NOT the removed hover overlay
 *
 * The card now shows the episode's full `description` (the publisher's, not our summary — operator
 * 2026-10-05), clamped to the artwork's height, with a "Read more" toggle
 * that expands it in place. This is not a return of the hover overlay: it is a tap (works on touch),
 * it clamps with `line-clamp` (no fixed-height `overflow-hidden` slicing), it never fades the card's
 * identity to `opacity-0`, and the text is always in the a11y tree. The rule of thumb is refined: a
 * list card shows a bounded preview by default and reveals the rest on an explicit, reversible tap.
 */
import { computed, ref } from "vue"
import { useClampedProse } from "../composables/useClampedProse"
import { useI18n } from "vue-i18n"
import { RouterLink } from "vue-router"
import type { EpisodeSummary } from "../services/types"
import { formatDuration, formatPublishDate } from "../utils/format"
import { borderClass } from "../utils/highlightColors"
import { episodeArtwork } from "../utils/episode"
import EpisodeActions from "./EpisodeActions.vue"
import MomentsLink from "./MomentsLink.vue"
import LanguageBadge from "./LanguageBadge.vue"
import { useCorpusLanguages } from "../composables/useCorpusLanguages"
import PlayedBadge from "./PlayedBadge.vue"

const props = defineProps<{
  episode: EpisodeSummary
  /** Show "▶ Moments" on the show line — browsing surfaces only (Discover, show pages). */
  momentsLink?: boolean
  /**
   * NARROW containers — the "More like this" rail (`w-56`, 224px) and the queue's recent list.
   *
   * The full card assumes a wide row: 128px of artwork plus a text column with room for a
   * multi-line title. In 224px the full card leaves ~80px of text, which is worse than what it
   * replaced. Compact keeps the identity (artwork, show, title) and drops the meta column.
   */
  compact?: boolean
  /**
   * Forwarded to `EpisodeActions`: drop the heart from the visible row and put it in the ⋯ instead.
   * Set by the Saved list, where every row is favourited by definition — see EpisodeActions.
   */
  hideFavorite?: boolean
  /** Forwarded to `EpisodeActions`: demote the queue toggle into the ⋯ (Recently played). */
  hideQueue?: boolean
  /** Forwarded to `EpisodeActions`: promote download into the visible row (Up next). */
  showDownload?: boolean
  /**
   * Drop the played marker.
   *
   * For a list that is played BY CONSTRUCTION — the queue panel's Recently played — where the badge
   * only restates the section heading while costing a line under the artwork (operator 2026-09-23).
   * The same rule as `hideFavorite` on Saved: a surface does not label every row with the thing the
   * surface already is.
   */
  hidePlayed?: boolean
  /**
   * Drop the action row entirely — the card is an IDENTITY HEADER, not something to act on.
   *
   * Set when the card labels a group whose rows carry their own controls (Revisit: every moment has
   * jump + mark-reviewed), where a favourite/queue/⋯ cluster on the header is a third set of
   * controls competing with them — and at compact width it wraps onto its own line, which is what
   * made the header thick (operator 2026-09-17: "remove actions").
   */
  hideActions?: boolean
  /**
   * Tighter vertical rhythm for a card used as a header: less padding, and no bottom rule, since the
   * block around it draws its own divider (operator 2026-09-17: "can we get it shorter").
   */
  dense?: boolean
}>()
const { t, locale } = useI18n()

const duration = computed(() => formatDuration(props.episode.duration_seconds))
const date = computed(() => formatPublishDate(props.episode.publish_date, locale.value))
// The key-points badge is GONE (operator 2026-09-17), not merely hidden on phones — it had been
// `hidden sm:inline-flex`, which kept it on every desktop browser long after the feature left the
// product. The card's facts are now date and duration.
// Prefer our locally-stored copy (artwork_url); fall back to the remote feed image URLs.
const artwork = computed(() => episodeArtwork(props.episode))

// Read more/less (BE.2): collapsed shows the full summary clamped to a few lines; expanded shows
// all of it in the row. Only offered when there's full prose beyond the one-line lede.
const summaryExpanded = ref(false)
// Measured, not a line count: the summary fills the artwork column's height, so whether it is
// actually cut off depends on the rendered width and the neighbouring column. Defaults TRUE — in
// jsdom and before first paint both heights read 0, and treating that as "fits" would HIDE a
// "Read more" the text needs.
const summaryEl = ref<HTMLElement | null>(null)
// The publisher's OWN description, never our summary (operator 2026-10-05): a list card says what
// the episode is in the show's words, like the show row does; our summary lives in the episode
// notes. No fallback to the summary — an episode without a description shows no prose.
const summaryFull = computed(() => props.episode.description?.trim() || "")

// The measurement lives in `useClampedProse` — shared with ShowRow, PersonCardContent and
// PodcastView, which all had hand-copied versions that drifted apart (see the composable).
const { clipped: summaryClipped } = useClampedProse(summaryEl, summaryExpanded)

// "Read more" only when the text is ACTUALLY cut off — the summary now fills the artwork column
// rather than a fixed line count, so on a short summary nothing is clipped and the toggle would be
// offering to reveal nothing.
//
// Gated on `summaryFull`, which is WHAT THE WINDOW RENDERS. It used to gate on `summary_text` while
// the window fell back to `summary_preview`, so an episode carrying only a preview could render
// prose the window genuinely clipped with no toggle able to appear — text cut off and no way to
// reach it (operator 2026-09-17). The condition must read the same value as the element it governs.
const canExpandSummary = computed(
  () => !!summaryFull.value.trim() && (summaryClipped.value || summaryExpanded.value)
)
// Containers whose only content may be the badge gate on what the badge itself shows.
const { badgeShown } = useCorpusLanguages()
</script>

<template>
  <article
    data-testid="episode-card"
    class="lp-media-row group relative -mx-3 gap-4 rounded-xl px-3 transition-colors sm:gap-5"
    :class="[
      dense ? 'py-2' : 'border-b border-border py-5',
      episode.color ? ['border-l-4', borderClass(episode.color)] : '',
    ]"
  >
    <!--
      LEFT COLUMN: artwork, then the facts about the episode (#2004 item 4).

      The artwork used to sit alone at 80px (96 at `sm`) while the text column carried the show
      name, the title, the lede, the insights control AND the date/duration — five things against
      one. Moving the facts under the artwork uses space that was dead and gives the summary room
      to be read.
    -->
    <div class="lp-media-aside">
      <img
        v-if="artwork"
        :src="artwork"
        :alt="episode.podcast_title ?? ''"
        loading="lazy"
        :class="
          compact
            ? 'h-20 w-20 rounded-lg bg-elevated object-cover'
            : 'h-32 w-32 rounded-lg bg-elevated object-cover'
        "
      />
      <!-- Without this the column has no fixed-width child and collapses, squeezing the facts
           beneath it. An episode with no artwork must still hold the same shape. -->
      <div
        v-else
        class="rounded-lg bg-elevated"
        :class="compact ? 'h-20 w-20' : 'h-32 w-32'"
        aria-hidden="true"
      />
      <!-- Facts directly under the artwork (operator: date/duration UP). -->
      <div
        v-if="!compact && (date || duration || badgeShown(episode.language))"
        class="flex items-center gap-1.5 text-xs font-medium text-muted"
      >
        <span v-if="date">{{ date }}</span>
        <span v-if="date && duration" aria-hidden="true">·</span>
        <span v-if="duration">{{ duration }}</span>
        <LanguageBadge :lang="episode.language" />
      </div>
      <!-- Compact drops the date row, but not the language: it is what the listener reads in. -->
      <LanguageBadge v-if="compact" :lang="episode.language" />
      <span
        v-if="episode.status !== 'ready'"
        class="w-fit rounded-full bg-overlay px-2 py-0.5 text-xs font-semibold text-warning"
      >
        {{ t("status.pending") }}
      </span>
      <!-- Shown in COMPACT too, unlike the date/duration row above it. Compact is what the queue's
           "recently played" renders, and a list of things you have heard is the one place the
           marker is load-bearing rather than incidental. -->
      <PlayedBadge v-if="!hidePlayed" :slug="episode.slug" />
      <!-- Surface-specific fact under the artwork, beside the date — Search puts its match count
           here. Empty everywhere else, so no other caller changes. -->
      <div v-if="$slots.aside" class="text-xs font-semibold text-muted">
        <slot name="aside" />
      </div>
      <!-- The shared EpisodeActions row (UXS-014: nobody rolls their own), directly UNDER the
           artwork it acts on — `w-32` matches the artwork's width. It used to be `mt-auto`, footed
           against the bottom of a stretched column, which left it floating below a gap whenever the
           summary was the taller side (operator 2026-09-17). The row is favourite + queue + ⋯
           (download + collect live in the ⋯), which is 120px and fits the 128px column in ONE row.
           Compact card (queue "recently played"): `w-20` matches the 80px artwork, too narrow for
           three targets, so EpisodeActions' retained `flex-wrap` folds the ⋯ under favourite+queue
           rather than widening the column past the artwork and eating the text.
           `relative z-30` keeps it tappable above the title's stretched card-link overlay; the
           queue's reorder ↑/↓ ride the slot. -->
      <EpisodeActions
        v-if="!hideActions"
        :slug="episode.slug"
        :moments="episode.has_gi"
        :hide-favorite="hideFavorite"
        :hide-queue="hideQueue"
        :show-download="showDownload"
        :class="compact ? 'relative z-30 mt-2 w-20' : 'relative z-30 w-32'"
      >
        <template #lead><slot name="lead-action" /></template>
        <slot name="actions" />
      </EpisodeActions>
    </div>
    <!-- RIGHT COLUMN: show name, title, summary — full width. The actions moved UNDER the artwork
         (left column), so nothing competes with the text here. NOT `relative` — the title's
         stretched ::after link stays anchored to the whole `article` so the artwork plays on tap. -->
    <!-- The summary FILLS the height the artwork column sets, then clips (operator 2026-09-17).
         A fixed `line-clamp-4` stopped the text short of the artwork's bottom, leaving dead space
         beside the picture and a line of description the card had room for but did not show. Both
         columns foot their last element with `mt-auto`, so the action row and "Read more" land on
         the same line. -->
    <div class="lp-media-body">
      <!-- Show name — full column width; only ellipsizes when genuinely long.
           A LINK only when there is a feed to link to. `feed_id` is optional on EpisodeSummary, and
           `router.resolve({ name: 'podcast', params: { feedId: undefined } })` THROWS rather than
           degrading — so a row whose source carries the show's NAME but not its id (Search groups
           its hits by episode, and the feed id lives in the hit metadata) took the whole view down.
           Plain text is the honest fallback: the name is still information without being a
           destination. -->
      <!-- The show line also carries "▶ Moments" on browsing surfaces (Discover, show pages;
           operator 2026-10-10): on a row that already exists, so the card grows no taller. -->
      <div class="flex items-center justify-between gap-2">
        <RouterLink
          v-if="episode.podcast_title && episode.feed_id"
          :to="{ name: 'podcast', params: { feedId: episode.feed_id } }"
          class="lp-kicker relative z-30 block min-w-0 truncate no-underline"
          :title="episode.podcast_title"
        >
          {{ episode.podcast_title }}
        </RouterLink>
        <span v-else-if="episode.podcast_title" class="lp-kicker block min-w-0 truncate" :title="episode.podcast_title">
          {{ episode.podcast_title }}
        </span>
        <span v-else />
        <MomentsLink v-if="momentsLink && episode.has_gi" :slug="episode.slug" />
      </div>

      <!-- Title (stretched link → Player). Never fades: card identity stays visible in every state. -->
      <RouterLink
        :to="{ name: 'player', params: { slug: episode.slug } }"
        class="mt-0.5 block font-display text-lg font-bold leading-snug text-canvas-foreground no-underline transition-opacity duration-200 after:absolute after:inset-0 sm:text-xl"
      >
        {{ episode.title }}
      </RouterLink>

      <!-- Surface-specific line between the title and the summary — Search puts WHY this episode
           matched here ("Matched: Transcript · Insight"), which belongs with the identity rather
           than after the prose it explains. Empty everywhere else. -->
      <div v-if="$slots.meta" class="relative z-30 mt-0.5">
        <slot name="meta" />
      </div>

      <!-- Description: the publisher's full prose, clipped to whatever the artwork column leaves
           and expanded in place by "Read more" (BE.2). The window (`lp-media-clip`) is what
           constrains it — see style.css; the <p> alone could only ever grow the row. -->
      <div
        v-if="summaryFull"
        ref="summaryEl"
        class="lp-media-clip mt-2"
        :class="summaryExpanded ? 'lp-media-clip--open' : ''"
      >
        <p class="text-sm leading-relaxed text-muted">{{ summaryFull }}</p>
      </div>
      <!-- `relative z-30` so the toggle sits above the title's stretched card-link overlay. -->
      <!-- Offered on COMPACT too (operator 2026-09-23). Recently played renders compact cards, and
           withholding the toggle there left prose visibly cut off with no way to reach the rest —
           the same complaint that put "Read more" on the full card in the first place. -->
      <button
        v-if="canExpandSummary"
        type="button"
        class="lp-media-foot relative z-30 mt-1 w-fit text-xs font-bold text-accent transition hover:opacity-80"
        data-testid="card-read-more"
        :aria-expanded="summaryExpanded"
        @click="summaryExpanded = !summaryExpanded"
      >
        {{ summaryExpanded ? t("card.readLess") : t("card.readMore") }}
      </button>
    </div>
  </article>
</template>
