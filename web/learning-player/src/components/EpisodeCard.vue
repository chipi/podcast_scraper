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
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import type { EpisodeSummary, FavoriteAdd } from '../services/types'
import { formatDuration, formatPublishDate } from '../utils/format'
import { episodeArtwork } from '../utils/episode'
import FavoriteButton from './FavoriteButton.vue'
import QueueButton from './QueueButton.vue'
import AddToCollectionButton from './AddToCollectionButton.vue'

const props = defineProps<{ episode: EpisodeSummary }>()
const { t, locale } = useI18n()

const duration = computed(() => formatDuration(props.episode.duration_seconds))
const date = computed(() => formatPublishDate(props.episode.publish_date, locale.value))
const bullets = computed(() => props.episode.summary_bullets ?? [])

/**
 * How many bullets the card shows when expanded, with the rest behind "Read full summary".
 *
 * Sized against PRODUCTION, not the fixtures — they differ enough to matter. Measured over 393
 * bullets from 50 live episodes (2026-08-13): median **207 chars**, p75 241, max 380, and **7.9
 * bullets per episode**. The synthetic corpora average 85 chars and 3 bullets, so anything sized
 * against them is ~2.4x too small per bullet and less than half the count.
 *
 * All 7.9 unclamped would put ~1,600 characters inside a list card — the same "doesn't fit" problem
 * the old whole-card overlay had, just opt-in. Four is roughly 20 lines on a phone: enough to be
 * genuinely useful, bounded enough to stay a card.
 */
const CARD_BULLETS = 4
const shownBullets = computed(() => bullets.value.slice(0, CARD_BULLETS))
// Show the insights affordance only when there's grounded summary content to reveal.
const hasInsights = computed(() => props.episode.has_gi && bullets.value.length > 0)
// Prefer our locally-stored copy (artwork_url); fall back to the remote feed image URLs.
const artwork = computed(() => episodeArtwork(props.episode))

const summaryOpen = ref(false)

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
    class="group relative -mx-3 flex gap-4 rounded-xl border-b border-border px-3 py-5 transition-colors sm:gap-5"
  >
    <img
      v-if="artwork"
      :src="artwork"
      :alt="episode.podcast_title ?? ''"
      loading="lazy"
      class="h-20 w-20 shrink-0 rounded-lg bg-elevated object-cover sm:h-24 sm:w-24"
    />
    <div class="flex min-w-0 flex-1 flex-col">
      <!-- Kicker row: podcast name (independent link) + status / insights / favorite / queue -->
      <div class="flex items-start justify-between gap-3">
        <RouterLink
          v-if="episode.podcast_title"
          :to="{ name: 'podcast', params: { feedId: episode.feed_id } }"
          class="lp-kicker relative z-30 inline-block min-w-0 no-underline"
        >
          {{ episode.podcast_title }}
        </RouterLink>
        <span v-else />
        <div class="flex shrink-0 items-center gap-2">
          <span
            v-if="episode.status !== 'ready'"
            class="relative z-30 rounded-full bg-overlay px-2 py-0.5 text-xs font-semibold text-warning"
          >
            {{ t('status.pending') }}
          </span>

          <FavoriteButton :item="favItem" class="relative z-30" />

          <QueueButton :slug="episode.slug" />

          <AddToCollectionButton :item="{ kind: 'episode', ref: episode.slug }" />

          <!-- Optional extra actions in the same icon row (e.g. the queue's reorder ↑/↓). -->
          <slot name="actions" />
        </div>
      </div>

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

      <!-- Insights: ONE affordance, identical on touch and pointer, expanding in flow.
           z-30 keeps it above the title's stretched-link ::after — without that, taps on the
           bullets click through and navigate to the player instead. -->
      <div v-if="hasInsights" class="relative z-30 self-start">
        <button
          type="button"
          class="mt-2 inline-flex items-center gap-1.5 rounded-full bg-overlay px-2.5 py-1 text-xs font-bold text-accent transition hover:bg-elevated"
          :aria-expanded="summaryOpen"
          :aria-controls="`insights-${episode.slug}`"
          @click="summaryOpen = !summaryOpen"
        >
          <svg viewBox="0 0 24 24" class="h-3.5 w-3.5" fill="currentColor" aria-hidden="true">
            <path d="M12 2.5l1.9 4.6 4.6 1.9-4.6 1.9L12 15.5l-1.9-4.6L5.5 9l4.6-1.9L12 2.5z" />
          </svg>
          {{ t('card.insightCount', { count: bullets.length }, bullets.length) }}
          <span class="text-[0.6rem] transition-transform" :class="summaryOpen ? 'rotate-180' : ''" aria-hidden="true">▼</span>
        </button>

        <!-- v-if, not v-show: opacity/display-only hiding leaves the text in the accessibility
             tree, so every collapsed card would read its full summary to a screen reader. -->
        <div
          v-if="summaryOpen"
          :id="`insights-${episode.slug}`"
          class="mt-2 border-t border-border pt-2"
        >
          <ul class="space-y-2">
            <li
              v-for="(b, i) in shownBullets"
              :key="i"
              class="flex gap-2 text-sm leading-relaxed text-surface-foreground"
            >
              <span class="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-grounded" aria-hidden="true" />
              <!-- No clamp. The user asked for these; truncating them mid-claim is the failure the
                   old overlay made. Length is bounded by CARD_BULLETS instead. -->
              <span>{{ b }}</span>
            </li>
          </ul>
          <RouterLink
            :to="{ name: 'player', params: { slug: episode.slug } }"
            class="mt-2 inline-block text-xs font-bold text-accent no-underline"
          >
            {{
              bullets.length > shownBullets.length
                ? t('card.moreInsights', { count: bullets.length - shownBullets.length })
                : t('card.readFullSummary')
            }}
          </RouterLink>
        </div>
      </div>

      <!-- Meta line: date · duration -->
      <div
        v-if="date || duration"
        class="mt-3 flex items-center gap-2 text-xs font-medium text-muted"
      >
        <span v-if="date">{{ date }}</span>
        <span v-if="date && duration" aria-hidden="true">·</span>
        <span v-if="duration">{{ duration }}</span>
      </div>

    </div>

  </article>
</template>
