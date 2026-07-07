<script setup lang="ts">
/**
 * Editorial-bold episode card (UXS-011 / PRD-038 FR3–FR4). Stays compact and readable: a clean
 * one-line lede — never the bullets jammed together — with the FULL summary one tap/hover away
 * behind a grounded "insights" affordance (popover on desktop hover/focus, tap-toggle on touch).
 *
 * Uses the "stretched link" pattern (no nested anchors): the title link's ::after overlay covers
 * the whole card → Player; the podcast kicker, queue toggle and insights control sit above it
 * (relative z-10/20) so they stay independently interactive.
 */
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink } from 'vue-router'
import type { EpisodeSummary, FavoriteAdd } from '../services/types'
import { formatDuration, formatPublishDate } from '../utils/format'
import { episodeArtwork } from '../utils/episode'
import FavoriteButton from './FavoriteButton.vue'
import QueueButton from './QueueButton.vue'

const props = defineProps<{ episode: EpisodeSummary }>()
const { t, locale } = useI18n()

const duration = computed(() => formatDuration(props.episode.duration_seconds))
const date = computed(() => formatPublishDate(props.episode.publish_date, locale.value))
const bullets = computed(() => props.episode.summary_bullets ?? [])
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
          class="lp-kicker relative z-30 inline-block min-w-0 no-underline transition-opacity duration-200 group-hover:opacity-0"
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

          <!-- Insights: its own hover/focus affordance → full-summary popover. Sibling icons stay
               visible; the wrapper lifts to z-50 only while open so the popover clears other cards. -->
          <div
            v-if="hasInsights"
            class="relative transition-opacity sm:opacity-0 sm:focus-within:opacity-100 sm:group-hover:opacity-100"
            :class="summaryOpen ? 'z-50 sm:opacity-100' : 'z-30'"
            @mouseenter="summaryOpen = true"
            @mouseleave="summaryOpen = false"
            @focusin="summaryOpen = true"
            @focusout="summaryOpen = false"
          >
            <button
              type="button"
              class="flex h-7 w-7 items-center justify-center rounded-full text-muted transition hover:bg-overlay hover:text-accent"
              :class="summaryOpen ? 'bg-overlay text-accent' : ''"
              :aria-label="t('card.insights')"
              :aria-expanded="summaryOpen"
              :aria-controls="`insights-pop-${episode.slug}`"
              @click="summaryOpen = !summaryOpen"
            >
              <svg viewBox="0 0 24 24" class="h-5 w-5" fill="currentColor" aria-hidden="true">
                <path d="M12 2.5l1.9 4.6 4.6 1.9-4.6 1.9L12 15.5l-1.9-4.6L5.5 9l4.6-1.9L12 2.5z" />
                <path d="M19 14.5l.95 2.3 2.3.95-2.3.95L19 21l-.95-2.3L15.75 18l2.3-.95L19 14.5z" opacity=".75" />
              </svg>
            </button>
            <transition name="lp-pop">
              <div
                v-show="summaryOpen"
                :id="`insights-pop-${episode.slug}`"
                role="group"
                :aria-label="t('card.insights')"
                class="absolute right-0 top-9 z-50 w-72 max-w-[80vw] rounded-xl border border-border bg-elevated p-4 text-left shadow-2xl"
              >
                <p class="lp-kicker">{{ t('card.insights') }}</p>
                <ul class="mt-2 space-y-2">
                  <li
                    v-for="(b, i) in bullets"
                    :key="i"
                    class="flex gap-2 text-sm leading-relaxed text-surface-foreground"
                  >
                    <span class="mt-2 h-1 w-1 shrink-0 rounded-full bg-accent" aria-hidden="true" />
                    <span>{{ b }}</span>
                  </li>
                </ul>
              </div>
            </transition>
          </div>

          <FavoriteButton :item="favItem" class="relative z-30" />

          <QueueButton :slug="episode.slug" />

          <!-- Optional extra actions in the same icon row (e.g. the queue's reorder ↑/↓). -->
          <slot name="actions" />
        </div>
      </div>

      <!-- Title (stretched link → Player). Fades out on hover so the summary overlay stands alone. -->
      <RouterLink
        :to="{ name: 'player', params: { slug: episode.slug } }"
        class="mt-1 font-display text-lg font-bold leading-snug text-canvas-foreground no-underline transition-opacity duration-200 after:absolute after:inset-0 group-hover:opacity-0 sm:text-xl"
      >
        {{ episode.title }}
      </RouterLink>

      <!-- Clean one-line lede (never the bullets jammed together) -->
      <p
        v-if="episode.summary_preview"
        class="mt-2 line-clamp-2 text-sm leading-relaxed text-muted transition-opacity duration-200 group-hover:opacity-0"
      >
        {{ episode.summary_preview }}
      </p>

      <!-- Meta line: date · duration -->
      <div
        v-if="date || duration"
        class="mt-3 flex items-center gap-2 text-xs font-medium text-muted transition-opacity duration-200 group-hover:opacity-0"
      >
        <span v-if="date">{{ date }}</span>
        <span v-if="date && duration" aria-hidden="true">·</span>
        <span v-if="duration">{{ duration }}</span>
      </div>

    </div>

    <!-- Whole-card hover (UXS-014): the episode summary as an editorial pull-quote across the ENTIRE
         card. Click-through; the top-row affordances (z-30) float above it; card still opens player. -->
    <div
      v-if="episode.summary_text || episode.summary_preview"
      class="pointer-events-none absolute inset-0 z-20 flex items-center overflow-hidden rounded-xl bg-gradient-to-br from-elevated via-canvas to-elevated px-5 py-4 opacity-0 shadow-2xl transition-opacity duration-200 group-hover:opacity-100 group-focus-within:opacity-100"
    >
      <p class="border-l-2 border-accent pl-4 font-display text-base font-semibold leading-snug text-canvas-foreground sm:text-lg">
        {{ episode.summary_text || episode.summary_preview }}
      </p>
    </div>
  </article>
</template>

<style scoped>
.lp-pop-enter-active,
.lp-pop-leave-active {
  transition:
    opacity 0.12s ease,
    transform 0.12s ease;
}
.lp-pop-enter-from,
.lp-pop-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}
</style>
