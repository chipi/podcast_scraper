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
import type { EpisodeSummary } from '../services/types'
import { useAuthStore } from '../stores/auth'
import { useQueueStore } from '../stores/queue'
import { formatDuration, formatPublishDate } from '../utils/format'

const props = defineProps<{ episode: EpisodeSummary }>()
const { t, locale } = useI18n()
const auth = useAuthStore()
const queue = useQueueStore()

const duration = computed(() => formatDuration(props.episode.duration_seconds))
const date = computed(() => formatPublishDate(props.episode.publish_date, locale.value))
const bullets = computed(() => props.episode.summary_bullets ?? [])
// Show the insights affordance only when there's grounded summary content to reveal.
const hasInsights = computed(() => props.episode.has_gi && bullets.value.length > 0)
// Prefer our locally-stored copy (artwork_url); fall back to the remote feed image URLs.
const artwork = computed(
  () =>
    props.episode.artwork_url ||
    props.episode.episode_image_url ||
    props.episode.feed_image_url,
)

const summaryOpen = ref(false)
</script>

<template>
  <article
    class="group relative -mx-3 flex gap-4 rounded-xl border-b border-border px-3 py-5 transition-colors hover:bg-overlay sm:gap-5"
  >
    <img
      v-if="artwork"
      :src="artwork"
      :alt="episode.podcast_title ?? ''"
      loading="lazy"
      class="h-20 w-20 shrink-0 rounded-lg bg-elevated object-cover sm:h-24 sm:w-24"
    />
    <div class="flex min-w-0 flex-1 flex-col">
      <!-- Kicker row: podcast name (independent link) + status / insights / queue affordances -->
      <div class="flex items-start justify-between gap-3">
        <RouterLink
          v-if="episode.podcast_title"
          :to="{ name: 'podcast', params: { feedId: episode.feed_id } }"
          class="lp-kicker relative z-10 inline-block min-w-0 truncate no-underline"
        >
          {{ episode.podcast_title }}
        </RouterLink>
        <span v-else />
        <div class="flex shrink-0 items-center gap-2">
          <span
            v-if="episode.status !== 'ready'"
            class="rounded-full bg-overlay px-2 py-0.5 text-xs font-semibold text-warning"
          >
            {{ t('status.pending') }}
          </span>

          <!-- Insights: cool icon → full-summary popover (hover/focus on desktop, tap on touch).
               Hidden at rest on desktop so the row stays summary-clean; always shown on touch. -->
          <div
            v-if="hasInsights"
            class="relative z-20 transition-opacity sm:opacity-0 sm:focus-within:opacity-100 sm:group-hover:opacity-100"
            :class="summaryOpen ? 'sm:opacity-100' : ''"
            @mouseenter="summaryOpen = true"
            @mouseleave="summaryOpen = false"
            @focusin="summaryOpen = true"
            @focusout="summaryOpen = false"
          >
            <button
              type="button"
              class="flex h-7 w-7 items-center justify-center rounded-full text-grounded transition hover:bg-overlay"
              :class="summaryOpen ? 'bg-overlay' : ''"
              :aria-label="t('card.insights')"
              :aria-expanded="summaryOpen"
              @click="summaryOpen = !summaryOpen"
            >
              <svg viewBox="0 0 24 24" class="h-5 w-5" fill="currentColor" aria-hidden="true">
                <path d="M12 2.5l1.9 4.6 4.6 1.9-4.6 1.9L12 15.5l-1.9-4.6L5.5 9l4.6-1.9L12 2.5z" />
                <path d="M19 14.5l.95 2.3 2.3.95-2.3.95L19 21l-.95-2.3L15.75 18l2.3-.95L19 14.5z" opacity=".75" />
              </svg>
            </button>
            <!-- Popover: the full grounded summary -->
            <transition name="lp-pop">
              <div
                v-show="summaryOpen"
                role="dialog"
                :aria-label="t('card.insights')"
                class="absolute right-0 top-9 z-30 w-72 max-w-[80vw] rounded-xl border border-border bg-elevated p-4 text-left shadow-2xl"
              >
                <p class="lp-kicker text-grounded">{{ t('card.insights') }}</p>
                <ul class="mt-2 space-y-2">
                  <li
                    v-for="(b, i) in bullets"
                    :key="i"
                    class="flex gap-2 text-sm leading-relaxed text-surface-foreground"
                  >
                    <span class="mt-2 h-1 w-1 shrink-0 rounded-full bg-grounded" aria-hidden="true" />
                    <span>{{ b }}</span>
                  </li>
                </ul>
              </div>
            </transition>
          </div>

          <button
            v-if="auth.isAuthenticated"
            type="button"
            class="relative z-10 flex h-7 w-7 items-center justify-center rounded-full border border-border text-sm leading-none"
            :class="queue.has(episode.slug) ? 'border-accent text-accent' : 'text-muted hover:text-canvas-foreground'"
            :aria-pressed="queue.has(episode.slug)"
            :aria-label="queue.has(episode.slug) ? t('queue.remove') : t('queue.add')"
            @click="queue.toggle(episode.slug)"
          >
            {{ queue.has(episode.slug) ? '✓' : '+' }}
          </button>
        </div>
      </div>

      <!-- Title (stretched link → Player) -->
      <RouterLink
        :to="{ name: 'player', params: { slug: episode.slug } }"
        class="mt-1 font-display text-lg font-bold leading-snug text-canvas-foreground no-underline after:absolute after:inset-0 sm:text-xl"
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
