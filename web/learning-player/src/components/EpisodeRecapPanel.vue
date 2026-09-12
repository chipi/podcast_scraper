<script setup lang="ts">
/**
 * Post-episode recap panel (RFC-122 / #2038) — the reinforcement surface.
 *
 * When an episode finishes (the player store's `justFinished` fires), this replaces the transport
 * IN PLACE on the episode page: what the listener just heard, consolidated. It renders one recap
 * model (summary key points + top insights + the single strongest attributed quote — the anchor)
 * plus a "listen more like this" mini-grid that reuses the existing related-episodes rail. The
 * same model will feed the daily digest email (#2039), so the two surfaces cannot drift.
 *
 * Bridge-only (PRD-035 Principle 4): everything here is transcript-derived text + KG metadata +
 * artwork; the panel never touches audio.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import CardRail from './CardRail.vue'
import EpisodeTile from './EpisodeTile.vue'
import type { EpisodeRecap, EpisodeSummary } from '../services/types'

const props = withDefaults(
  defineProps<{ recap: EpisodeRecap; related?: EpisodeSummary[] }>(),
  { related: () => [] },
)
const emit = defineEmits<{ (e: 'dismiss'): void }>()

const { t } = useI18n()

// Bullets are the gist; when an episode has none, the prose summary is the fallback lede (one item)
// so the panel always leads with something to consolidate rather than an empty heading.
const keyPoints = computed<string[]>(() => {
  if (props.recap.key_points.length) return props.recap.key_points
  const prose = props.recap.summary_text?.trim()
  return prose ? [prose] : []
})
const quote = computed(() => props.recap.signature_quote)
const insights = computed(() => props.recap.insights)
</script>

<template>
  <section
    class="flex max-h-[78dvh] flex-col overflow-hidden rounded-2xl border border-border bg-surface"
    :aria-label="t('player.recapRegion')"
    data-testid="episode-recap-panel"
  >
    <!-- Header: what just happened + the "we took notes" reassurance, with a dismiss back to the
         finished player pinned to the right. -->
    <header class="flex items-start justify-between gap-3 border-b border-border p-4">
      <div class="min-w-0">
        <p class="lp-kicker text-accent">{{ t('player.recapKicker') }}</p>
        <h2 class="mt-0.5 truncate font-display text-lg font-bold text-canvas-foreground">
          {{ recap.title }}
        </h2>
        <p class="mt-1 flex items-center gap-1.5 text-sm text-muted">
          <svg
            viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
            stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4 shrink-0" aria-hidden="true"
          >
            <path d="M4 4h11l5 5v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1Z" />
            <path d="M14 4v5h5M8 13h8M8 17h5" />
          </svg>
          {{ t('player.recapNotes') }}
        </p>
      </div>
      <button
        type="button"
        class="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-border text-muted transition hover:bg-overlay hover:text-canvas-foreground"
        :aria-label="t('player.recapDismiss')"
        :title="t('player.recapDismiss')"
        data-testid="recap-dismiss"
        @click="emit('dismiss')"
      >
        <svg
          viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
          stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" aria-hidden="true"
        >
          <path d="M18 6 6 18M6 6l12 12" />
        </svg>
      </button>
    </header>

    <div class="min-h-0 flex-1 space-y-5 overflow-y-auto p-4">
      <!-- Key points — the gist to consolidate. -->
      <section v-if="keyPoints.length" data-testid="recap-key-points">
        <h3 class="lp-kicker mb-2 text-muted">{{ t('player.recapKeyPoints') }}</h3>
        <ul class="space-y-1.5">
          <li
            v-for="(point, i) in keyPoints"
            :key="i"
            class="flex gap-2 text-sm leading-relaxed text-canvas-foreground"
          >
            <span class="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" aria-hidden="true" />
            <span>{{ point }}</span>
          </li>
        </ul>
      </section>

      <!-- The anchor: one memorable line, attributed when the graph can name the speaker. -->
      <blockquote
        v-if="quote"
        class="border-l-2 border-accent pl-3"
        data-testid="recap-quote"
      >
        <p class="font-display text-base italic leading-snug text-canvas-foreground">
          “{{ quote.text }}”
        </p>
        <footer v-if="quote.speaker" class="mt-1 text-sm text-muted">
          {{ t('player.recapQuoteBy', { speaker: quote.speaker }) }}
        </footer>
      </blockquote>

      <!-- Top insights (salience-ranked, capped server-side). -->
      <section v-if="insights.length" data-testid="recap-insights">
        <h3 class="lp-kicker mb-2 text-muted">{{ t('player.recapInsights') }}</h3>
        <ul class="space-y-2">
          <li
            v-for="ins in insights"
            :key="ins.id"
            class="text-sm leading-relaxed text-canvas-foreground"
          >
            {{ ins.text }}
          </li>
        </ul>
      </section>

      <!-- Listen more like this — reuses the related-episodes rail. Hidden when empty. -->
      <section v-if="related.length" data-testid="recap-more-like-this">
        <h3 class="lp-kicker mb-2 text-muted">{{ t('player.recapMoreLikeThis') }}</h3>
        <CardRail>
          <li v-for="ep in related" :key="ep.slug" class="w-36 shrink-0 sm:w-40">
            <EpisodeTile :episode="ep" />
          </li>
        </CardRail>
      </section>
    </div>

    <footer class="flex items-center border-t border-border p-3">
      <button
        type="button"
        class="ml-auto rounded-full bg-accent px-4 py-2 text-sm font-semibold text-accent-foreground transition hover:opacity-90"
        data-testid="recap-back"
        @click="emit('dismiss')"
      >
        {{ t('player.recapDismiss') }}
      </button>
    </footer>
  </section>
</template>
