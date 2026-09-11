<script setup lang="ts">
/**
 * Loading skeleton for the player (F1.3) — reserves the real shape of the episode surface so a cold,
 * UNCACHED load fills in place instead of shoving the page down: kicker + title, the square hero
 * artwork, the transport/controls bar, and the first transcript lines. Mirrors the vertical rhythm
 * of PlayerView's left rail (mt-3, aspect-square artwork, sticky controls). Decorative — every block
 * is `aria-hidden`; one sr-only live line announces the load.
 *
 * A cached episode never sees this: PlayerView paints from its device snapshot instantly. This is
 * only the first-ever / offline-uncached path.
 */
import { useI18n } from 'vue-i18n'
const { t } = useI18n()
</script>

<template>
  <div class="mt-3" :aria-busy="true" data-testid="player-skeleton">
    <span class="sr-only">{{ t('section.loading') }}</span>

    <!-- kicker + title -->
    <div class="h-3 w-24 animate-pulse rounded bg-elevated" aria-hidden="true" />
    <div class="mt-2 h-7 w-3/4 animate-pulse rounded bg-elevated sm:h-9" aria-hidden="true" />
    <div class="mt-2 h-4 w-1/3 animate-pulse rounded bg-elevated" aria-hidden="true" />

    <!-- hero artwork (same aspect-square footprint as the real image) -->
    <div
      class="mt-3 aspect-square w-full animate-pulse rounded-2xl border border-border bg-elevated"
      aria-hidden="true"
    />

    <!-- transport: scrubber + control buttons -->
    <div class="mt-4 space-y-3" aria-hidden="true">
      <div class="h-1.5 w-full animate-pulse rounded-full bg-elevated" />
      <div class="flex items-center justify-center gap-4">
        <div class="h-10 w-10 animate-pulse rounded-full bg-elevated" />
        <div class="h-14 w-14 animate-pulse rounded-full bg-elevated" />
        <div class="h-10 w-10 animate-pulse rounded-full bg-elevated" />
      </div>
    </div>

    <!-- first transcript lines -->
    <div class="mt-6 space-y-2" aria-hidden="true">
      <div class="h-4 w-full animate-pulse rounded bg-elevated" />
      <div class="h-4 w-11/12 animate-pulse rounded bg-elevated" />
      <div class="h-4 w-4/5 animate-pulse rounded bg-elevated" />
    </div>
  </div>
</template>
