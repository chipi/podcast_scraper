<script setup lang="ts">
/**
 * Persistent mini-player (#1587).
 *
 * The visible half of moving audio ownership into the store. Audio now survives navigation, so
 * something must show what is playing and offer a way back — otherwise playback continues with no
 * evidence of it, which is worse than stopping.
 *
 * Hidden on the player page itself (the full transport is right there) and whenever nothing is
 * loaded. On mobile it sits directly ABOVE the bottom tab bar; on desktop it pins to the bottom.
 *
 * Opaque background, same reason as BottomNav: a translucent fixed bar composites its text against
 * whatever is scrolled behind it, so its contrast ratio — and therefore WCAG conformance — varies
 * with page content. Pinned by `spec-conformance.test.ts`.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { storeToRefs } from 'pinia'
import { RouterLink, useRoute } from 'vue-router'
import { usePlayerStore } from '../stores/player'

const { t } = useI18n()
const route = useRoute()
const player = usePlayerStore()
const { playing, currentTime, duration, currentSlug, currentTitle, currentArtwork } =
  storeToRefs(player)

const onPlayerPage = computed(() => route.name === 'player' && route.params.slug === currentSlug.value)
const visible = computed(() => !!currentSlug.value && !onPlayerPage.value)
const progress = computed(() =>
  duration.value > 0 ? Math.min(100, (currentTime.value / duration.value) * 100) : 0,
)
</script>

<template>
  <div
    v-if="visible"
    data-testid="mini-player"
    class="fixed inset-x-0 bottom-[calc(3.25rem+env(safe-area-inset-bottom))] z-40 border-t border-border bg-elevated sm:bottom-0 sm:pb-[env(safe-area-inset-bottom)]"
  >
    <!-- Progress as a hairline along the top edge: present without competing with the tab bar. -->
    <div class="h-0.5 w-full bg-overlay">
      <div class="h-full bg-accent transition-[width] duration-500" :style="{ width: `${progress}%` }" />
    </div>

    <div class="mx-auto flex max-w-6xl items-center gap-3 px-3 py-2">
      <RouterLink
        :to="{ name: 'player', params: { slug: currentSlug } }"
        class="flex min-w-0 flex-1 items-center gap-3 no-underline text-canvas-foreground"
        data-testid="mini-player-open"
      >
        <img
          v-if="currentArtwork"
          :src="currentArtwork"
          alt=""
          class="h-9 w-9 shrink-0 rounded bg-canvas object-cover"
        />
        <div v-else class="h-9 w-9 shrink-0 rounded bg-canvas" />
        <span class="min-w-0 flex-1 truncate text-xs font-bold">{{ currentTitle ?? t('player.loading') }}</span>
      </RouterLink>

      <button
        type="button"
        data-testid="mini-player-toggle"
        class="flex h-11 w-11 shrink-0 items-center justify-center rounded-full text-canvas-foreground transition hover:bg-overlay"
        :aria-label="playing ? t('player.pause') : t('player.play')"
        @click="player.toggle()"
      >
        <svg viewBox="0 0 24 24" fill="currentColor" class="h-5 w-5" aria-hidden="true">
          <template v-if="playing"><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></template>
          <template v-else><path d="M8 5.5v13l11-6.5z" /></template>
        </svg>
      </button>
    </div>
  </div>
</template>
