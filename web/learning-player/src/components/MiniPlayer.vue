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
 *
 * It renders `audioError` too, because audio can die while the listener is anywhere in the app and
 * this bar is the only thing on screen that claims to know about playback. Without it, a dead
 * source left a normal-looking play button that silently did nothing every time it was pressed —
 * the failure was reported on exactly one route, the one route where the user is least likely to be
 * when auto-advance hits a bad episode.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { storeToRefs } from 'pinia'
import { RouterLink, useRoute } from 'vue-router'
import { usePlayerStore } from '../stores/player'
import { artworkThumb } from '../utils/episode'
import RouteButton from './RouteButton.vue'

const { t } = useI18n()
const route = useRoute()
const player = usePlayerStore()
const {
  playing,
  currentTime,
  duration,
  currentSlug,
  currentTitle,
  currentShowTitle,
  currentArtwork,
  audioError,
} = storeToRefs(player)

const onPlayerPage = computed(() => route.name === 'player' && route.params.slug === currentSlug.value)
const visible = computed(() => !!currentSlug.value && !onPlayerPage.value)
/** While a Moments reel plays (operator 2026-10-10): "Moments · 3 / 8" in place of the show. */
const reelLabel = computed(() => {
  const r = player.reel
  return r ? t('moments.mini', { n: r.index + 1, total: r.moments.length }) : null
})
const openTo = computed(() =>
  player.reel
    ? { name: 'player', params: { slug: currentSlug.value }, query: { moments: '1' } }
    : { name: 'player', params: { slug: currentSlug.value } },
)
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
    <!-- Progress as a hairline along the top edge: present without competing with the tab bar.

         DRIVEN BY `transform: scaleX`, NOT `width` (2026-09-27). `width` is a LAYOUT property and
         this is bound to `currentTime`, so it re-laid out this bar roughly four times a second for
         as long as anything is playing, each tick animated over 500ms. `scaleX` is compositor-only:
         same picture, no layout, no style recalc.

         Worth doing on its own merits, and it is also half of the advisor's proposed fix for the
         scroll artifact the operator hit the same day — fixed chrome painting mid-list during
         momentum scroll. The theory there is a main-thread layer-tree commit racing the UI
         process's scrolling thread, and a fixed layer forcing a commit four times a second is the
         most obvious commit generator in the app. That is UNCONFIRMED — the operator could not
         reproduce the artifact on demand — so this is NOT claimed as the fix for it. It is claimed
         as removing a real per-tick layout from a bar that is on screen whenever audio plays.

         `origin-left` is what makes a scale read as a fill rather than a zoom from the centre.
         `motion-reduce` still drops the animation for users who asked software to stop moving. -->
    <div class="h-0.5 w-full bg-overlay">
      <div
        class="h-full w-full origin-left bg-accent transition-transform duration-500 motion-reduce:transition-none"
        :style="{ transform: `scaleX(${Math.max(0, Math.min(100, progress)) / 100})` }"
      />
    </div>

    <div class="mx-auto flex max-w-6xl items-center gap-3 px-3 py-2">
      <RouterLink
        :to="openTo"
        class="flex min-w-0 flex-1 items-center gap-3 no-underline text-canvas-foreground"
        data-testid="mini-player-open"
      >
        <img
          v-if="currentArtwork"
          :src="artworkThumb(currentArtwork) ?? undefined"
          alt=""
          class="h-9 w-9 shrink-0 rounded bg-canvas object-cover"
        />
        <div v-else class="h-9 w-9 shrink-0 rounded bg-canvas" />
        <span class="min-w-0 flex-1">
          <!-- Show above, episode below — the compact-row shape the rest of the app uses.
               It was ONE line of episode title, so a long one ended in an ellipsis having said
               nothing about whose show it was. The kicker is omitted rather than faked when the
               caller did not supply a show. -->
          <span v-if="reelLabel" class="lp-kicker block truncate text-accent" data-testid="mini-player-moments">{{ reelLabel }}</span>
          <span v-else-if="currentShowTitle" class="lp-kicker block truncate">{{ currentShowTitle }}</span>
          <span class="block truncate text-xs font-bold">{{ currentTitle ?? t('player.loading') }}</span>
          <!-- role=status so a screen-reader user hears it too; the icon change alone is silent. -->
          <span
            v-if="audioError"
            data-testid="mini-player-error"
            role="status"
            class="block truncate text-[0.6875rem] text-danger"
          >{{ t('player.audioErrorShort') }}</span>
        </span>
      </RouterLink>

      <!-- ONE right-aligned group, every control the same 32px circle on the same 12px gap
           (operator 2026-10-08: with the close button added, five uneven controls — play and close
           were 44px boxes — spread across the bar and squeezed the show and title to a few letters).
           12px is the smallest gap that keeps each control's 44px touch area (`lp-tap`) clear of its
           neighbour's. -->
      <div class="flex shrink-0 items-center gap-[12px]" data-testid="mini-player-actions">
      <!-- No save (heart) or board control here (operator 2026-10-09). The bar is for playback —
           what is on, play / pause, where the sound goes, close — and both live one tap away on the
           player page, which the bar opens. The queue control left for the same reason earlier:
           the masthead carries it at every width. -->

      <!-- Output routing, next to the transport it affects. Self-hides when the platform reports
           no route available, so this costs nothing on a device with nowhere to send audio. -->
      <RouteButton />

      <button
        type="button"
        data-testid="mini-player-toggle"
        class="lp-tap flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-canvas-foreground transition hover:bg-overlay disabled:opacity-40"
        :disabled="audioError"
        :aria-label="audioError ? t('player.audioErrorShort') : playing ? t('player.pause') : t('player.play')"
        @click="player.toggle()"
      >
        <svg v-if="audioError" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" class="h-5 w-5 text-danger" aria-hidden="true">
          <circle cx="12" cy="12" r="9" /><path d="M12 8v5" /><path d="M12 16h.01" />
        </svg>
        <svg v-else viewBox="0 0 24 24" fill="currentColor" class="h-5 w-5" aria-hidden="true">
          <template v-if="playing"><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></template>
          <template v-else><path d="M8 5.5v13l11-6.5z" /></template>
        </svg>
      </button>

      <!-- Close (operator 2026-10-07): stop and dismiss. Last, after play/pause, so a thumb
           reaching for the transport lands on play first. The position is kept, so the episode
           resumes where it was from anywhere else in the app. -->
      <button
        type="button"
        data-testid="mini-player-close"
        class="lp-tap flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-muted transition hover:bg-overlay hover:text-canvas-foreground"
        :aria-label="t('player.closeMini')"
        @click="player.close()"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" class="h-5 w-5" aria-hidden="true">
          <path d="M6 6l12 12M18 6L6 18" />
        </svg>
      </button>
      </div>
    </div>
  </div>
</template>
