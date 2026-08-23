<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, RouterView, useRouter } from 'vue-router'
import SkipLink from './components/SkipLink.vue'
import BottomNav from './components/BottomNav.vue'
import MiniPlayer from './components/MiniPlayer.vue'
import NavIconLink from './components/NavIconLink.vue'
import PwaUpdateToast from './components/PwaUpdateToast.vue'
import TierSwitch from './components/TierSwitch.vue'
import { useAuthStore } from './stores/auth'
import { useQueueStore } from './stores/queue'
import { usePlayerStore } from './stores/player'
import { getAudioSource, getEpisode, putPlayback } from './services/api'
import { episodeArtwork } from './utils/episode'
import type { NextUp } from './stores/player'
import { useFavoritesStore } from './stores/favorites'
import { initNativeAuth } from './services/native'

const { t } = useI18n()
const auth = useAuthStore()
const queue = useQueueStore()
const player = usePlayerStore()
const favorites = useFavoritesStore()
const router = useRouter()

async function hydrateUser(): Promise<void> {
  if (auth.isAuthenticated) {
    await queue.ensureLoaded()
    await favorites.ensureLoaded()
  }
}

/**
 * Queue-advance lives in the shell, not the store or the view (#1587).
 *
 * The store must not import the queue or the API — it would couple playback state to data fetching.
 * The view cannot own it any more, because audio outlives the view. The shell is mounted for the
 * whole session, so it is the right place.
 *
 * Resolved ON DEMAND at `ended`, not cached at load. The first version answered "what plays next?"
 * when the current episode started and never asked again, so "Play next", a reorder, or a first
 * queue item added mid-listen all had no effect — and mid-listen is when every one of those inputs
 * happens. It also fetched the audio URL an hour before using it, which a signed origin URL may not
 * survive.
 *
 * Metadata comes along for the ride because auto-advance runs with no view mounted: without a title
 * and artwork here, the mini-player reads "Loading…" for the whole episode and the lock screen
 * keeps showing the previous one.
 */
async function resolveNextUp(): Promise<NextUp | null> {
  const slug = player.currentSlug
  if (!slug) return null
  const next = queue.nextAfter(slug)
  if (!next) return null
  const [src, detail] = await Promise.all([
    getAudioSource(next).catch(() => null),
    getEpisode(next).catch(() => null),
  ])
  if (!src?.url) return null
  return {
    slug: next,
    url: src.url,
    title: detail?.title ?? null,
    artwork: detail ? (episodeArtwork(detail) ?? null) : null,
  }
}
player.setAdvanceResolver(resolveNextUp)

/**
 * Position persistence is wired here for the same reason queue-advance is: the store must not
 * import the API, and the view cannot own it because audio outlives the view.
 *
 * The store decides WHEN and for WHICH episode (it holds both halves of that pair); the shell only
 * supplies the writer. A rejected save is swallowed — a lost position is a small, self-correcting
 * annoyance, and a signed-out user 401s on every tick.
 */
player.setPositionPersister((slug, seconds, finished) => {
  void putPlayback(slug, seconds, finished).catch(() => {})
})

onMounted(async () => {
  // Native (#1310): rehydrate the stored bearer token + register the OAuth deep-link handler BEFORE
  // the first refresh so a saved session is picked up; the callback re-refreshes after a fresh login.
  await initNativeAuth(async () => {
    await auth.refresh()
    await hydrateUser()
  })
  // Best-effort: resolve the session (cookie on web, bearer token on native) to a user — null when
  // signed out, reads still work.
  await auth.refresh()
  await hydrateUser()
})

/**
 * Reserve exactly what is on screen: the tab bar (mobile only) plus the mini-player (only when
 * something is loaded), plus the home-indicator inset.
 */
const mainBottomPadding = computed(() =>
  player.currentSlug
    ? 'pb-[calc(7.25rem+env(safe-area-inset-bottom))] sm:pb-[calc(5rem+env(safe-area-inset-bottom))]'
    : 'pb-[calc(4rem+env(safe-area-inset-bottom))] sm:pb-6',
)

async function onSignOut(): Promise<void> {
  await auth.logout()
  await router.push({ name: 'catalog' })
}
</script>

<template>
  <SkipLink />
  <div class="min-h-dvh bg-canvas text-canvas-foreground font-sans">
    <!-- dvh (not vh) avoids the iOS 100vh over-report; safe-area top so the nav clears the
         notch / Dynamic Island, and side insets for landscape rounded corners. -->
    <header class="border-b border-border px-5 pb-4 pt-[max(1rem,env(safe-area-inset-top))] pl-[max(1.25rem,env(safe-area-inset-left))] pr-[max(1.25rem,env(safe-area-inset-right))]">
      <div class="mx-auto flex max-w-6xl items-center justify-between">
      <RouterLink :to="{ name: 'home' }" class="no-underline">
        <span class="lp-kicker block">{{ t('app.tagline') }}</span>
        <span class="font-display text-2xl font-extrabold tracking-tight">{{ t('app.title') }}</span>
      </RouterLink>
      <nav class="text-sm flex items-center gap-1.5">
        <TierSwitch />
        <!--
          Icon links are DESKTOP-only (#1594 follow-up).

          The bottom tab bar shipped without hiding these, so a phone carried two navigation systems
          at once: Search appeared twice, and Library/Profile were reachable from both the top and
          the bottom of the same screen. Two navs is worse than either one — it makes the app feel
          like two designs stacked, and it wastes the scarcest space on a phone.

          Browse lives only here, and that is fine: it is a corpus index, not a daily destination
          (the reason it did not take a tab), and Home carries a "Browse all →" link plus the
          catalogue link in the empty shows state. The auth buttons below stay visible at every
          width — signing in is not a tab.
        -->
        <span class="hidden items-center gap-1.5 sm:flex">
        <NavIconLink :to="{ name: 'catalog' }" :label="t('nav.browse')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-5 w-5" aria-hidden="true">
            <circle cx="12" cy="12" r="10" />
            <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" />
          </svg>
        </NavIconLink>
        <!-- Search is the differentiator — corpus-wide semantic search with jump-to-moment, which
             neither Spotify nor Apple Podcasts offers. It had exactly ONE entry point (the Home
             search box), so from the catalogue, player, library or a show page there was no way to
             reach it at all (#1588). Public, like Browse: reads are open. -->
        <NavIconLink :to="{ name: 'search' }" :label="t('nav.search')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-5 w-5" aria-hidden="true">
            <circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" />
          </svg>
        </NavIconLink>
        <template v-if="auth.isAuthenticated">
          <NavIconLink :to="{ name: 'library' }" :label="t('library.title')">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-5 w-5" aria-hidden="true">
              <path d="m16 6 4 14" /><path d="M12 6v14" /><path d="M8 8v12" /><path d="M4 4v16" />
            </svg>
          </NavIconLink>
          <NavIconLink :to="{ name: 'profile' }" :label="auth.user?.name || t('profile.title')">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-5 w-5" aria-hidden="true">
              <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" />
            </svg>
          </NavIconLink>
        </template>
        </span>
        <template v-if="auth.isAuthenticated">
          <button
            type="button"
            class="shrink-0 whitespace-nowrap rounded-full border border-border px-4 py-2 font-bold text-canvas-foreground transition hover:bg-overlay"
            @click="onSignOut"
          >
            {{ t('auth.signOut') }}
          </button>
        </template>
        <template v-else>
          <RouterLink
            :to="{ name: 'login' }"
            class="shrink-0 whitespace-nowrap rounded-full border border-border px-4 py-2 font-bold text-canvas-foreground no-underline transition hover:bg-overlay"
          >
            {{ t('auth.signIn') }}
          </RouterLink>
          <RouterLink
            :to="{ name: 'login', query: { mode: 'signup' } }"
            class="shrink-0 whitespace-nowrap rounded-full bg-accent px-4 py-2 font-bold text-accent-foreground no-underline"
          >
            {{ t('auth.signUp') }}
          </RouterLink>
        </template>
      </nav>
      </div>
    </header>

    <!--
      Bottom padding is COMPUTED, not a constant.

      It shipped as `pb-24 sm:pb-6` — 96px on mobile against a tab bar (~52px) plus a mini-player
      (~62px) = ~114px before safe-area insets, and 24px on desktop against a 62px mini-player. So
      the last card on every page sat under the transport whenever anything was playing, on both
      viewports. The string check in mobile-invariants.test.ts asserted the classes existed, which
      is exactly what made it invisible: the classes were present and the geometry was wrong.

      `scroll-padding-bottom` matches, so keyboard-tabbing to the last element scrolls it clear of
      the bars instead of underneath them.
    -->
    <main
      id="main"
      tabindex="-1"
      class="mx-auto max-w-6xl px-5 pt-6 outline-none"
      :class="mainBottomPadding"
      :style="{ scrollPaddingBottom: '8rem' }"
    >
      <RouterView />
    </main>

    <MiniPlayer />
    <BottomNav />

    <PwaUpdateToast />
  </div>
</template>
