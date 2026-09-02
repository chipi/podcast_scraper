<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, RouterView, useRouter } from 'vue-router'
import SkipLink from './components/SkipLink.vue'
import BottomNav from './components/BottomNav.vue'
import MiniPlayer from './components/MiniPlayer.vue'
import NavIconLink from './components/NavIconLink.vue'
import PwaUpdateToast from './components/PwaUpdateToast.vue'
import TierSwitch from './components/TierSwitch.vue'
import BrandGlyph from './components/BrandGlyph.vue'
import AppSplash from './components/AppSplash.vue'
import { SplashScreen } from '@capacitor/splash-screen'
import { useAuthStore } from './stores/auth'
import { useQueueStore } from './stores/queue'
import { usePlayerStore } from './stores/player'
import { getAudioSource, getEpisode, getPlayback, putPlayback } from './services/api'
import { localArtworkFor, localSourceFor, refreshLocalUris } from './services/downloads'
import { ANON_NAMESPACE, useDownloadsStore } from './stores/downloads'
import { startDownloadScheduler } from './services/downloadScheduler'
import {
  ANON_NAMESPACE as POSITIONS_ANON,
  flushPendingPositions,
  hydratePositions,
  recordPosition,
} from './services/playbackPositions'
import { Network } from '@capacitor/network'
import { episodeArtwork } from './utils/episode'
import { deriveShowAccent } from './theme/accent'
import type { NextUp } from './stores/player'
import { useFavoritesStore } from './stores/favorites'
import { useUserPreferencesStore } from './stores/userPreferences'
import { initNativeAuth, isNative } from './services/native'

// Bottom-nav tab views to keep mounted across navigation (matches each view's `name`). Detail views
// are omitted so they stay fresh per-route. Keep in sync with router/index.ts tab routes.
const KEEP_ALIVE_TABS = ['HomeView', 'SearchView', 'LibraryView', 'ProfileView', 'CatalogView', 'BrowseView']

const { t } = useI18n()
const auth = useAuthStore()
const queue = useQueueStore()
const player = usePlayerStore()
const favorites = useFavoritesStore()
const router = useRouter()

// Native launch overlay: shown briefly after the native splash hands off, while the app + its first
// data load behind it (prolongs the branded splash + surfaces the build version). Native only.
const booting = ref(isNative())
const appVersion = `v${__APP_VERSION__} · ${(__BUILD_SHA__ || '').slice(0, 7)}`

async function hydrateUser(): Promise<void> {
  if (!auth.isAuthenticated) return
  // allSettled, not sequential awaits (#1906): offline these reject, and an unhandled rejection
  // here aborted the rest of boot. Each store keeps whatever it already had on failure, which is
  // the "a failed refresh must not delete the old stuff" rule.
  await Promise.allSettled([queue.ensureLoaded(), favorites.ensureLoaded()])
  // Preferences hydrate only once a session exists (they 401 otherwise); do it here, right after
  // auth resolves, so a signed-in user's synced prefs are loaded without the signed-out boot 401.
  void useUserPreferencesStore().hydrate()
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
  // Offline, both calls fail — but if the next episode is on disk it can still play. Without
  // this, auto-advance stopped at the end of every episode even with the whole queue downloaded
  // (#1905/#1906), which is exactly the journey offline downloads exist for.
  const entry = useDownloadsStore().entry(next)
  const localSrc = localSourceFor(next)
  if (!src?.url && !localSrc) return null
  return {
    slug: next,
    url: src?.url ?? localSrc ?? '',
    title: detail?.title ?? entry?.title ?? null,
    // Offline the API gave us no detail; the downloaded copy carries its own art.
    artwork: (detail ? episodeArtwork(detail) : null) ?? localArtworkFor(next),
  }
}
player.setAdvanceResolver(resolveNextUp)
// Downloaded episodes play from disk (#1905). Same injection rationale as the advance resolver:
// the player store must not know about downloads.
player.setSourceResolver(localSourceFor)

// A sign-out or account switch mid-session must swap the registry too, or the previous account's
// downloads stay on screen for whoever signs in next.
watch(
  () => auth.user?.user_id ?? ANON_NAMESPACE,
  (ns) => {
    // Per-user in-memory state must not survive an identity change: without this, signing out and
    // in as someone else left A's queue and favourites on screen until a reload (#1906).
    queue.$reset()
    favorites.$reset()
    void useDownloadsStore().setNamespace(ns)
    // Positions are per-account for the same reason the registry is; leaving them behind let one
    // account's progress be flushed under another's session (#1906).
    void hydratePositions(ns)
  },
)

/**
 * Position persistence is wired here for the same reason queue-advance is: the store must not
 * import the API, and the view cannot own it because audio outlives the view.
 *
 * The store decides WHEN and for WHICH episode (it holds both halves of that pair); the shell only
 * supplies the writer. A rejected save is swallowed — a lost position is a small, self-correcting
 * annoyance, and a signed-out user 401s on every tick.
 */
player.setPositionPersister((slug, seconds, finished) => {
  // Also kept on the device (#1906): GET /playback fails offline, so without a local copy every
  // downloaded episode resumes at 0. A failed server write marks the position pending, and the
  // reconnect handler below pushes it.
  void putPlayback(slug, seconds, finished)
    .then(() => recordPosition(slug, seconds, finished, true))
    .catch(() => recordPosition(slug, seconds, finished, false))
})

/**
 * Push positions recorded while offline.
 *
 * Reads the server's value first so a phone coming back from airplane mode cannot overwrite
 * progress made on another device while it was away (#1906).
 */
function pushPendingPositions(): void {
  void flushPendingPositions(
    (slug, seconds, finished) => putPlayback(slug, seconds, finished),
    async (slug) => {
      const p = await getPlayback(slug)
      return p
        ? { seconds: p.position_seconds, finished: !!p.finished, updatedAt: p.updated_at }
        : null
    },
  )
}

void Network.addListener('networkStatusChange', (status) => {
  if (!status.connected) return
  // One offline blip used to write off preferences sync for the whole session (#1906).
  const prefs = useUserPreferencesStore()
  prefs.resetAvailability()
  void prefs.hydrate()
  pushPendingPositions()
})

// Per-show adaptive accent (UXS-011, #1598): `--lp-accent` tracks the current episode's artwork,
// contrast-clamped, falling back to the brand default when there is no artwork or extraction fails.
// Driven from the store because audio (and thus the "current show") outlives any single view.
watch(
  () => player.currentArtwork,
  (url) => {
    void deriveShowAccent(url)
  },
  { immediate: true },
)

onMounted(async () => {
  // The shell is mounted — hand off from the native launch splash to our web overlay NOW (seamless,
  // same image), without blocking on the network below. launchAutoHide:false means we own the
  // dismissal; fire-and-forget. Then prolong the branded overlay ~1.8s so the app + its first data
  // load behind it, and fade it out. No-op on web.
  if (isNative()) {
    // Hide the native splash only AFTER our web overlay (AppSplash) has actually painted — otherwise
    // the native splash lifts on a frame where the overlay isn't on screen yet, so the app/white
    // flashes through and then the (identical) web splash appears: the "glitch, then splash again"
    // double-splash (#7). onMounted runs before paint; two rAFs guarantee the overlay is committed to
    // the screen first, so the hand-off is seamless (same image, no flash).
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        void SplashScreen.hide().catch(() => {})
      }),
    )
    window.setTimeout(() => {
      booting.value = false
    }, 1800)
  }
  // Native (#1310): rehydrate the stored bearer token + register the OAuth deep-link handler BEFORE
  // the first refresh so a saved session is picked up; the callback re-refreshes after a fresh login.
  await initNativeAuth(async () => {
    await auth.refresh()
    await hydrateUser()
  })
  // Paint the last known identity first so an offline launch is signed in immediately, then
  // revalidate. `refresh()` no longer throws, so a dead network cannot abort boot (#1906).
  await auth.hydrateFromDevice()
  await auth.refresh()
  await hydrateUser()
  // Downloaded files get fresh URIs (iOS regenerates the container UUID on app update) and any
  // record whose file vanished is dropped. Fire-and-forget: nothing on screen waits for it.
  // Point the downloads registry at this account BEFORE anything reads it (#1905): the list of
  // downloaded episodes is listening history and must not cross accounts on a shared device.
  await useDownloadsStore().setNamespace(auth.user?.user_id ?? ANON_NAMESPACE)
  await hydratePositions(auth.user?.user_id ?? POSITIONS_ANON)
  // The common case — listen offline, kill the app, relaunch online — fires NO network status
  // change, so without a boot flush those writes sat pending forever (#1906).
  pushPendingPositions()
  void refreshLocalUris()
  // L1 download triggers: network change while foregrounded, and app resume (#1905).
  void startDownloadScheduler()
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
  <!-- Native launch overlay (prolonged branded splash + build version); fades out once booting. -->
  <Transition name="splash-fade">
    <AppSplash v-if="booting" :version="appVersion" />
  </Transition>
  <div class="min-h-dvh bg-canvas text-canvas-foreground font-sans">
    <!-- dvh (not vh) avoids the iOS 100vh over-report; safe-area top so the nav clears the
         notch / Dynamic Island, and side insets for landscape rounded corners. -->
    <header class="border-b border-border px-5 pb-2 pt-[max(0.55rem,env(safe-area-inset-top))] pl-[max(1.25rem,env(safe-area-inset-left))] pr-[max(1.25rem,env(safe-area-inset-right))]">
      <div class="mx-auto flex max-w-6xl items-center justify-between gap-3">
      <RouterLink :to="{ name: 'home' }" class="flex min-w-0 items-center gap-2 no-underline">
        <BrandGlyph class="h-7 w-auto shrink-0 sm:h-9" />
        <span class="block min-w-0">
          <!-- Tiny purple tagline above the name on phones — kept at 9px/leading-none so it adds
               almost no header height, but the line is still there. The desktop lockup keeps the
               larger ember kicker (lp-kicker) below the glyph. -->
          <span class="block whitespace-nowrap pl-[3px] text-[8px] font-bold uppercase leading-none tracking-[0.035em] text-topic sm:hidden">{{ t('app.tagline') }}</span>
          <span class="lp-kicker hidden sm:block">{{ t('app.tagline') }}</span>
          <span class="block whitespace-nowrap font-display text-[21px] font-extrabold leading-tight tracking-tight sm:text-2xl">{{ t('app.title') }}</span>
        </span>
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
            class="shrink-0 whitespace-nowrap rounded-full border border-border px-3.5 py-1.5 text-sm font-bold text-canvas-foreground no-underline transition hover:bg-overlay sm:px-4 sm:py-2 sm:text-base"
          >
            {{ t('auth.signIn') }}
          </RouterLink>
          <!-- Sign up is redundant in the phone header (the login screen has a sign-up toggle); hide it
               on mobile so the brand name fits on one compact row. Shown on wider screens. -->
          <RouterLink
            :to="{ name: 'login', query: { mode: 'signup' } }"
            class="hidden shrink-0 whitespace-nowrap rounded-full bg-accent px-4 py-2 font-bold text-accent-foreground no-underline sm:inline-flex"
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
      <!-- Keep the bottom-nav tabs alive so returning to Home/Search/etc. does NOT factory-refresh
           (re-run onMounted + re-fetch). Detail views (player/podcast/topic/person) are NOT included,
           so each still mounts fresh. Volatile sections refresh via onActivated in their view. -->
      <RouterView v-slot="{ Component }">
        <keep-alive :include="KEEP_ALIVE_TABS">
          <component :is="Component" />
        </keep-alive>
      </RouterView>
    </main>

    <MiniPlayer />
    <BottomNav />

    <PwaUpdateToast />
  </div>
</template>

<style>
/* Launch overlay fade-out (leave only — it starts visible and is dismissed once booting). */
.splash-fade-leave-active {
  transition: opacity 0.45s ease;
}
.splash-fade-leave-to {
  opacity: 0;
}
</style>
