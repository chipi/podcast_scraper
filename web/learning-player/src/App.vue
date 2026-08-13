<script setup lang="ts">
import { onMounted, watch } from 'vue'
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
import { getAudioSource } from './services/api'
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
 * whole session, so it is the right place. Resolving the URL is async while `ended` is sync, so the
 * next episode's source is prefetched as soon as the current one loads.
 */
let nextUp: { slug: string; url: string; title?: string | null } | null = null
watch(
  () => player.currentSlug,
  async (slug) => {
    nextUp = null
    if (!slug) return
    const next = queue.nextAfter(slug)
    if (!next) return
    const src = await getAudioSource(next).catch(() => null)
    if (src?.url) nextUp = { slug: next, url: src.url }
  },
  { immediate: true },
)
player.setAdvanceResolver(() => nextUp)

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
          <button
            type="button"
            class="ml-1 shrink-0 whitespace-nowrap rounded-full border border-border px-4 py-2 font-bold text-canvas-foreground transition hover:bg-overlay"
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

    <!-- pb-24 on mobile clears the fixed bottom nav (#1594); sm:pb-6 restores the desktop rhythm
         where the bar is hidden. -->
    <main id="main" tabindex="-1" class="mx-auto max-w-6xl px-5 pt-6 pb-24 outline-none sm:pb-6">
      <RouterView />
    </main>

    <MiniPlayer />
    <BottomNav />

    <PwaUpdateToast />
  </div>
</template>
