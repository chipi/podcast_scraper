<script setup lang="ts">
import { onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, RouterView, useRouter } from 'vue-router'
import SkipLink from './components/SkipLink.vue'
import NavIconLink from './components/NavIconLink.vue'
import PwaUpdateToast from './components/PwaUpdateToast.vue'
import { useAuthStore } from './stores/auth'
import { useQueueStore } from './stores/queue'
import { useFavoritesStore } from './stores/favorites'

const { t } = useI18n()
const auth = useAuthStore()
const queue = useQueueStore()
const favorites = useFavoritesStore()
const router = useRouter()

onMounted(async () => {
  // Best-effort: resolve the session cookie to a user (null when signed out — reads still work).
  await auth.refresh()
  if (auth.isAuthenticated) {
    await queue.ensureLoaded()
    await favorites.ensureLoaded()
  }
})

async function onSignOut(): Promise<void> {
  await auth.logout()
  await router.push({ name: 'catalog' })
}
</script>

<template>
  <SkipLink />
  <div class="min-h-screen bg-canvas text-canvas-foreground font-sans">
    <header class="border-b border-border px-5 py-4">
      <div class="mx-auto flex max-w-6xl items-center justify-between">
      <RouterLink :to="{ name: 'home' }" class="no-underline">
        <span class="lp-kicker block">{{ t('app.tagline') }}</span>
        <span class="font-display text-2xl font-extrabold tracking-tight">{{ t('app.title') }}</span>
      </RouterLink>
      <nav class="text-sm flex items-center gap-1.5">
        <NavIconLink :to="{ name: 'catalog' }" :label="t('nav.browse')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="h-5 w-5" aria-hidden="true">
            <circle cx="12" cy="12" r="10" />
            <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" />
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
            class="ml-1 rounded-full border border-border px-4 py-2 font-bold text-canvas-foreground"
            @click="onSignOut"
          >
            {{ t('auth.signOut') }}
          </button>
        </template>
        <template v-else>
          <RouterLink :to="{ name: 'login' }" class="text-muted no-underline">
            {{ t('auth.signIn') }}
          </RouterLink>
          <RouterLink
            :to="{ name: 'login', query: { mode: 'signup' } }"
            class="rounded-full bg-accent px-4 py-2 font-bold text-accent-foreground no-underline"
          >
            {{ t('auth.signUp') }}
          </RouterLink>
        </template>
      </nav>
      </div>
    </header>

    <main id="main" tabindex="-1" class="mx-auto max-w-6xl px-5 py-6 outline-none">
      <RouterView />
    </main>

    <PwaUpdateToast />
  </div>
</template>
