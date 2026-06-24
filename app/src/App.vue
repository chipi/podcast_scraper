<script setup lang="ts">
import { onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, RouterView, useRouter } from 'vue-router'
import SkipLink from './components/SkipLink.vue'
import { useAuthStore } from './stores/auth'
import { useQueueStore } from './stores/queue'

const { t } = useI18n()
const auth = useAuthStore()
const queue = useQueueStore()
const router = useRouter()

onMounted(async () => {
  // Best-effort: resolve the session cookie to a user (null when signed out — reads still work).
  await auth.refresh()
  if (auth.isAuthenticated) await queue.ensureLoaded()
})

async function onSignOut(): Promise<void> {
  await auth.logout()
  await router.push({ name: 'catalog' })
}
</script>

<template>
  <SkipLink />
  <div class="min-h-screen bg-canvas text-canvas-foreground font-sans">
    <header class="border-b border-border px-5 py-4 flex items-center justify-between">
      <RouterLink :to="{ name: 'catalog' }" class="no-underline">
        <span class="lp-kicker block">{{ t('app.tagline') }}</span>
        <span class="font-display text-2xl font-extrabold tracking-tight">{{ t('app.title') }}</span>
      </RouterLink>
      <nav class="text-sm flex items-center gap-3">
        <template v-if="auth.isAuthenticated">
          <RouterLink :to="{ name: 'queue' }" class="text-muted no-underline">
            {{ t('queue.title') }}<span v-if="queue.count"> · {{ queue.count }}</span>
          </RouterLink>
          <span class="text-muted hidden sm:inline">
            {{ t('auth.signedInAs', { name: auth.user?.name }) }}
          </span>
          <button
            type="button"
            class="rounded-full border border-border px-4 py-2 font-bold text-canvas-foreground"
            @click="onSignOut"
          >
            {{ t('auth.signOut') }}
          </button>
        </template>
        <RouterLink
          v-else
          :to="{ name: 'login' }"
          class="rounded-full bg-accent px-4 py-2 font-bold text-accent-foreground no-underline"
        >
          {{ t('auth.signIn') }}
        </RouterLink>
      </nav>
    </header>

    <main id="main" tabindex="-1" class="px-5 py-6 outline-none">
      <RouterView />
    </main>
  </div>
</template>
