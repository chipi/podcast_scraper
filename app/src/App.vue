<script setup lang="ts">
import { onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, RouterView } from 'vue-router'
import SkipLink from './components/SkipLink.vue'
import { useAuthStore } from './stores/auth'

const { t } = useI18n()
const auth = useAuthStore()

onMounted(() => {
  // Best-effort: resolve the session cookie to a user (null when signed out — reads still work).
  void auth.refresh()
})
</script>

<template>
  <SkipLink />
  <div class="min-h-screen bg-canvas text-canvas-foreground font-sans">
    <header class="border-b border-border px-5 py-4 flex items-center justify-between">
      <RouterLink :to="{ name: 'catalog' }" class="no-underline">
        <span class="lp-kicker block">{{ t('app.tagline') }}</span>
        <span class="font-display text-2xl font-extrabold tracking-tight">{{ t('app.title') }}</span>
      </RouterLink>
      <nav class="text-sm">
        <span v-if="auth.isAuthenticated" class="text-muted">
          {{ t('auth.signedInAs', { name: auth.user?.name }) }}
        </span>
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
