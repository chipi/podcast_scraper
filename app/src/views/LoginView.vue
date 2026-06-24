<script setup lang="ts">
/** Sign-in entry (C2/#1081 fleshes this out). Redirects into the OAuth flow on click. */
import { useI18n } from 'vue-i18n'
import { useAuthStore } from '../stores/auth'

const { t } = useI18n()
const auth = useAuthStore()
</script>

<template>
  <section class="max-w-md">
    <span class="lp-kicker">{{ t('app.tagline') }}</span>
    <h1 class="font-display text-3xl font-extrabold tracking-tight mt-1 mb-6">
      {{ t('app.title') }}
    </h1>
    <button
      type="button"
      class="rounded-full bg-accent px-6 py-3 font-bold text-accent-foreground"
      @click="auth.login()"
    >
      {{ t('auth.signIn') }}
    </button>
  </section>
</template>
