<script setup lang="ts">
/**
 * Auth entry (C2/#1081). One view, two modes (`?mode=signup` vs sign-in) — both drive the
 * same OAuth flow: with open signup the provider get-or-creates the account, so "sign up" and
 * "sign in" converge on the same redirect. A link toggles between the two framings.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRoute } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const { t } = useI18n()
const route = useRoute()
const auth = useAuthStore()

const isSignup = computed(() => route.query.mode === 'signup')
</script>

<template>
  <section class="max-w-md">
    <span class="lp-kicker">{{ t('app.tagline') }}</span>
    <h1 class="mb-2 mt-1 font-display text-3xl font-extrabold tracking-tight">
      {{ isSignup ? t('auth.signupTitle') : t('auth.loginTitle') }}
    </h1>
    <p class="mb-6 text-sm text-muted">
      {{ isSignup ? t('auth.signupTagline') : t('auth.loginTagline') }}
    </p>

    <button
      type="button"
      class="rounded-full bg-accent px-6 py-3 font-bold text-accent-foreground"
      @click="auth.login()"
    >
      {{ isSignup ? t('auth.signUp') : t('auth.signIn') }}
    </button>

    <p class="mt-5 text-sm text-muted">
      <template v-if="isSignup">
        {{ t('auth.haveAccount') }}
        <RouterLink :to="{ name: 'login' }" class="font-bold text-accent no-underline">
          {{ t('auth.signIn') }}
        </RouterLink>
      </template>
      <template v-else>
        {{ t('auth.newHere') }}
        <RouterLink :to="{ name: 'login', query: { mode: 'signup' } }" class="font-bold text-accent no-underline">
          {{ t('auth.signUp') }}
        </RouterLink>
      </template>
    </p>
  </section>
</template>
