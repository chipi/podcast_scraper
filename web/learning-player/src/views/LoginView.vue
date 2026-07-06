<script setup lang="ts">
/**
 * Auth entry (C2/#1081). One view, two modes (`?mode=signup` vs sign-in) — both drive the
 * same OAuth flow: with open signup the provider get-or-creates the account, so "sign up" and
 * "sign in" converge on the same redirect. A link toggles between the two framings.
 *
 * Dev (mock provider): a picker lets you sign in as a seeded user or a custom name (#1128).
 */
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { RouterLink, useRoute } from 'vue-router'
import { getDevUsers, type DevUser } from '../services/api'
import { useAuthStore } from '../stores/auth'

const { t } = useI18n()
const route = useRoute()
const auth = useAuthStore()

const isSignup = computed(() => route.query.mode === 'signup')

const devEnabled = ref(false)
const devUsers = ref<DevUser[]>([])
const custom = ref('')

onMounted(async () => {
  const { enabled, users } = await getDevUsers()
  devEnabled.value = enabled
  devUsers.value = users
})

function signInCustom(): void {
  const name = custom.value.trim()
  if (name) auth.login(name)
}
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

    <!-- Dev (mock provider): pick a predefined user, or type a custom one. -->
    <div v-if="devEnabled">
      <div v-if="devUsers.length" class="space-y-1.5" data-testid="dev-user-list">
        <p class="text-xs font-medium uppercase tracking-wide text-muted">Sign in as</p>
        <button
          v-for="u in devUsers"
          :key="u.hint"
          type="button"
          class="flex w-full items-center justify-between rounded-xl border border-border px-4 py-2.5 text-left hover:bg-surface"
          :data-testid="`dev-user-${u.hint}`"
          @click="auth.login(u.hint)"
        >
          <span class="font-bold">{{ u.name }}</span>
          <span class="rounded-full bg-surface px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted">{{ u.role }}</span>
        </button>
      </div>

      <form class="mt-4 flex gap-2" @submit.prevent="signInCustom">
        <input
          v-model="custom"
          type="text"
          placeholder="or a custom name…"
          class="min-w-0 flex-1 rounded-full border border-border bg-canvas px-4 py-2 text-sm"
          data-testid="dev-custom-input"
        />
        <button
          type="submit"
          :disabled="!custom.trim()"
          class="rounded-full bg-accent px-5 py-2 font-bold text-accent-foreground disabled:opacity-50"
          data-testid="dev-custom-submit"
        >
          {{ t('auth.signIn') }}
        </button>
      </form>
      <p class="mt-3 text-xs text-muted">Dev sign-in (mock OAuth) — picked users keep their role.</p>
    </div>

    <!-- Real provider: the normal sign-in button. -->
    <button
      v-else
      type="button"
      class="rounded-full bg-accent px-6 py-3 font-bold text-accent-foreground"
      data-testid="signin-button"
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
